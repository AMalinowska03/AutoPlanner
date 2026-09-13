import copy
import io
import os
import shelve
from datetime import datetime, timedelta
import time
from typing import Optional, List

import torch

from data.DbHelper import MonthSimulationSession, sort_tasks_by_deadline_and_priority
# from data.DbHelper import Repository
from data.DbModels import User, Task, BreakTask
from simulation.UserSimulator import UserSimulator
from models.PPOEnv import PPOPlannerEnv, make_wrapped_env
from spinup.algos.pytorch.ppo.ppo import ppo, core

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DisruptorsMap = dict[int, list[tuple[float, Task]]]

class PPOPlanner:
    def __init__(self, user: Optional[User] = None):
        self.user = user
        self.base_storage_path = "db/models_store.db"
        self.storage_path = f"db/models_store_u{user.id}.db" if user else self.base_storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.repository = None

    def pretrain(self, all_users: List[User], pretrain_tasks: dict, epochs: int = 150):
        def env_fn():
            return make_wrapped_env(users_pool=all_users, tasks=pretrain_tasks)

        print("PPO ------ Start pretraining...")
        ppo(
            env_fn=env_fn,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=2000,
            epochs=epochs,
            pi_lr=1e-4,
            vf_lr=5e-4,
            target_kl=0.025,
            logger_kwargs=dict(output_dir="./PPOGenerated/pretrain", exp_name="pretrain")
        )
        # save model to NoSQL
        loaded_model = torch.load("./PPOGenerated/pretrain/pyt_save/model.pt", map_location=device, weights_only=False)
        self._save_to_storage("ppo_base_pretrained", loaded_model)

    def finetune_user(self, user: User, finetune_tasks: dict, epochs: int = 15):
        import shelve, io
        base_model = None
        with shelve.open(self.storage_path) as db:
            if "ppo_base_pretrained" in db:
                buffer = io.BytesIO(db["ppo_base_pretrained"])
                base_model = torch.load(buffer, map_location=device, weights_only=False)
            else:
                print("[Warning] No model 'ppo_base_pretrained'. Training from scratch.")

        def pretrained_actor_critic(obs_space, act_space, **kwargs):
            ac = core.MLPActorCritic(obs_space, act_space, **kwargs)

            if base_model is not None:
                ac.load_state_dict(base_model.state_dict())

            return ac

        def env_fn():
            return make_wrapped_env(users_pool=[user], tasks=finetune_tasks)

        print(f"PPO ------ Finetuning for user #{user.id}...")
        ppo(
            env_fn=env_fn,
            actor_critic=pretrained_actor_critic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=1000,
            epochs=epochs,
            pi_lr=5e-5,
            vf_lr=2e-4,
            target_kl=0.02,
            train_pi_iters=40,
            train_v_iters=40,
            logger_kwargs=dict(output_dir=f"./PPOGenerated/finetune_u{user.id}", exp_name=f"finetune_u{user.id}")
        )
        loaded_model = torch.load(f"./PPOGenerated/finetune_u{user.id}/pyt_save/model.pt", map_location=device, weights_only=False)
        self._save_to_storage(f"ppo_user_{user.id}_finetuned", loaded_model)

    def plan_and_simulate_month(self, session: MonthSimulationSession, user: User, month_tasks: List[Task],
                                disruptors_map: Optional[DisruptorsMap] = None, start_date=datetime(2027, 1, 4)):
        """
        Simulates 1 month of work saving plan, it's re-plans and execution to db
        Manages disruptor injections with re-planning.
        :param user: user that works on given plan
        :param month_tasks: sorted by deadline and priority tasks for given month of work
                (one phase order in phase "online" or "disruptions")
        :param group_id: id for plan group to identify all re-plans
        :param disruptors_map: list of possible disruptor tasks for given month of work
                with injection date+time specified
        :param start_date: start date for given month of work
        :param phase: name of experiment phase
        :param phase_order: month inside phase
        :return:
        """
        disr_map = copy.deepcopy(disruptors_map)
        # self.repository = Repository(user, phase, phase_order, start_date)
        model_key = f"ppo_user_{user.id}_active"
        model_bytes = None
        with shelve.open(self.storage_path) as user_db:
            if model_key in user_db:
                model_bytes = user_db[model_key]
        if model_bytes is None:
            with shelve.open(self.base_storage_path) as db:
                if model_key not in db:
                    base_key = f"ppo_user_{user.id}_finetuned"
                    if base_key not in db:
                        raise ValueError(f"No finetuned model for user {user.id}. Run finetuning first.")
                    model_bytes = db[base_key]

            with shelve.open(self.storage_path) as user_db:
                user_db[model_key] = model_bytes

        buffer = io.BytesIO(model_bytes)
        ac_model = torch.load(buffer, map_location=device, weights_only=False)
        ac_model.eval()

        training_scenarios = []
        # environment used only to plan, not simulate
        raw_env = PPOPlannerEnv(users_pool=[user], divided_tasks={0: []}, max_tasks_count=50, planning_mode=True,
                                start_day=start_date)
        env = make_wrapped_env(raw_env=raw_env)
        simulator = UserSimulator(user)

        current_generation = 0
        remaining_to_plan = copy.deepcopy(month_tasks)
        remaining_to_plan = sort_tasks_by_deadline_and_priority(remaining_to_plan)
        previous_plan_state = []

        sim_day = 0
        sim_time = raw_env.work_start_hour

        disruption_occurrence_time = None
        time_since_last_break = 0.0
        total_break_time_today = 0.0
        while remaining_to_plan and sim_day < raw_env.total_days:
            print(f"PPO ------ Planning: user {user.id} | generation: {current_generation} | start date: {start_date}")
            current_plan, gen_time = self._generate_plan(
                env, raw_env, ac_model, remaining_to_plan, previous_plan_state, sim_day, sim_time,
                time_since_last_break, total_break_time_today, raw_env.last_task_type
            )

            # save plan to db
            session.record_plan(
                planned_tasks=current_plan,
                generation=current_generation,
                generating_time=gen_time,
                disruption_time=disruption_occurrence_time
            )
            disruption_occurrence_time = None

            replan_needed = False

            # go through all planned tasks until they are possible to be completed
            while current_plan:
                plan_item = current_plan.pop(0)
                calendar_days_passed = sim_day + (sim_day // 5) * 2

                # execute plan item
                if plan_item.get("is_break"):
                    simulator.process_break(duration=plan_item["duration"], time=sim_time)
                    current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                            minutes=int((sim_time % 1) * 60))
                    break_obj = BreakTask(duration_hours=plan_item["duration"])
                    session.record_execution(
                        task=break_obj,
                        planned_start=plan_item["start_time"],
                        planned_end=plan_item["end_time"],
                        actual_start=current_sim_dt,
                        actual_end=current_sim_dt + timedelta(hours=plan_item["duration"]),
                    )
                    sim_time += plan_item["duration"]
                    time_since_last_break = 0.0
                    total_break_time_today += plan_item["duration"]
                    raw_env.last_task_type = None
                else:
                    task = plan_item["task"]
                    actual_dur, end_time, energy = simulator.execute_task(task, sim_time, raw_env.last_task_type)
                    current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                            minutes=int((sim_time % 1) * 60))

                    session.record_execution(
                        task=task,
                        planned_start=plan_item["start_time"],
                        planned_end=plan_item["end_time"],
                        actual_start=current_sim_dt,
                        actual_end=current_sim_dt + timedelta(hours=actual_dur),
                        energy=energy
                    )
                    raw_env.last_task_type = task.type
                    sim_time = end_time
                    time_since_last_break += actual_dur

                #  check if disruptor is supposed to appear
                if disr_map and sim_day in disr_map:
                    pending_disruptors = disr_map[sim_day]
                    disruptors_appeared = []
                    while pending_disruptors and pending_disruptors[0][0] <= sim_time:
                        disrupt_time, disruptor_task = pending_disruptors.pop(0)
                        disruptors_appeared.append((disrupt_time, disruptor_task))
                    if disruptors_appeared:
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        remaining_to_plan.extend(disruptor_task for _, disruptor_task in disruptors_appeared)
                        remaining_to_plan = sort_tasks_by_deadline_and_priority(remaining_to_plan)
                        first_disruption_time = min(disrupt_time for disrupt_time, _ in disruptors_appeared)
                        dh = int(first_disruption_time)
                        dm = int((first_disruption_time - dh) * 60)
                        disruption_occurrence_time = start_date + timedelta(days=calendar_days_passed, hours=dh,
                                                                            minutes=dm)
                        previous_plan_state = copy.deepcopy(current_plan)
                        replan_needed = True
                        training_scenarios.append({
                            "day": sim_day,
                            "time": sim_time,
                            "previous_plan": copy.deepcopy(previous_plan_state),
                            "remaining_tasks": copy.deepcopy(remaining_to_plan),
                            "time_since_last_break": time_since_last_break,
                            "total_break_time": total_break_time_today,
                            "last_task_type": raw_env.last_task_type,
                            "task_energy_usage": simulator.task_energy_usage,
                            "energy_debt": simulator.energy_debt,
                            "start_energy": simulator.start_energy,
                        })
                        break
                    else:
                        disruption_occurrence_time = None
                else:
                    disruption_occurrence_time = None

                current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                        minutes=int((sim_time % 1) * 60))

                # if we finish tasks for the day earlier we might want to re-plan to not waste work day
                if current_plan:
                    next_item = current_plan[0]
                    next_start_dt = next_item["start_time"]

                    # re-plan if next task is in next day, and we still have over 0.5h of work day
                    if next_start_dt.date() > current_sim_dt.date() and raw_env.work_end_hour - sim_time >= 0.5:
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        remaining_to_plan = sort_tasks_by_deadline_and_priority(remaining_to_plan)
                        previous_plan_state = copy.deepcopy(raw_env.current_plan)
                        replan_needed = True
                        training_scenarios.append({
                            "day": sim_day,
                            "time": sim_time,
                            "previous_plan": copy.deepcopy(previous_plan_state),
                            "remaining_tasks": copy.deepcopy(remaining_to_plan),
                            "time_since_last_break": time_since_last_break,
                            "total_break_time": total_break_time_today,
                            "last_task_type": raw_env.last_task_type,
                            "task_energy_usage": simulator.task_energy_usage,
                            "energy_debt": simulator.energy_debt,
                            "start_energy": simulator.start_energy,
                        })
                        break

                # end of day
                if sim_time >= raw_env.work_end_hour:
                    simulator.reset(sim_time, raw_env.work_end_hour, weekly=((sim_day+1) % 5 == 0))
                    sim_day += 1
                    if sim_day >= raw_env.total_days:
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        remaining_to_plan = sort_tasks_by_deadline_and_priority(remaining_to_plan)
                        if remaining_to_plan:
                            training_scenarios.append({
                                "day": sim_day,
                                "time": sim_time,
                                "previous_plan": copy.deepcopy(previous_plan_state),
                                "remaining_tasks": copy.deepcopy(remaining_to_plan),
                                "time_since_last_break": time_since_last_break,
                                "total_break_time": total_break_time_today,
                                "last_task_type": raw_env.last_task_type,
                                "task_energy_usage": simulator.task_energy_usage,
                                "energy_debt": simulator.energy_debt,
                                "start_energy": simulator.start_energy,
                            })
                        remaining_to_plan = []
                        break
                    sim_time = raw_env.work_start_hour
                    time_since_last_break = 0.0
                    total_break_time_today = 0.0
                    raw_env.last_task_type = None

                    if current_plan:  # still have tasks in plan
                        next_item = current_plan[0]
                        next_start_dt = next_item["start_time"]
                        if next_start_dt.date() <= current_sim_dt.date():
                            remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                            remaining_to_plan = sort_tasks_by_deadline_and_priority(remaining_to_plan)
                            previous_plan_state = copy.deepcopy(raw_env.current_plan)
                            replan_needed = True
                            training_scenarios.append({
                                "day": sim_day,
                                "time": sim_time,
                                "previous_plan": copy.deepcopy(previous_plan_state),
                                "remaining_tasks": copy.deepcopy(remaining_to_plan),
                                "time_since_last_break": time_since_last_break,
                                "total_break_time": total_break_time_today,
                                "last_task_type": raw_env.last_task_type,
                                "task_energy_usage": simulator.task_energy_usage,
                                "energy_debt": simulator.energy_debt,
                                "start_energy": simulator.start_energy,
                            })
                            break
            # if we moved through tasks without re-planning we finish month
            if not replan_needed:
                if not remaining_to_plan:
                    break
                else:
                    remaining_to_plan = []
            else:
                current_generation += 1

        if training_scenarios:
            self._finetune_on_history(user, training_scenarios, start_date)

        return {
            "total_replans": current_generation,
            "days_used": sim_day,
            "scenarios_collected": len(training_scenarios)
        }

    def _generate_plan(self, env, raw_env, ac_model, remaining_tasks, previous_plan, sim_day, sim_time,
                       time_since_last_break, total_break_time, last_task_type):
        t0 = time.time()
        obs = env.reset()

        raw_env.current_day = sim_day
        raw_env.current_time_in_day = sim_time
        raw_env.previous_plan = previous_plan
        raw_env.current_plan = []
        raw_env.time_since_last_break = time_since_last_break
        raw_env.total_break_time_today = total_break_time
        raw_env.last_task_type = last_task_type

        raw_env.remaining_tasks = remaining_tasks[:raw_env.max_tasks_count]
        raw_env.backlog = remaining_tasks[raw_env.max_tasks_count:]

        obs = raw_env._get_obs()

        done = False
        just_took_break = False
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                pi = ac_model.pi._distribution(obs_t)
                logits = pi.logits.clone()
                # skip empty space in tasks when we move toward end of list
                valid_flat_actions = (len(raw_env.remaining_tasks) + 1) * 12
                logits[valid_flat_actions:] = -torch.inf

                # 2. Zablokuj przerwę, jeśli właśnie była przerwa lub dopiero startuje dzień:
                if just_took_break or raw_env.time_since_last_break < 1.5:
                    logits[:12] = -torch.inf
                action = torch.argmax(logits).item()
            obs, _, done, _ = env.step(action)
            task_act = action // 12
            just_took_break = (task_act == 0)

        generation_time = round(time.time() - t0, 4)
        return raw_env.current_plan, generation_time

    def print_plan_after_train(self, user, tasks):
        model_key = f"ppo_base_pretrained"
        with shelve.open(self.base_storage_path) as db:
            buffer = io.BytesIO(db[model_key])
            ac_model = torch.load(buffer, map_location=device, weights_only=False)
            ac_model.eval()
        start_date = datetime(year=2027, month=1, day=4)
        raw_env = PPOPlannerEnv(users_pool=[user], divided_tasks={0: []}, max_tasks_count=50, planning_mode=True,
                                start_day=start_date)
        env = make_wrapped_env(raw_env=raw_env)

        remaining_to_plan = copy.deepcopy(tasks)
        print("lista zadań podana")
        for task in remaining_to_plan:
            print(f"---- [ZADANIE ID: {task.id:3}] | Prio: {task.priority:6} | Typ: {task.type:10} | "
                  f"Deadline: {task.deadline} | "
                  f"(Czas: {task.workhours:.2f}h)")
        previous_plan_state = []

        sim_day = 0
        sim_time = raw_env.work_start_hour

        time_since_last_break = 0.0
        total_break_time_today = 0.0
        current_plan, gen_time = self._generate_plan(
                env, raw_env, ac_model, remaining_to_plan, previous_plan_state, sim_day, sim_time,
                time_since_last_break, total_break_time_today, raw_env.last_task_type
            )
        print(f"\n================ WYGENEROWANY PLAN ================")
        for item in current_plan:
            if item.get("is_break"):
                print(
                    f"☕ [PRZERWA] {item['start_time'].strftime('%H:%M')} - {item['end_time'].strftime('%H:%M')} (Czas: {item['duration']:.2f}h)")
            else:
                task = item["task"]
                dl_str = task.deadline.strftime('%Y-%m-%d %H:%M') if task.deadline else "Brak"
                print(f"📋 [ZADANIE ID: {task.id:3}] | Prio: {task.priority:6} | Typ: {task.type:10} | "
                      f"Deadline: {dl_str} | "
                      f"Zaplanowano: {item['start_time'].strftime('%d-%m %H:%M')} -> {item['end_time'].strftime('%d-%m %H:%M')} "
                      f"(Czas: {item['duration']:.2f}h)")
        print("===================================================\n")
        # --- KALKULACJA MIAR JAKOŚCI PLANU ---
        planned_tasks = [p for p in current_plan if not p.get("is_break")]
        total_tasks = len(planned_tasks)
        on_time_tasks = 0
        delayed_tasks = 0
        total_delay_hours = 0.0
        urgent_delayed = 0
        high_delayed = 0

        total_breaks_duration = 0.0
        break_count = 0
        total_work_duration = 0.0

        for item in current_plan:
            if item.get("is_break"):
                total_breaks_duration += item["duration"]
                break_count += 1
            else:
                total_work_duration += item["duration"]
                task = item["task"]
                if task.deadline:
                    delay = (item["end_time"] - task.deadline).total_seconds() / 3600.0
                    if delay > 0:
                        delayed_tasks += 1
                        total_delay_hours += delay
                        if task.priority == "urgent":
                            urgent_delayed += 1
                        elif task.priority == "high":
                            high_delayed += 1
                    else:
                        on_time_tasks += 1
                else:
                    on_time_tasks += 1

        pct_on_time = (on_time_tasks / total_tasks * 100.0) if total_tasks > 0 else 0.0
        break_ratio = (total_breaks_duration / max(0.1, total_work_duration)) * 100.0
        avg_delay_on_delayed = (total_delay_hours / max(1, delayed_tasks))

        print("======================== MIARY JAKOŚCI HARMONOGRAMU ========================")
        print(f"Liczba zaplanowanych zadań:    {total_tasks} szt. (z puli {len(tasks)} podanych)")
        print(f"Zadania ukończone na czas:     {on_time_tasks} ({pct_on_time:.1f}%)")
        print(f"Zadania opóźnione:             {delayed_tasks} (w tym urgent: {urgent_delayed}, high: {high_delayed})")
        print(f"Łączna suma opóźnień:          {total_delay_hours:.2f} godz.")
        print(f"Średnie opóźnienie (spóźnione):{avg_delay_on_delayed:.2f} godz./zadanie")
        print(f"Łączny czas samej pracy:       {total_work_duration:.2f} h (nominalnie ~160h)")
        print(f"Liczba przerw:                 {break_count} (łączny czas: {total_breaks_duration:.2f} h)")
        print(f"Udział przerw w czasie pracy:  {break_ratio:.1f}% (cel ergonomiczny: 10–15%)")
        print("===========================================================================\n")


    def _finetune_on_history(self, user: User, scenarios: list, start_day: datetime, epochs: int = 5):
        print(f"PPO: Finetuning after month on {len(scenarios)} performed scenarios...")
        model_key = f"ppo_user_{user.id}_active"
        with shelve.open(self.storage_path) as db:
            buffer = io.BytesIO(db[model_key])
            working_model = torch.load(buffer, map_location=device, weights_only=False)

        def pretrained_actor_critic(obs_space, act_space, **kwargs):
            ac = core.MLPActorCritic(obs_space, act_space, **kwargs)
            ac.load_state_dict(working_model.state_dict())
            return ac

        def env_fn():
            raw_env = PPOPlannerEnv(users_pool=[user], divided_tasks={0: []}, training_on_history=True,
                                    historical_scenarios=scenarios, start_day=start_day)
            return make_wrapped_env(raw_env=raw_env)

        ppo(
            env_fn=env_fn,
            actor_critic=pretrained_actor_critic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=max(1000, len(scenarios) * 100),  # epoch length depending on scenarios count
            epochs=epochs,
            pi_lr=1e-5,  # low learning rate to just adjust the model and not change drastically
            vf_lr=5e-5,
            target_kl=0.02,
            train_pi_iters=40,
            train_v_iters=40,
            logger_kwargs=dict(output_dir=f"./PPOGenerated/{model_key}", exp_name=f"{model_key}")
        )

        # update the active model
        loaded_model = torch.load(f"./PPOGenerated/{model_key}/pyt_save/model.pt", map_location=device, weights_only=False)
        self._save_to_storage(model_key, loaded_model, True)
        print(f"PPO: ------ Finetune END ------")

    def _save_to_storage(self, key: str, model, finetune_on_history: bool = False):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        path = self.storage_path if finetune_on_history else self.base_storage_path
        with shelve.open(path) as db:
            db[key] = buffer.getvalue()
