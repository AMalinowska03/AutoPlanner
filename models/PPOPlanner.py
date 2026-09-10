import copy
import io
import os
import shelve
from datetime import datetime, timedelta
import time
from typing import Optional, List

import torch

from data.DbHelper import Repository
from data.DbModels import User, Task
from simulation.UserSimulator import UserSimulator
from models.PPOEnv import PPOPlannerEnv, make_wrapped_env
from spinup.algos.pytorch.ppo.ppo import ppo, core

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DisruptorsMap = dict[int, list[tuple[float, Task]]]

class PPOPlanner:
    def __init__(self, user: Optional[User] = None):
        self.user = user
        self.storage_path = "db/models_store.db"
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
            pi_lr=3e-4,
            vf_lr=1e-3,
            target_kl=0.015,
            logger_kwargs=dict(output_dir="./PPOGenerated/pretrain", exp_name="pretrain")
        )
        # save model to NoSQL
        loaded_model = torch.load("./PPOGenerated/pretrain/pyt_save/model.pt", map_location=device, weights_only=False)
        self._save_to_storage("ppo_base_pretrained", loaded_model)

    def finetune_user(self, user: User, finetune_tasks: dict, epochs: int = 30):
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
            steps_per_epoch=2000,
            epochs=epochs,
            pi_lr=5e-5,
            vf_lr=2e-4,
            logger_kwargs=dict(output_dir="./PPOGenerated/finetune_u{user.id}", exp_name=f"finetune_u{user.id}")
        )
        loaded_model = torch.load(f"./PPOGenerated/finetune_u{user.id}/pyt_save/model.pt", map_location=device, weights_only=False)
        self._save_to_storage(f"ppo_user_{user.id}_finetuned", loaded_model)

    def plan_and_simulate_month(self, user: User, month_tasks: List[Task], group_id: int,
                                disruptors_map: Optional[DisruptorsMap] = None, phase='online',
                                phase_order=0, start_date=datetime(2027, 1, 4)):
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
        self.repository = Repository(user, phase, phase_order, start_date)
        model_key = f"ppo_user_{user.id}_active"
        with shelve.open(self.storage_path) as db:
            if model_key not in db:
                base_key = f"ppo_user_{user.id}_finetuned"
                if base_key not in db:
                    raise ValueError(f"No finetuned model for user {user.id}. Run finetuning first.")
                db[model_key] = db[base_key]
            buffer = io.BytesIO(db[model_key])
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
            plan_record = self.repository.create_plan_records(
                algorithm="ppo", planned_tasks=current_plan, group_id=group_id,
                generation=current_generation, disruption_time=disruption_occurrence_time, generating_time=gen_time
            )

            print(f"PPO ------ Simulating ------")
            replan_needed = False

            # go through all planned tasks until they are possible to be completed
            while current_plan:
                plan_item = current_plan.pop(0)
                calendar_days_passed = sim_day + (sim_day // 5) * 2

                # execute plan item
                if plan_item.get("is_break"):
                    simulator.process_break(duration=plan_item["duration"], time=sim_time)
                    self.repository.save_break(plan_item["duration"], plan_record, sim_time, plan_item["start_time"])
                    sim_time += plan_item["duration"]
                    time_since_last_break = 0.0
                    total_break_time_today += plan_item["duration"]
                    raw_env.last_task_type = None
                else:
                    task = plan_item["task"]
                    actual_dur, end_time, energy = simulator.execute_task(task, sim_time, raw_env.last_task_type)
                    current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                            minutes=int((sim_time % 1) * 60))
                    self.repository.save_execution_to_db(
                        plan_record, task, current_sim_dt,
                        current_sim_dt + timedelta(hours=actual_dur), energy
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
                        first_disruption_time = min(disrupt_time for disrupt_time, _ in disruptors_appeared)
                        dh = int(first_disruption_time)
                        dm = int((first_disruption_time - dh) * 60)
                        disruption_occurrence_time = start_date + timedelta(days=calendar_days_passed, hours=dh,
                                                                            minutes=dm)
                        previous_plan_state = copy.deepcopy(current_plan)
                        replan_needed = True
                        print(f"PPO ------ Disruptor occurred: RE-PLANNING ------")
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
                        previous_plan_state = copy.deepcopy(raw_env.current_plan)
                        replan_needed = True
                        print(f"PPO ------ Have time left: RE-PLANNING ------")
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
                            print(f"PPO ------ End of month with tasks left ------")
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
                            previous_plan_state = copy.deepcopy(raw_env.current_plan)
                            replan_needed = True
                            print(f"PPO ------ Tasks left from day: RE-PLANNING ------")
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
                remaining_to_plan = []
                print(f"PPO ------ Simulation END ------")
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
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                pi = ac_model.pi._distribution(obs_t)
                logits = pi.logits.clone()
                # skip empty space in tasks when we move toward end of list
                valid_flat_actions = (len(raw_env.remaining_tasks) + 1) * 12
                logits[valid_flat_actions:] = -torch.inf
                action = torch.argmax(logits).item()
            obs, _, done, _ = env.step(action)

        generation_time = round(time.time() - t0, 4)
        return raw_env.current_plan, generation_time

    def print_plan_after_train(self, user, tasks):
        model_key = f"ppo_base_pretrained"
        with shelve.open(self.storage_path) as db:
            buffer = io.BytesIO(db[model_key])
            ac_model = torch.load(buffer, map_location=device, weights_only=False)
            ac_model.eval()
        start_date = datetime(year=2027, month=2, day=1)
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
            logger_kwargs=dict(output_dir=f"./PPOGenerated/{model_key}", exp_name=f"{model_key}")
        )

        # update the active model
        loaded_model = torch.load(f"./PPOGenerated/{model_key}/pyt_save/model.pt", map_location=device, weights_only=False)
        self._save_to_storage(model_key, loaded_model)
        print(f"PPO: ------ Finetune END ------")

    def _save_to_storage(self, key: str, model):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        with shelve.open(self.storage_path) as db:
            db[key] = buffer.getvalue()
