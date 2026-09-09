import copy
import io
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


class PPOPlanner:
    def __init__(self, user: Optional[User] = None):
        self.user = user
        self.storage_path = "db/models_store.db"
        self.repository = None

    def pretrain(self, all_users: List[User], pretrain_tasks: dict, epochs: int = 150):
        def env_fn():
            return make_wrapped_env(users_pool=all_users, tasks=pretrain_tasks)

        print("[PPOPlanner] Start pretraining...")
        ppo(
            env_fn=env_fn,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=2000,
            epochs=epochs,
            pi_lr=3e-4,
            vf_lr=1e-3,
            logger_kwargs=dict(output_dir="./PPOGenerated", exp_name="pretrain")
        )
        # save model to NoSQL
        loaded_model = torch.load("./PPOGenerated/pretrain/pyt_save/model.pt", map_location=device)
        self._save_to_storage("ppo_base_pretrained", loaded_model)

    def finetune_user(self, user: User, finetune_tasks: dict, epochs: int = 30):
        import shelve, io
        base_model = None
        with shelve.open(self.storage_path) as db:
            if "base_pretrained" in db:
                buffer = io.BytesIO(db["base_pretrained"])
                base_model = torch.load(buffer, map_location=device)
            else:
                print("[Warning] No model 'base_pretrained'. Training from scratch.")

        def pretrained_actor_critic(obs_space, act_space, **kwargs):
            ac = core.MLPActorCritic(obs_space, act_space, **kwargs)

            if base_model is not None:
                ac.load_state_dict(base_model.state_dict())

            return ac

        def env_fn():
            return make_wrapped_env(users_pool=[user], tasks=finetune_tasks)

        print(f"[PPOPlanner] Finetuning for user #{user.id}...")
        ppo(
            env_fn=env_fn,
            actor_critic=pretrained_actor_critic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=2000,
            epochs=epochs,
            pi_lr=5e-5,
            vf_lr=2e-4,
            logger_kwargs=dict(output_dir="./PPOGenerated", exp_name=f"finetune_u{user.id}")
        )
        loaded_model = torch.load(f"./PPOGenerated/finetune_u{user.id}/pyt_save/model.pt", map_location=device)
        self._save_to_storage(f"ppo_user_{user.id}_finetuned", loaded_model)

    def plan_and_simulate_month(self, user: User, month_tasks: List[Task], group_id: int, disruptors_map: dict = None,
                                phase='online', phase_order=0, start_date=datetime(2027, 1, 4)):
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
        self.repository = Repository(user, phase, phase_order, start_date)
        model_key = f"ppo_user_{user.id}_active"
        with shelve.open(self.storage_path) as db:
            if model_key not in db:
                base_key = f"ppo_user_{user.id}_finetuned"
                if base_key not in db:
                    raise ValueError(f"No finetuned model for user {user.id}. Run finetuning first.")
                db[model_key] = db[base_key]
            buffer = io.BytesIO(db[model_key])
            ac_model = torch.load(buffer, map_location=device)
            ac_model.eval()

        training_scenarios = []
        # environment used only to plan, not simulate
        raw_env = PPOPlannerEnv(users_pool=[user], divided_tasks={0: []}, max_tasks_count=50, planning_mode=True)
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
                if disruptors_map and sim_day in disruptors_map:
                    pending_disruptors = disruptors_map[sim_day]
                    # Jeśli pierwszy w kolejce disruptor miał się pojawić przed obecnym czasem symulacji
                    if pending_disruptors and pending_disruptors[0][0] <= sim_time:
                        disrupt_time, disruptor_task = pending_disruptors.pop(0)
                        dh = int(disrupt_time)
                        dm = int((disrupt_time - dh) * 60)
                        disruption_occurrence_time = start_date + timedelta(days=calendar_days_passed, hours=dh,
                                                                            minutes=dm)
                        # Zbieramy wszystkie zadania, które jeszcze nie wystartowały
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        remaining_to_plan.append(disruptor_task)

                        # Ustawienie historii, aby nałożyć kary za zmieniane czasy
                        previous_plan_state = copy.deepcopy(raw_env.current_plan)
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
                            "task_energy_usage": simulator.task_energy_usage
                        })
                        break
                disruption_occurrence_time = None

                current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                        minutes=int((sim_time % 1) * 60))
                # if we finish task earlier we might want to re-plan
                # because other task might be more efficiently performed in that gap
                if current_plan:
                    next_item = current_plan[0]
                    next_start_dt = next_item["start_time"]

                    # time to next task
                    gap_seconds = (next_start_dt - current_sim_dt).total_seconds()

                    # re-plan if:
                    # - there is more than one hour to next task start
                    # - next task is in next day, and we still have over 0.5h of work day
                    if gap_seconds > 3600 or (
                            next_start_dt.date() > current_sim_dt.date() and raw_env.work_end_hour - sim_time > 0.5):
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
                            "task_energy_usage": simulator.task_energy_usage
                        })
                        break

                # end of day
                if sim_time >= raw_env.work_end_hour:
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
                                "task_energy_usage": simulator.task_energy_usage
                            })
                            print(f"PPO ------ End of month with tasks left ------")
                        remaining_to_plan = []
                        break
                    sim_time = raw_env.work_start_hour
                    time_since_last_break = 0.0
                    total_break_time_today = 0.0
                    raw_env.last_task_type = None
                    simulator.reset(sim_time, raw_env.work_end_hour, weekly=(sim_day % 5 == 0))

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
                                "task_energy_usage": simulator.task_energy_usage
                            })
                            break

            # if we moved through tasks without re-planning we finish month
            if not replan_needed:
                remaining_to_plan = []
                print(f"PPO ------ Simulation END ------")
            else:
                current_generation += 1

        if training_scenarios:
            self._finetune_on_history(user, training_scenarios)

        return {
            "total_replans": current_generation,
            "days_used": sim_day,
            "scenarios_collected": len(training_scenarios)
        }

    def _generate_plan(self, env, raw_env, ac_model, remaining_tasks, previous_plan, sim_day, sim_time,
                       time_since_last_break, total_break_time, last_task_type):
        t0 = time.time()
        obs, _ = env.reset()

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
                action = torch.argmax(pi.logits).item()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        generation_time = round(time.time() - t0, 4)
        return raw_env.current_plan, generation_time

    def _finetune_on_history(self, user: User, scenarios: list, epochs: int = 5):
        print(f"PPO: Finetuning after month on {len(scenarios)} performed scenarios...")
        model_key = f"user_{user.id}_active"
        with shelve.open(self.storage_path) as db:
            buffer = io.BytesIO(db[model_key])
            working_model = torch.load(buffer, map_location=device)

        def pretrained_actor_critic(obs_space, act_space, **kwargs):
            ac = core.MLPActorCritic(obs_space, act_space, **kwargs)
            ac.load_state_dict(working_model.state_dict())
            return ac

        def env_fn():
            raw_env = PPOPlannerEnv(
                users_pool=[user], divided_tasks={0: []},
                training_on_history=True, historical_scenarios=scenarios
            )
            return make_wrapped_env(raw_env=raw_env)

        ppo(
            env_fn=env_fn,
            actor_critic=pretrained_actor_critic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=max(1000, len(scenarios) * 100),  # epoch length depending on scenarios count
            epochs=epochs,
            pi_lr=1e-5,  # low learning rate to just adjust the model and not change drastically
            vf_lr=5e-5,
            logger_kwargs=dict(output_dir="./PPOGenerated", exp_name=f"{model_key}")
        )

        # update the active model
        loaded_model = torch.load(f"./PPOGenerated/{model_key}/pyt_save/model.pt", map_location=device)
        self._save_to_storage(model_key, loaded_model)
        print(f"PPO: ------ Finetune END ------")

    def _save_to_storage(self, key: str, model):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        with shelve.open(self.storage_path) as db:
            db[key] = buffer.getvalue()
