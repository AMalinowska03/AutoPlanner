import copy
import math
import io
import shelve
from asyncio import all_tasks
from datetime import datetime, time, timedelta
from typing import Optional, List, Dict

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch

from data.DbHelper import Repository
from data.DbModels import User, Task, Plan, PlanTask
from data.database import SessionLocal
from simulation import UserSimulator
from simulation.UserSimulator import SKILL_ATTR_MAP, calculate_switch_lag
from spinup.algos.pytorch.ppo.ppo import ppo, core

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# - day
# - hour
# - time since last break
# - day break time count
# - last task type (encoded embedding 0.0 - 1.0)
NUM_GLOBAL_FEATURES = 5

# - time to deadline
# - priority
# - task type (encoded embedding 0.0 - 1.0)
# - presumed hourly workload
NUM_TASK_FEATURES = 4

TIME_MULTIPLIERS = [0.75, 1.0,  1.25, 1.5,  1.75, 2.0]


class PPOPlannerEnv(gym.Env):
    def __init__(
            self,
            users_pool: List[User],
            divided_tasks: dict,
            save_to_db: bool = False,
            group_id: int = 0,
            max_tasks_count=50,
            start_day: datetime = datetime(2027, 2, 1),
            planning_mode: bool = False,
    ):
        super().__init__()
        self.start_day = start_day
        self.planning_mode = planning_mode
        self.save_to_db = save_to_db

        self.user = users_pool[0]
        self.users_pool = users_pool
        self.divided_tasks = divided_tasks
        self.max_tasks_count = max_tasks_count

        self.simulator = UserSimulator(self.user)
        self.action_space = spaces.MultiDiscrete([max_tasks_count + 1, 12])
        self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(NUM_GLOBAL_FEATURES + max_tasks_count*NUM_TASK_FEATURES,),
                dtype=np.float32)

        self.current_day = 0
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0

        self.last_task_type = None
        self.remaining_tasks = []
        self.backlog = []
        self.current_plan = []
        self.previous_plan = []

        self.metrics_history = []
        self.plan_record = None
        self.total_days = 20  # 4 weeks * 5 days for experiment
        self.group_id = group_id
        self.current_time_in_day = None
        self._update_work_hours()

    def _update_work_hours(self):
        self.work_start_hour = (
            self.user.work_start_time.hour + self.user.work_start_time.minute / 60.0
            if isinstance(self.user.work_start_time, datetime) else 8.0
        )
        self.work_end_hour = (
            self.user.work_end_time.hour + self.user.work_end_time.minute / 60.0
            if isinstance(self.user.work_end_time, datetime) else 16.0
        )
        self.daily_work_time = self.work_end_hour - self.work_start_hour
        self.current_time_in_day = self.work_start_hour

    def reset(self, seed=None, options=None):
        # if training we switch user for each epoch
        if len(self.users_pool) > 1:
            self.user = np.random.choice(self.users_pool)
            self.simulator = UserSimulator(self.user)
            self._update_work_hours()

        if self.save_to_db:
            chosen_tasks_set = self.divided_tasks[self.group_id]
            self.group_id += 1
        else:
            available_orders = list(self.divided_tasks.keys())
            chosen_order = np.random.choice(available_orders)
            chosen_tasks_set = copy.deepcopy(self.divided_tasks[chosen_order])

        self.simulator.reset()
        self.current_plan = {}
        self.previous_plan = {}
        self.current_time_in_day = self.work_start_hour
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0
        self.last_task_type = None
        self.remaining_tasks = chosen_tasks_set[:self.max_tasks_count]
        self.backlog = chosen_tasks_set[self.max_tasks_count:]
        self.metrics_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        norm_day = self.current_day / float(self.total_days)
        norm_hour = (self.current_time_in_day - self.work_start_hour) / self.daily_work_time
        work_elapsed = max(0.1, self.current_time_in_day - self.work_start_hour)
        break_ratio = self.total_break_time_today / work_elapsed

        task_params = SKILL_ATTR_MAP.get(self.last_task_type)
        last_type_num = task_params["embedding"] if task_params is not None else 0

        obs = [
            norm_day,
            norm_hour,
            last_type_num,
            self.time_since_last_break / 4.0,  # why
            break_ratio
        ]
        current_abs_time = self.current_day * 24.0 + self.current_time_in_day
        total_experiment_hours = self.total_days * 24.0

        for i in range(self.max_tasks_count):
            if i < len(self.remaining_tasks):
                task = self.remaining_tasks[i]
                prio_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "urgent": 1.0}
                current_task_param = SKILL_ATTR_MAP.get(task.type)
                current_emb = current_task_param["embedding"] if current_task_param else 0.0

                deadline_hour = self._get_deadline_in_hours(task.deadline)
                hours_left = max(0.0, deadline_hour - current_abs_time)
                norm_deadline = min(1.0, hours_left / total_experiment_hours)
                obs.extend([
                    float(task.workhours) / self.daily_work_time,
                    prio_map.get(task.priority, 0.5),
                    current_emb,
                    norm_deadline
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action_type, action_time = action[0], action[1]
        reward = 0.0
        terminated = False
        truncated = False

        task_duration = (action_time+1)*5/60
        calendar_days_passed = self.current_day if self.current_day <= 5 else self.current_day + 2 if self.current_day <= 10 else self.current_day + 4 if self.current_day <= 15 else self.current_day + 6
        h = int(self.current_time_in_day)
        m = int((self.current_time_in_day - h) * 60)
        start_task_time = self.start_day + timedelta(days=calendar_days_passed, hours=h, minutes=m)
        dh = int(self.current_time_in_day + task_duration)
        dm = int((self.current_time_in_day + task_duration - dh) * 60)
        end_task_time = start_task_time + timedelta(hours=dh, minutes=dm)

        if action_type == 0:
            break_duration = task_duration
            self.simulator.process_break(break_duration, self.current_time_in_day)

            if self.planning_mode:
                self.current_plan.append({"is_break": True, "start_time": start_task_time, "end_time": end_task_time})

            reward += self._calculate_break_reward(break_duration)

            self.current_time_in_day += break_duration
            self.total_break_time_today += break_duration
            self.time_since_last_break = 0.0
            # break resets the mind, so we don't have a lag before the next task - simplified logic
            self.last_task_type = None
            end_time = self.current_time_in_day
        else:
            current_abs_start = self.current_day * 24.0 + self.current_time_in_day  # hourly since experiment start
            task_idx = action_type - 1
            if task_idx >= len(self.remaining_tasks):
                reward -= 10.0
                return self._get_obs(), reward, terminated, truncated, {}

            task = self.remaining_tasks.pop(task_idx)
            if self.backlog:
                self.remaining_tasks.append(self.backlog.pop(0))

            if self.planning_mode:
                self.current_plan.append({"task_id": task.id, "start_time": start_task_time, "end_time": end_task_time})

            lag = calculate_switch_lag(self.last_task_type, task.type)
            attention_at_start = self.simulator.get_current_attention(self.current_time_in_day)
            energy_at_start = self.simulator.get_current_energy(self.current_time_in_day)

            actual_duration, end_time, energy_used = self.simulator.execute_task(
                task=task,
                start_time=self.current_time_in_day,
                context_switch_lag=lag
            )

            reward += self._calculate_deadline_reward(task, end_time)
            reward += self._calculate_time_allotment_reward(task, action_time, actual_duration)
            reward += self._calculate_disruption_reward(task, current_abs_start)

            self.current_plan[task.id] = current_abs_start
            self.current_time_in_day = end_time
            self.time_since_last_break += actual_duration
            self.last_task_type = task.type

        if end_time > self.work_end_hour:
            reward += self._calculate_overtime_reward(end_time)
            reward += self._calculate_end_day_break_reward()
            self._advance_to_next_day()

        if len(self.remaining_tasks) == 0:
            reward += 60.0
            terminated = True
        elif self.current_day >= self.total_days:
            reward -= len(self.remaining_tasks) * 20.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _advance_to_next_day(self):
        self.current_day += 1
        self.current_time_in_day = self.work_start_hour
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0
        self.last_task_type = None
        self.simulator.reset(weekly=(self.current_day % 5 == 0))

    def _calculate_break_reward(self, break_duration: float):
        break_reward = 0.0
        # penalty (beginning/end of day break, too many/few breaks)
        if self.current_time_in_day == self.work_start_hour:
            break_reward -= break_duration * 3.0
        if self.current_time_in_day + break_duration == self.work_end_hour:
            break_reward -= break_duration * 4.0

        # reward (break for eating midday)
        if 11.30 < self.current_time_in_day < 14.5 and 0.25 < break_duration < 0.75:
            break_reward += break_duration * 5.0
        return break_reward

    def _calculate_deadline_reward(self, task: Task, end_time: float):
        deadline_reward = 0.0
        global_end_hour = self.current_day * 24.0 + end_time
        deadline_hours = self._get_deadline_in_hours(task.deadline)

        prio_weights = {"low": 1.0, "medium": 2.0, "high": 4.0, "urgent": 8.0}
        w_prio = prio_weights.get(task.priority, 1.0)

        if global_end_hour > deadline_hours:
            tardiness = global_end_hour - deadline_hours
            deadline_reward -= (10.0 + tardiness * 2.0) * w_prio
        else:
            deadline_reward += 8.0 * w_prio
        return deadline_reward

    def _calculate_time_allotment_reward(self, task:Task, action_time, actual_duration: float):
        time_reward = 0.0
        time_multiplier = TIME_MULTIPLIERS[action_time % len(TIME_MULTIPLIERS)]
        planned_duration = time_multiplier * float(task.workhours)
        # penalty (task execution time exceeded/finished early - exponential, disruption, overtime)
        planning_time_difference = actual_duration - planned_duration
        if -0.25 < planning_time_difference < 0.25:
            time_reward += 5.0 / (1.0 + planning_time_difference * 4.0)
        elif planning_time_difference > 0.25:
            time_reward -= planning_time_difference ** 2 * 6.0
        elif planning_time_difference < -0.25:
            time_reward -= abs(planning_time_difference) * 2.0

        return time_reward

    def _calculate_disruption_reward(self, task: Task, current_abs_start: float):
        disruption_reward = 0.0
        if task.id in self.previous_plan:
            prev_start = self.previous_plan[task.id]
            shift = abs(current_abs_start - prev_start)  # how much was task shifted in time
            if shift > 0.1:
                # the closer to the event in previous plan and the bigger the shift the bigger penalty
                time_to_event = max(0.1, prev_start - current_abs_start)
                disruption_reward -= (shift * 4.0) / math.sqrt(time_to_event)
            else:
                disruption_reward += 2.0
        return disruption_reward

    def _calculate_overtime_reward(self, end_time: float):
        # overtime penalty - the longer task takes to end the bigger penalty
        overtime_reward = 0.0
        overtime = end_time - self.work_end_hour
        if overtime > 0.0:
            overtime_reward -= overtime ** 2.0 * 3.0
        return overtime_reward

    def _calculate_end_day_break_reward(self):
        break_reward = 0.0
        worked_today = max(0.1, self.current_time_in_day - self.work_start_hour)
        break_pct = self.total_break_time_today / worked_today
        if 0.10 <= break_pct <= 0.15:
            break_reward += 3.0
        elif break_pct < 0.10:
            break_reward -= 0.5
        else:
            break_reward -= 5 * break_pct
        return break_reward

    def _get_deadline_in_hours(self, deadline):
        """
        Recalculates deadline time to amount of hours since month start
        :param deadline:
        :return:
        """
        if deadline is None:
            return float(self.total_days * 24.0)

        if isinstance(deadline, datetime):
            if isinstance(self.start_day, datetime):
                start = self.start_day
                base_datetime = datetime(start.year, start.month, start.day, int(self.work_start_hour))
            else:
                base_datetime = datetime(deadline.year, deadline.month, deadline.day, int(self.work_start_hour))

            diff_seconds = (deadline - base_datetime).total_seconds()
            return max(0.0, diff_seconds / 3600.0)

        if isinstance(deadline, (int, float)):
            if deadline <= self.total_days:
                return float(deadline * 24.0)
            return float(deadline)

        return float(self.total_days * 24.0)


class FlattenMultiDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, num_time_bins: int = 12):
        super().__init__(env)
        self.num_time_bins = num_time_bins
        total_actions = env.action_space.nvec[0] * self.num_time_bins
        self.action_space = spaces.Discrete(total_actions)

    def action(self, action):
        """
        Decoding scalar value 0..N*M-1 to tensor (task, time)
        :param action:
        :return:
        """
        action_idx = int(action)
        time_act = action_idx % self.num_time_bins
        task_act = action_idx // self.num_time_bins
        return np.array([task_act, time_act])


class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Converts format returned by Gymnasium (5 values with terminated and truncated)
    to standard Gym format (4 values where done = terminated or truncated)
    and makes sure reset returns only observation without information.
    """
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return step_out


def make_wrapped_env(
        users_pool=None,
        tasks=None,
        save_to_db: bool = False,
        group_id: int = 0,
        start_day: datetime = datetime(year=2027, day=1, month=2),
        raw_env=None
):
    if raw_env is not None:
        env = raw_env
    else:
        if users_pool is None or tasks is None:
            raise ValueError("Both users_pool and tasks must be provided")
        env = PPOPlannerEnv(users_pool, tasks, save_to_db, group_id, start_day=start_day, max_tasks_count=50)
    env = FlattenMultiDiscreteActionWrapper(env, num_time_bins=12)
    env = GymnasiumToGymWrapper(env)
    return env


class PPOPlanner:
    def __init__(self, user: Optional[User] = None):
        self.list = None
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
        loaded_model = torch.load("./PPOGenerated/pyt_save/model.pt", map_location=device)
        self._save_to_storage("base_pretrained", loaded_model)

    def finetune_user(self, user: User, finetune_tasks: dict, epochs: int = 30):
        def env_fn():
            return make_wrapped_env(users_pool=[user], tasks=finetune_tasks)

        print(f"[PPOPlanner] Finetuning for user #{user.id}...")
        ppo(
            env_fn=env_fn,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=(128, 128)),
            steps_per_epoch=2000,
            epochs=epochs,
            pi_lr=5e-5,
            vf_lr=2e-4,
            logger_kwargs=dict(output_dir="./PPOGenerated", exp_name=f"finetune_u{user.id}")
        )
        loaded_model = torch.load("./PPOGenerated/pyt_save/model.pt", map_location=device)
        self._save_to_storage(f"user_{user.id}_finetuned", loaded_model)

    def plan_and_simulate_month(
            self,
            user: User,
            month_tasks: List[Task],
            group_id: int,
            disruptors_map: Optional[Dict[int, List[Task]]] = None,
            phase: str = 'online',
            phase_order: int = 0,
            start_date: datetime = datetime(2027, 1, 4),
    ):
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
        # load fine tuned model
        model_key = f"user_{user.id}_active"
        with shelve.open(self.storage_path) as db:
            if model_key not in db:
                model_key = f"user_{user.id}_finetuned"
            buffer = io.BytesIO(db[model_key])
            ac_model = torch.load(buffer, map_location=device)
            ac_model.eval()

        raw_env = PPOPlannerEnv(
            users_pool=[user],
            divided_tasks={0: month_tasks},
            max_tasks_count=self.max_tasks_count,
            save_to_db=True,
            group_id=group_id
        )
        env = make_wrapped_env(raw_env=raw_env)

        obs = env.reset()
        current_generation = 0

        # save first plan generation
        self._create_plan_records(raw_env, group_id, current_generation)

        done = False
        while not done:
            current_day = raw_env.current_day

            # if at around current time planned, inject disruptor task to remaining tasks in env
            if disruptors_map and current_day in disruptors_map:
                new_tasks = disruptors_map.pop(current_day)
                raw_env.remaining_tasks = new_tasks + raw_env.remaining_tasks
                current_generation += 1
                # save re-plan
                self._create_plan_records(raw_env, group_id, current_generation)

            # choose next action through model
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                pi = ac_model.pi._distribution(obs_t)
                action = torch.argmax(pi.logits).item()

            obs, reward, done, _ = env.step(action)

        print(f"[PPOPlanner] Zakończono symulację Miesiąca (Group {group_id}).")


    def _save_to_storage(self, key: str, model):
        import shelve, io
        buffer = io.BytesIO()
        torch.save(model, buffer)
        with shelve.open(self.storage_path) as db:
            db[key] = buffer.getvalue()