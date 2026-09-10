import copy
import math
from datetime import datetime, timedelta
import random
from typing import Optional, List, Dict

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch

from data.DbModels import User, Task
from data.DbHelper import get_user_work_hours, sim_time_to_datetime
from simulation.UserSimulator import SKILL_ATTR_MAP, UserSimulator

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
# - how much task was moved if in previous plan
NUM_TASK_FEATURES = 5

TIME_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]

PENALTY_WEIGHT_DEADLINE = 6.0
PENALTY_WEIGHT_EFFICIENCY = 3.0
PENALTY_WEIGHT_STABILITY = 2.0
PENALTY_WEIGHT_HEALTH = 1.5


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
            training_on_history: bool = False,
            historical_scenarios: Optional[list] = None
    ):
        super().__init__()
        self.start_day = start_day
        self.planning_mode = planning_mode
        self.save_to_db = save_to_db
        self.training_on_history = training_on_history
        self.historical_scenarios = historical_scenarios

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
        self.work_start_hour, self.work_end_hour = get_user_work_hours(self.user)
        self.daily_work_time = self.work_end_hour - self.work_start_hour
        self.current_time_in_day = self.work_start_hour

    def reset(self, seed=None, options=None):
        """
        Method run after each episode that ends if days count exceeds 20 or there are no more tasks to plan
        During training we stage disruption scenario randomly with 30% probability, setting previous plan to random list
        moving current day further on and removing "completed" tasks from remaining, so we are only re-planning
        It teaches the model stability when re-planning
        :param seed:
        :param options:
        :return:
        """
        self.current_plan = []
        self.previous_plan = []
        self.current_time_in_day = self.work_start_hour
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0
        self.last_task_type = None
        self.remaining_tasks = []

        # if training we switch user for each epoch
        if len(self.users_pool) > 1:
            self.user = np.random.choice(self.users_pool)
            self.simulator = UserSimulator(self.user)
            self._update_work_hours()

        if self.planning_mode:
            chosen_tasks_set = self.divided_tasks[self.group_id]
            self.group_id += 1
            self.simulator.reset(self.work_start_hour, self.work_end_hour, self.current_day % 5 == 0)
        elif self.training_on_history and self.historical_scenarios:
            scenario = random.choice(self.historical_scenarios)

            self.current_day = scenario["day"]
            self.current_time_in_day = scenario["time"]
            self.previous_plan = copy.deepcopy(scenario["previous_plan"])
            self.time_since_last_break = scenario["time_since_last_break"]
            self.total_break_time_today = scenario["total_break_time"]
            self.last_task_type = scenario["last_task_type"]
            self.simulator.task_energy_usage = scenario["task_energy_usage"]

            chosen_tasks_set = copy.deepcopy(scenario["remaining_tasks"])

            self.remaining_tasks = chosen_tasks_set[:self.max_tasks_count]
            self.backlog = chosen_tasks_set[self.max_tasks_count:]
        else:
            available_orders = list(self.divided_tasks.keys())
            chosen_order = np.random.choice(available_orders)
            chosen_tasks_set = copy.deepcopy(self.divided_tasks[chosen_order])

            train_disruption = random.random()
            if train_disruption < 0.3:
                self.current_day = random.randint(2, 12)
                self.current_time_in_day = random.uniform(self.work_start_hour, self.work_end_hour)
                target_abs_time = self.current_day * 24.0 + self.current_time_in_day

                temp_day = 0
                temp_time = self.work_start_hour
                tasks_to_keep = []
                had_break = False

                for task in chosen_tasks_set:
                    if temp_time >= 12.0 and not had_break:
                        b_duration = 0.5
                        if temp_time + b_duration <= self.work_end_hour:
                            calendar_days_passed = self.current_day + (self.current_day // 5) * 2
                            b_start = self.start_day + timedelta(days=calendar_days_passed, hours=int(temp_time),
                                                                 minutes=int((temp_time % 1) * 60))
                            b_end = b_start + timedelta(hours=int(b_duration), minutes=int((b_duration % 1) * 60))

                            self.previous_plan.append({
                                "is_break": True,
                                "start_time": b_start,
                                "end_time": b_end
                            })
                            temp_time += b_duration
                        had_break = True
                    task_duration = float(task.workhours)

                    # overspill tasks to next day
                    if temp_time + task_duration > self.work_end_hour:
                        temp_day += 1
                        temp_time = self.work_start_hour
                        had_break = False

                    # save task to previous plan
                    cal_days = temp_day if temp_day <= 5 else temp_day + 2 if temp_day <= 10 else temp_day + 4 if temp_day <= 15 else temp_day + 6
                    t_start = self.start_day + timedelta(days=cal_days, hours=int(temp_time),
                                                         minutes=int((temp_time % 1) * 60))
                    start_abs_time = temp_day * 24.0 + temp_time

                    # move planning clock
                    temp_time += task_duration
                    t_end = self.start_day + timedelta(days=cal_days, hours=int(temp_time),
                                                       minutes=int((temp_time % 1) * 60))
                    end_abs_time = temp_day * 24.0 + temp_time
                    self.previous_plan.append({
                        "task_id": task.id,
                        "start_time": t_start,
                        "end_time": t_end,
                        "_abs_start": start_abs_time  # added just for training to not calculate it again in reward
                    })

                    # keep tasks that are after target date
                    if end_abs_time > target_abs_time:
                        tasks_to_keep.append(task)
                self.remaining_tasks = tasks_to_keep[:self.max_tasks_count]
                self.backlog = tasks_to_keep[self.max_tasks_count:]

            self.simulator.reset(self.work_start_hour, self.work_end_hour, self.current_day % 5 == 0)

        if len(self.remaining_tasks) == 0:
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
        current_date = sim_time_to_datetime(self.start_day, self.current_day, self.current_time_in_day)
        total_experiment_hours = self.total_days * 24.0
        total_calendar_hours = (self.total_days + 8) * 24.0

        for i in range(self.max_tasks_count):
            if i < len(self.remaining_tasks):
                task = self.remaining_tasks[i]
                prio_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "urgent": 1.0}
                current_task_param = SKILL_ATTR_MAP.get(task.type)
                current_emb = current_task_param["embedding"] if current_task_param else 0.0

                hours_left = max(0.0, (task.deadline - current_date).total_seconds() / 3600.0)
                norm_deadline = min(1.0, hours_left / total_calendar_hours)
                prev_record = next((item for item in self.previous_plan if item.get("task_id") == task.id), None)
                if prev_record:
                    prev_start = prev_record.get("_abs_start")
                    time_diff = (prev_start - current_abs_time) / total_experiment_hours
                else:
                    time_diff = -1.0
                obs.extend([
                    float(task.workhours) / self.daily_work_time,
                    prio_map.get(task.priority, 0.5),
                    current_emb,
                    norm_deadline,
                    time_diff
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action_type, action_time = action[0], action[1]
        reward = 0.0
        terminated = False
        truncated = False

        task_duration = (action_time+1)*5/60
        calendar_days_passed = self.current_day + (self.current_day // 5) * 2  # add weekends to get date
        h = int(self.current_time_in_day)
        m = int((self.current_time_in_day - h) * 60)
        start_task_time = self.start_day + timedelta(days=calendar_days_passed, hours=h, minutes=m)
        dh = int(task_duration)
        dm = int((task_duration - dh) * 60)
        end_task_time = start_task_time + timedelta(hours=dh, minutes=dm)

        if action_type == 0:
            break_duration = task_duration
            self.simulator.process_break(break_duration, self.current_time_in_day)

            if self.planning_mode:
                self.current_plan.append({"is_break": True, "start_time": start_task_time,
                                          "end_time": end_task_time, "duration": break_duration})
            else:
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
                time_multiplier = TIME_MULTIPLIERS[action_time % len(TIME_MULTIPLIERS)]
                actual_duration = float(task.workhours) * time_multiplier
                end_time = self.current_time_in_day + actual_duration
                dh = int(actual_duration)
                dm = int((actual_duration - dh) * 60)
                end_task_time = start_task_time + timedelta(hours=dh, minutes=dm)

                self.current_plan.append({
                    "task_id": task.id, "task": task,
                    "start_time": start_task_time, "end_time": end_task_time,
                    "_abs_start": current_abs_start, "duration": actual_duration,
                })
            else:
                actual_duration, end_time, energy_used = self.simulator.execute_task(
                    task=task,
                    start_time=self.current_time_in_day,
                    prev_task_type=self.last_task_type
                )

                reward += self._calculate_deadline_reward(task, end_time)
                reward += self._calculate_time_allotment_reward(task, action_time, actual_duration)
                reward += self._calculate_disruption_reward(task, current_abs_start)

            self.current_time_in_day = end_time
            self.time_since_last_break += actual_duration
            self.last_task_type = task.type

        if end_time > self.work_end_hour:
            if self.planning_mode is False:
                reward += self._calculate_overtime_reward(end_time)
                reward += self._calculate_end_day_break_reward()
            self._advance_to_next_day()

        if len(self.remaining_tasks) == 0:
            if self.planning_mode is False:
                reward += 60.0
            terminated = True
        elif self.current_day >= self.total_days:
            if self.planning_mode is False:
                reward -= len(self.remaining_tasks) * 20.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _advance_to_next_day(self):
        self.simulator.reset(self.current_time_in_day, self.work_end_hour, weekly=(self.current_day % 5 == 0))
        self.current_day += 1
        self.current_time_in_day = self.work_start_hour
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0
        self.last_task_type = None

    def _calculate_break_reward(self, break_duration: float):
        break_reward = 0.0
        # penalty (beginning/end of day break, too many/few breaks)
        if self.current_time_in_day == self.work_start_hour:
            break_reward -= break_duration * PENALTY_WEIGHT_EFFICIENCY
        if self.current_time_in_day + break_duration == self.work_end_hour:
            break_reward -= break_duration * (PENALTY_WEIGHT_EFFICIENCY + 1)

        # reward (break for eating midday)
        if 11.30 < self.current_time_in_day < 14.5 and 0.25 < break_duration < 0.75:
            break_reward += break_duration * PENALTY_WEIGHT_EFFICIENCY
        return break_reward

    def _calculate_deadline_reward(self, task: Task, end_time: float):
        deadline_reward = 0.0
        end_date = sim_time_to_datetime(self.start_day, self.current_day, end_time)
        tardiness = (end_date - task.deadline).total_seconds() / 3600.0

        prio_weights = {"low": 1.0, "medium": 2.0, "high": 4.0, "urgent": 8.0}
        w_prio = prio_weights.get(task.priority, 1.0)

        if tardiness > 0:
            # missing deadline is more crucial to correct than rewarding for doing task on time
            deadline_reward -= (1.5 * PENALTY_WEIGHT_DEADLINE + tardiness * 2.0) * w_prio
        else:
            deadline_reward += PENALTY_WEIGHT_DEADLINE * w_prio
        return deadline_reward

    def _calculate_time_allotment_reward(self, task:Task, action_time, actual_duration: float):
        time_reward = 0.0
        time_multiplier = TIME_MULTIPLIERS[action_time % len(TIME_MULTIPLIERS)]
        planned_duration = time_multiplier * float(task.workhours)
        # penalty (task execution time exceeded/finished early - exponential, disruption, overtime)
        planning_time_difference = actual_duration - planned_duration
        # 15min grace period
        if -0.25 < planning_time_difference < 0.25:
            time_reward += 5.0 / (1.0 + abs(planning_time_difference) * PENALTY_WEIGHT_EFFICIENCY)
        elif planning_time_difference > 0.25:
            # the more time was actually needed to complete the task the more penalty exponentially
            time_reward -= planning_time_difference ** 2 * (2 * PENALTY_WEIGHT_EFFICIENCY)
        elif planning_time_difference < -0.25:  # finished before time
            # we could save plan time here but giving a bit more time is always better than not giving enough
            time_reward -= abs(planning_time_difference) * PENALTY_WEIGHT_EFFICIENCY

        return time_reward

    def _calculate_disruption_reward(self, task: Task, current_abs_start: float):
        disruption_reward = 0.0
        prev_record = next((item for item in self.previous_plan if item.get("task_id") == task.id), None)
        if prev_record:
            prev_start = prev_record.get("_abs_start", current_abs_start)
            shift = abs(current_abs_start - prev_start)

            if shift > 0.1:  # we don't include small shifts of a few minutes
                time_to_event = max(0.5, prev_start - current_abs_start)
                # if bigger the shift the less penalty grows - it's a big change anyway (logarithmic)
                base_penalty = math.log(1.0 + shift) * PENALTY_WEIGHT_STABILITY

                # the further the event was originally planned the less impact it has
                proximity_multiplier = 1.0 / time_to_event

                disruption_reward -= base_penalty * proximity_multiplier
            else:
                # small reward for keeping the task unmoved
                disruption_reward += PENALTY_WEIGHT_STABILITY

        return disruption_reward

    def _calculate_overtime_reward(self, end_time: float):
        # overtime penalty - the longer task takes to end the bigger penalty
        overtime_reward = 0.0
        overtime = end_time - self.work_end_hour
        if overtime > 0.0:
            overtime_reward -= overtime ** 2.0 * PENALTY_WEIGHT_HEALTH
        return overtime_reward

    def _calculate_end_day_break_reward(self):
        break_reward = 0.0
        worked_today = max(0.1, self.current_time_in_day - self.work_start_hour)
        break_pct = self.total_break_time_today / worked_today
        if 0.10 <= break_pct <= 0.15:
            break_reward += 2 * PENALTY_WEIGHT_HEALTH
        elif break_pct < 0.10:
            break_reward -= PENALTY_WEIGHT_HEALTH
        else:
            break_reward -= 5 * PENALTY_WEIGHT_HEALTH * (break_pct - 0.15)
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

