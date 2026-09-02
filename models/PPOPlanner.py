import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import math
from datetime import datetime

from data.DbModels import User, Task
from simulation import UserSimulator
from simulation.UserSimulator import SKILL_ATTR_MAP, calculate_switch_lag

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

class FlattenMultiDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Np. (51, 12) -> Discrete(612)
        self.dims = env.action_space.nvec
        self.action_space = gym.spaces.Discrete(int(np.prod(self.dims)))

    def action(self, action):
        """
        Decoding scalar value 0..N*M-1 to tensor (task, time)
        :param action:
        :return:
        """
        action_idx = int(action)
        time_act = action_idx % self.dims[1]
        task_act = action_idx // self.dims[1]
        return np.array([task_act, time_act])

class PPOPlannerEnv(gym.Env):
    def __init__(self, user: User, tasks: list, max_tasks_count=50):
        super().__init__()
        self.user = user
        self.tasks = tasks
        self.max_tasks_count = max_tasks_count
        self.training_simulator = UserSimulator(user)
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
        self.current_plan = {}
        self.metrics_history = []
        self.previous_plan = {}

        self.work_start_hour = (
            user.work_start_time.hour + user.work_start_time.minute / 60.0
            if isinstance(user.work_start_time, datetime) else 8.0
        )
        self.work_end_hour = (
            user.work_end_time.hour + user.work_end_time.minute / 60.0
            if isinstance(user.work_end_time, datetime) else 16.0
        )
        self.daily_work_time = self.work_end_hour - self.work_start_hour
        self.total_days = 20  # 4 weeks * 5 days for experiment
        self.current_time_in_day = self.work_start_hour

    def reset(self, seed=None, options=None):
        self.training_simulator.reset()
        self.previous_plan = {}
        self.current_time_in_day = self.work_start_hour
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0
        self.last_task_type = None
        self.remaining_tasks = [t for t in self.tasks[:self.max_tasks_count]]
        self.current_plan = {}
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

        end_time = self.current_time_in_day

        if action_type == 0:
            break_duration = (action_time+1)*5/60
            self.training_simulator.process_break(break_duration, self.current_time_in_day)

            reward += self._calculate_break_reward(break_duration)

            self.current_time_in_day += break_duration
            self.total_break_time_today += break_duration
            self.time_since_last_break = 0.0
            end_time = self.current_time_in_day
        else:
            current_abs_start = self.current_day * 24.0 + self.current_time_in_day  # hourly since experiment start
            task_idx = action_type - 1
            if task_idx >= len(self.remaining_tasks):
                reward -= 10.0
                return self._get_obs(), reward, terminated, truncated, {}

            task = self.remaining_tasks.pop(task_idx)
            lag = calculate_switch_lag(self.last_task_type, task.type)
            attention_at_start = self.training_simulator.get_current_attention(self.current_time_in_day)
            energy_at_start = self.training_simulator.get_current_energy(self.current_time_in_day)

            actual_duration, end_time, energy_used = self.training_simulator.execute_task(
                task=task,
                start_time=self.current_time_in_day,
                context_switch_lag=lag
            )

            # save to db - TODO

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
        self.training_simulator.reset(weekly=(self.current_day % 5 == 0))

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
        Przelicza termin zadania (deadline) na liczbę godzin
        od momentu startu eksperymentu (Dzień 0, self.work_start_hour).
        """
        if deadline is None:
            return float(self.total_days * 24.0)

        if isinstance(deadline, datetime):
            if isinstance(self.user.work_start_time, datetime):
                base_datetime = self.user.work_start_time
            else:
                base_datetime = datetime(deadline.year, deadline.month, deadline.day, int(self.work_start_hour))

            diff_seconds = (deadline - base_datetime).total_seconds()
            return max(0.0, diff_seconds / 3600.0)

        if isinstance(deadline, (int, float)):
            if deadline <= self.total_days:
                return float(deadline * 24.0)
            return float(deadline)

        return float(self.total_days * 24.0)

class PPOPlanner:
    def __init__(self, user: User):
        self.list = None

    def train(self):
        pass

    def execute_plan(self):
        pass