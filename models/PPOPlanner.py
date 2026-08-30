import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from datetime import datetime

from data.DbModels import User
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
        self.remaining_tasks = []
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
            self.time_since_last_break / 4.0,
            break_ratio
        ]

        for i in range(self.max_tasks_count):
            if i < len(self.remaining_tasks):
                t = self.remaining_tasks[i]
                prio_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "urgent": 1.0}
                obs.extend([
                    float(t.workhours) / 8.0,
                    prio_map.get(t.priority, 0.5),
                    task_params["embedding"],
                    1.0
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action_type, break_bins = action[0], action[1]
        reward = 0.0
        terminated = False
        truncated = False

        if action_type == 0:
            break_duration = (break_bins+1)*5/60

            # penalty (beginning/end of day break, too many/few breaks)
            # reward (break for eating midday)
        else:
            task_idx = action_type - 1
            if task_idx >= len(self.remaining_tasks):
                reward -= 10.0 # TODO: check number
                return self._get_obs(), reward, terminated, truncated, {}

            task = self.remaining_tasks.pop(task_idx)
            lag = calculate_switch_lag(self.last_task_type, task.type)
            if lag > 0:
                reward -= (lag * 6.0)

            attention_at_start = self.training_simulator.get_current_attention(self.current_time_in_day)
            energy_at_start = self.training_simulator.get_current_energy(self.current_time_in_day)

            actual_duration, end_time, energy_used = self.simulator.execute_task(
                task=task,
                start_time=self.current_time_in_day,
                context_switch_lag=lag
            )

            # save to db
            #penalty (task execution time exceeded/finished early - exponential, disruption, overtime)

        if len(self.remaining_tasks) == 0:
            reward += 60.0
            terminated = True
        elif self.current_day >= self.total_days:
            reward -= len(self.remaining_tasks) * 20.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _advance_to_next_day(self):
        worked_today = max(0.1, self.current_time_in_day - self.work_start_hour)
        break_pct = self.total_break_time_today / worked_today
        if 0.10 <= break_pct <= 0.15:
            # Prawidłowy bilans przerw
            pass
        elif break_pct < 0.10:
            pass  # Lekka kara za zbyt mało przerw
        else:
            pass  # Kara za nadmiar przerw

        self.current_day += 1
        self.current_time_in_day = self.work_start_hour
        self.time_since_last_break = 0.0
        self.total_break_time_today = 0.0
        self.last_task_type = None
        self.training_simulator.reset(weekly=(self.current_day % 5 == 0))


class PPOPlanner:
    def __init__(self):
        self.list = None