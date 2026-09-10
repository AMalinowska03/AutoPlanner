import math
from datetime import datetime


from data.DbModels import User, Task

CHRONOTYPES = {
    "morning_lark": {
        "peak_attention_factor": 9,
    },
    "intermediate": {
        "peak_attention_factor": 12,
    },
    "night_owl": {
        "peak_attention_factor": 16,
    },
}

SKILL_ATTR_MAP = {
    "routine": {"attr_name": "routine_skill", "embedding": 0.2},
    "communication": {"attr_name": "communication_skill", "embedding": 0.4},
    "creativity": {"attr_name": "creativity_skill", "embedding": 0.6},
    "technical": {"attr_name": "technical_skill", "embedding": 0.8},
    "analytical": {"attr_name": "analytical_skill", "embedding": 1.0},
}

SWITCH_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    # ------------------ from: COMMUNICATION ------------------
    ("communication", "communication"): (0.0, 1.0),
    ("communication", "routine"): (0.1, 5 / 60),        # 5 min (switch to mentally easier task)
    ("communication", "creativity"): (0.2, 15 / 60),    # 15 min (wyciszenie i zmiana trybu na generatywny)
    ("communication", "technical"): (0.6, 15 / 60),     # 15 min (przejście do logiki implementacyjnej)
    ("communication", "analytical"): (0.6, 20 / 60),    # 20 min (wejście w stan głębokiej dedukcji po rozmowach)

    # ------------------ from: CREATIVITY ------------------
    ("creativity", "creativity"): (0.0, 1.0),
    ("creativity", "routine"): (0.1, 5 / 60),           # 5 min
    ("creativity", "communication"): (0.4, 10 / 60),    # 10 min (wyjście ze stanu flow do interakcji)
    ("creativity", "technical"): (0.6, 20 / 60),        # 20 min (przestawienie z myślenia dywergencyjnego na syntaktyczne)
    ("creativity", "analytical"): (0.8, 25 / 60),       # 25 min (największy koszt poznawczy: kreacja -> ścisła weryfikacja)

    # ------------------ from: TECHNICAL ------------------
    ("technical", "technical"): (0.0, 1.0),
    ("technical", "routine"): (0.2, 5 / 60),            # 5 min
    ("technical", "communication"): (0.4, 10 / 60),     # 10 min (wybicie z kodu/architektury do rozmowy)
    ("technical", "analytical"): (0.4, 10 / 60),        # 10 min (pokrewne domeny ścisłe, mały narzut)
    ("technical", "creativity"): (0.4, 20 / 60),        # 20 min (przejście z wąskich reguł technicznych do otwartej kreacji)

    # ------------------ from: ROUTINE ------------------
    ("routine", "routine"): (0.0, 1.0),
    ("routine", "communication"): (0.2, 5 / 60),        # 5 min (łatwe przejście z zadań odtwórczych)
    ("routine", "technical"): (0.8, 15 / 60),           # 15 min (wejście w wysokie skupienie ze stanu niskiego wysiłku)
    ("routine", "analytical"): (0.8, 15 / 60),          # 15 min
    ("routine", "creativity"): (0.4, 15 / 60),          # 15 min

    # ------------------ from: ANALYTICAL ------------------
    ("analytical", "analytical"): (0.0, 1.0),
    ("analytical", "routine"): (0.2, 5 / 60),           # 5 min
    ("analytical", "technical"): (0.4, 10 / 60),        # 10 min (pokrewny tryb skupienia)
    ("analytical", "communication"): (0.4, 10 / 60),    # 10 min (wyjście z analizy danych)
    ("analytical", "creativity"): (0.4, 25 / 60),       # 25 min (przejście ze ścisłych reguł konwergencyjnych do swobodnej kreacji)
}
PRODUCTIVITY_REF = 0.8
SKILL_REF = 0.5
SKILL_SPEED_WEIGHT = 0.5
ENERGY_PER_WORKHOUR = 0.05
DEBT_THRESHOLD = 0.3


def calculate_switch_lag(prev_task_type: str | None, current_task_type: str) -> tuple[float, float]:
    if prev_task_type is None:
        return 0.0, 1.0
    return SWITCH_MATRIX.get((prev_task_type, current_task_type), (0.4, 0.25))


class UserSimulator:
    def __init__(self, user: User):
        self.user_profile = user
        self.start_energy = 1
        self.task_energy_usage = 0
        self.energy_debt = 0
        self.peak_attention_factor = CHRONOTYPES[self.user_profile.chronotype]["peak_attention_factor"]

    def _get_skill_level(self, task_type: str) -> float:
        """Retrieves skill level of user for given task type (0.5 by default)."""
        task_params = SKILL_ATTR_MAP.get(task_type)
        if task_params and hasattr(self.user_profile, task_params["attr_name"]):
            val = getattr(self.user_profile, task_params["attr_name"])
            return float(val) if val is not None else 0.5
        return 0.5

    def process_passive_energy_usage(self, time):
        """
        Energy left at given time if there was only passive energy usage
        :param time: hour of day expressed in decimal format (example: 14.5 -> 14:30)
        :return:
        """
        start_hour = (
            self.user_profile.work_start_time.hour + self.user_profile.work_start_time.minute / 60.0
            if isinstance(self.user_profile.work_start_time, datetime)
            else float(self.user_profile.work_start_time or 8.0)
        )
        return self.start_energy - 0.01 * max(0.0, time - start_hour)

    def get_current_energy(self, time: float) -> float:
        """
        Energy level of user based on given timestamp
        dependent on cumulative energy usage and energy used for performing tasks
        :param time: `float` time expressed in decimal format (example: 14.5 -> 14:30)
        :return: `float` current energy level
        """
        energy = self.process_passive_energy_usage(time)
        energy -= self.task_energy_usage
        return max(0.0, min(1.0, energy))

    def get_current_attention(self, time: float) -> float:
        """
        User attention for given timestamp
        based on user's circadian rhythm and peak attention time distributed as a sinusoid
        :param time: `float` time expressed in decimal format (example: 14.5 -> 14:30)
        :return:
        """
        amplitude = 0.15
        return amplitude * math.cos(2*math.pi*(time - self.peak_attention_factor)/24)

    def get_productivity(self, time: float) -> float:
        """
        User productivity at given timestamp dependent on current energy level and attention
        :param time:
        :return:
        """
        return min(1.0, max(0.0, (self.get_current_energy(time) + self.get_current_attention(time))))

    def execute_task(self, task: Task, start_time: float, prev_task_type: str | None = None) -> tuple[float, float, float]:
        """
        Processes energy usage and time needed for a given task at a given timestamp
        adding it toward task energy usage for the day
        :param task:
        :param start_time:
        :param prev_task_type: type of previous task to calculate lag
        :return:
        """
        dt = 0.017  # around a minute, step duration for performing task
        work_remaining = float(task.workhours)
        skill = self._get_skill_level(task.type)
        skill_speed_factor = 1 + SKILL_SPEED_WEIGHT * (skill - SKILL_REF)

        # standard_drain_rate = base_drain_rate * (1.5 - skill)

        current_time = start_time
        total_task_energy = 0.0

        while work_remaining > 0:
            elapsed = current_time - start_time
            lag_cost, lag_duration = calculate_switch_lag(prev_task_type, task.type)
            in_switch_phase = elapsed <= lag_duration and  lag_cost > 0 and lag_duration > 0

            # current execution efficiency for given time
            productivity = self.get_productivity(current_time)
            effective_speed = min(1.5, max(0.25, productivity/PRODUCTIVITY_REF * skill_speed_factor))

            # if we are in context switch window we slow down task execution,
            # exponentially changing the impact to represent adjustment period
            if in_switch_phase:
                # impact on both execution speed and energy drain
                effective_speed *= 1 - lag_cost*math.exp(-elapsed/lag_duration)

            # how much work was performed in given time, depending on productivity and skill level
            work_done = effective_speed * dt
            work_remaining -= work_done

            # energy usage in given time step
            step_energy = ENERGY_PER_WORKHOUR * work_done
            self.task_energy_usage += step_energy
            total_task_energy += step_energy

            current_time += dt

        actual_duration = current_time - start_time
        return actual_duration, current_time, total_task_energy

    def process_break(self, duration: float, time: float):
        """
        Recovers task energy usage for a break at given time based on recovery factor
        used asymptotic function so
        :param duration:
        :param time:
        :return:
        """
        recovery_factor = 0.5
        break_energy_restoration = ((self.start_energy - self.get_current_energy(time)) *
                                    (1 - math.exp(-recovery_factor * duration)))
        self.task_energy_usage = max(0.0, self.task_energy_usage - break_energy_restoration)

    def reset(self, current_time: float, user_work_end_time: float, weekly=False):
        """
        Reset energy counters for user after a day/week
        :param current_time: time of day after performing tasks
        :param user_work_end_time: time of day when user finishes work:
        :param weekly:
        :return:
        """
        if current_time >= user_work_end_time and current_time - user_work_end_time >= DEBT_THRESHOLD:
            end_energy = self.get_current_energy(current_time)
            deficit_ratio = (DEBT_THRESHOLD - end_energy) / DEBT_THRESHOLD

            self.energy_debt = max(0.0, min(0.4, 0.2 * deficit_ratio * (1.0 - end_energy)))

        self.task_energy_usage = 0
        if weekly:
            self.energy_debt = 0
            self.start_energy = 1
        else:
            self.start_energy = 1 - self.energy_debt

