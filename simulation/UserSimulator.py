import math
import random
from datetime import datetime

from sqlalchemy.sql.functions import user

from data.DbModels import User, Task, Plan, PlanTask, Execution

CHRONOTYPES = {
    "morning_lark": {
        "peak_attention_factor": 3,
    },
    "intermediate": {
        "peak_attention_factor": 6,
    },
    "night_owl": {
        "peak_attention_factor": 10,
    },
}

SKILL_ATTR_MAP = {
    "communication": "communication_skill",
    "creativity": "creativity_skill",
    "technical": "technical_skill",
    "routine": "routine_skill",
    "analytical": "analytical_skill",
}

SWITCH_MATRIX: dict[tuple[str, str], float] = {
    # ------------------ Z: COMMUNICATION ------------------
    ("communication", "communication"): 0.0,
    ("communication", "routine"): 5 / 60,        # 5 min (rozładowanie uwagi na proste zadania)
    ("communication", "creativity"): 15 / 60,    # 15 min (wyciszenie i zmiana trybu na generatywny)
    ("communication", "technical"): 15 / 60,     # 15 min (przejście do logiki implementacyjnej)
    ("communication", "analytical"): 20 / 60,    # 20 min (wejście w stan głębokiej dedukcji po rozmowach)

    # ------------------ Z: CREATIVITY ------------------
    ("creativity", "creativity"): 0.0,
    ("creativity", "routine"): 5 / 60,           # 5 min
    ("creativity", "communication"): 10 / 60,    # 10 min (wyjście ze stanu flow do interakcji)
    ("creativity", "technical"): 20 / 60,        # 20 min (przestawienie z myślenia dywergencyjnego na syntaktyczne)
    ("creativity", "analytical"): 25 / 60,       # 25 min (największy koszt poznawczy: kreacja -> ścisła weryfikacja)

    # ------------------ Z: TECHNICAL ------------------
    ("technical", "technical"): 0.0,
    ("technical", "routine"): 5 / 60,            # 5 min
    ("technical", "communication"): 10 / 60,     # 10 min (wybicie z kodu/architektury do rozmowy)
    ("technical", "analytical"): 10 / 60,        # 10 min (pokrewne domeny ścisłe, mały narzut)
    ("technical", "creativity"): 20 / 60,        # 20 min (przejście z wąskich reguł technicznych do otwartej kreacji)

    # ------------------ Z: ROUTINE ------------------
    ("routine", "routine"): 0.0,
    ("routine", "communication"): 5 / 60,        # 5 min (łatwe przejście z zadań odtwórczych)
    ("routine", "technical"): 15 / 60,           # 15 min (wejście w wysokie skupienie ze stanu niskiego wysiłku)
    ("routine", "analytical"): 15 / 60,          # 15 min
    ("routine", "creativity"): 15 / 60,          # 15 min

    # ------------------ Z: ANALYTICAL ------------------
    ("analytical", "analytical"): 0.0,
    ("analytical", "routine"): 5 / 60,           # 5 min
    ("analytical", "technical"): 10 / 60,        # 10 min (pokrewny tryb skupienia)
    ("analytical", "communication"): 10 / 60,    # 10 min (wyjście z analizy danych)
    ("analytical", "creativity"): 25 / 60,       # 25 min (przejście ze ścisłych reguł konwergencyjnych do swobodnej kreacji)
}


def calculate_switch_lag(prev_task_type: str | None, current_task_type: str) -> float:
    if prev_task_type is None:
        return 0.0
    return SWITCH_MATRIX.get((prev_task_type, current_task_type), 10 / 60)


class UserSimulator:
    def __init__(self, user: User):
        self.user_profile = user
        self.start_energy = 1
        self.task_energy_usage = 0
        self.energy_debt = 0
        self.peak_attention_factor = CHRONOTYPES[self.user_profile.chronotype]["peak_attention_factor"]

    def _get_skill_level(self, task_type: str) -> float:
        """Retrieves skill level of user for given task type (0.5 by default)."""
        attr_name = SKILL_ATTR_MAP.get(task_type)
        if attr_name and hasattr(self.user_profile, attr_name):
            val = getattr(self.user_profile, attr_name)
            return float(val) if val is not None else 0.5
        return 0.5

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
        base_attention = 0.7
        amplitude = 0.3
        return base_attention + amplitude * math.sin(2*math.pi*(time - self.peak_attention_factor)/24)

    def get_productivity(self, time: float) -> float:
        """
        User productivity at given timestamp dependent on current energy level and attention
        :param time:
        :return:
        """
        return self.get_current_energy(time) * self.get_current_attention(time)

    def execute_task(self, task: Task, start_time: float, context_switch_lag: float = 0.0) -> tuple[float, float, float]:
        """
        Processes energy usage and time needed for a given task at a given timestamp
        adding it toward task energy usage for the day
        :param task:
        :param start_time:
        :param context_switch_lag: if task type changed, given time will be for adjustment, slowing down execution
        :return:
        """
        dt = 0.017  # around a minute, step duration for performing task
        work_remaining = float(task.workhours)
        skill = self._get_skill_level(task.type)
        skill_speed_factor = 0.5 + 0.5 * skill

        base_drain_rate = 0.05  # bazowe zużycie na godzinę przy wykonywaniu pracy
        standard_drain_rate = base_drain_rate * (1.5 - skill)

        current_time = start_time
        total_task_energy = 0.0

        while work_remaining > 0:
            elapsed = current_time - start_time
            in_switch_phase = elapsed < context_switch_lag

            # Aktualna wydajność w chwili t z uwzględnieniem skilla
            productivity = self.get_productivity(current_time)
            effective_speed = max(0.1, productivity * skill_speed_factor)

            # 2. Modyfikatory w fazie przełączania kontekstu
            if in_switch_phase:
                # Drastyczny spadek tempa realizacji właściwego zadania (np. 80% spowolnienia)
                effective_speed *= 0.20
                # Zwiększony drenaż energii przez wysiłek skupienia i zmianę reguł (1.5x)
                current_drain_rate = standard_drain_rate * 1.50
            else:
                current_drain_rate = standard_drain_rate

            # Postęp wykonany w kroku dt
            work_done = effective_speed * dt
            work_remaining -= work_done

            # Zużycie energii w kroku dt
            step_energy = current_drain_rate * dt
            self.task_energy_usage += step_energy
            total_task_energy += step_energy

            current_time += dt

        actual_duration = current_time - start_time
        return actual_duration, current_time, total_task_energy

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
        return self.start_energy - 0.02 * max(0.0, time - start_hour)

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
        self.task_energy_usage = min(0.0, self.task_energy_usage - break_energy_restoration)

    def reset(self, weekly=False):
        """
        Reset energy counters for user after a day/week
        :param weekly:
        :return:
        """
        if weekly:
            self.energy_debt = 0
            self.start_energy = 1
        else:
            self.start_energy = 1 - self.energy_debt

