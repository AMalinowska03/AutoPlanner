import math
import random

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
class UserSimulator:
    def __init__(self, user_profile):
        self.user_profile = user_profile
        self.start_energy = 1
        self.task_energy_usage = 0
        self.energy_debt = 0
        self.peak_attention_factor = CHRONOTYPES[user_profile.chronotype]["peak_attention_factor"]

    def get_current_energy(self, time: float) -> float:
        """
        Energy level of user based on given timestamp
        dependent on cumulative energy usage and energy used for performing tasks
        :param time: `float` time expressed in decimal format (example: 14.5 -> 14:30)
        :return: `float` current energy level
        """
        energy = self.process_cumulative_energy_usage(time)
        return energy - self.task_energy_usage

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

    def process_task_energy(self, task: float):
        print("----- PROCESS TASK ENERGY: start -----")
        self.task_energy_usage -= 0.1

    def process_task_duration(self, task, time):
        print("----- PROCESS TASK DURATION: start -----")
        efficiency = self.get_productivity(time)

    def process_cumulative_energy_usage(self, time):
        print("----- PROCESS Cumulative ENERGY: start -----")
        return self.start_energy

    def process_break(self, duration: float, time: float):
        """
         TODO: change so it is not linear as first 15 min of a break have more impact
        Recovers task energy usage for a break at given time based on recovery factor
        :param duration:
        :param time:
        :return:
        """
        recovery_factor = 0.25
        self.task_energy_usage += recovery_factor * (self.start_energy - self.get_current_energy(time)) * duration

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
