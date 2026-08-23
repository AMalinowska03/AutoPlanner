class TimeSimulator:
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    WEEKEND = 6
    def __init__(self):
        weekday = self.MONDAY
        time = 0  # time of day expressed as the current minute
