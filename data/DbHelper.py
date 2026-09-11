from datetime import timedelta, datetime, time
from typing import Optional

from data.DbModels import Task, User
from data.database import SessionLocal

DisruptorsMap = dict[int, list[tuple[float, Task]]]

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from data.DbModels import ExperimentMetric
from evaluation.metrics import metrics


class MonthSimulationSession:
    def __init__(self, user, algorithm: str, phase: str, phase_order: int, group_id: int):
        self.user = user
        self.algorithm = algorithm
        self.phase = phase
        self.phase_order = phase_order
        self.group_id = group_id

        # memory containers
        self.plans_history: List[Dict[str, Any]] = []
        self.executed_tasks: List[Dict[str, Any]] = []
        self.generation_times: List[float] = []
        self.initial_planned_tasks_count: int = 0
        self.daily_completion_rates: List[float] = []

    def record_plan(self, planned_tasks: list, generation: int, generating_time: float,
                    disruption_time: Optional[datetime]):
        """
        Records plan iterations
        :param planned_tasks:
        :param generation:
        :param generating_time:
        :param disruption_time:
        :return:
        """
        self.generation_times.append(generating_time)

        tasks_map = {}
        for item in planned_tasks:
            if not item.get("is_break"):
                tasks_map[item["task_id"]] = {
                    "task_id": item["task_id"],
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                    "duration": item.get("duration", (item["end_time"] - item["start_time"]).total_seconds() / 3600.0),
                    "task": item["task"]
                }

        if generation == 0:
            self.initial_planned_tasks_count = len(tasks_map)

        self.plans_history.append({
            "generation": generation,
            "disruption_time": disruption_time,
            "tasks": tasks_map
        })

    def record_execution(self, task, planned_start: datetime, planned_end: datetime,
                         actual_start: datetime, actual_end: datetime, energy: float = 0.0):
        """
        Zapisuje wykonanie wraz z planem algorytmu z momentu rozpoczęcia zadania.
        Pozwala to porównać estymację algorytmu z fizyczną symulacją użytkownika.
        """
        self.executed_tasks.append({
            "task": task,
            "planned_start": planned_start,
            "planned_end": planned_end,
            "planned_duration_sec": (planned_end - planned_start).total_seconds(),
            "actual_start": actual_start,
            "actual_end": actual_end,
            "actual_duration_sec": (actual_end - actual_start).total_seconds(),
            "energy": energy
        })

    def compute_and_save_to_db(self, session_maker, total_replans: int, days_used: int):
        avg_gen_time = (
            sum(self.generation_times) / len(self.generation_times)
            if self.generation_times else 0.0
        )

        # calculate metrics
        tasks_executed_no_breaks = [e for e in self.executed_tasks if not getattr(e["task"], "is_break", False)]
        monthly_completion_sc = round(len(tasks_executed_no_breaks) / max(self.initial_planned_tasks_count, 1), 4)
        daily_completion_sc = metrics.daily_task_completion_score(self.plans_history, self.executed_tasks)
        estimation_err = metrics.task_time_estimation_score(self.executed_tasks)
        delay_sc = metrics.task_execution_delay_score(self.executed_tasks)
        energy_sc = metrics.energy_distribution_score(self.executed_tasks, self.user)
        switch_metrics = metrics.context_switch_score(self.executed_tasks)
        instability_sc = metrics.instability_score(self.plans_history)

        # save sumup of month to db
        with session_maker() as session:
            metric_record = ExperimentMetric(
                experiment_type=self.phase,
                algorithm=self.algorithm,
                user_id=self.user.id,
                phase_order=self.phase_order,
                group_id=self.group_id,
                total_replans=total_replans,
                days_used=days_used,
                avg_generating_time=round(avg_gen_time, 4),
                monthly_completion_score=monthly_completion_sc,
                daily_completion_score=daily_completion_sc,
                time_estimation_error=estimation_err,
                delay_score=delay_sc,
                energy_score=energy_sc,
                switch_efficiency=switch_metrics["efficiency_ratio"],
                instability=instability_sc
            )
            session.add(metric_record)
            session.commit()


def get_user_work_hours(user):
    work_start_hour = (
        user.work_start_time.hour + user.work_start_time.minute / 60.0
        if isinstance(user.work_start_time, datetime) else 8.0
    )
    work_end_hour = (
        user.work_end_time.hour + user.work_end_time.minute / 60.0
        if isinstance(user.work_end_time, datetime) else 16.0
    )
    return work_start_hour, work_end_hour


def sim_time_to_datetime(
        start_date: datetime,
        sim_day: int,
        hour_decimal: float,
) -> datetime:
    calendar_days = sim_day + (sim_day // 5) * 2

    base_day = start_date + timedelta(days=calendar_days)

    return datetime(
        base_day.year,
        base_day.month,
        base_day.day,
    ) + timedelta(hours=hour_decimal)



def build_disruptors_map(disruptor_tasks: list[Task], start_date: datetime) -> DisruptorsMap:
    disruptions_map = {}
    start_day = start_date.date()

    for task in disruptor_tasks:
        injection = task.injection_time

        if injection is None:
            continue

        injection_date = injection.date()

        calendar_days = (injection_date - start_day).days

        if calendar_days < 0:
            continue

        weeks = calendar_days // 7
        weekday = injection_date.weekday()

        if weekday >= 5:
            continue

        sim_day = weeks * 5 + weekday
        if not 0 <= sim_day < 20:
            continue

        injection_hour = (
            injection.hour
            + injection.minute / 60.0
            + injection.second / 3600.0
        )

        disruptions_map.setdefault(sim_day, []).append((injection_hour, task))

    # important because planners inspect [0]
    for day in disruptions_map:
        disruptions_map[day].sort(
            key=lambda item: item[0]
        )

    return disruptions_map
