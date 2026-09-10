from datetime import timedelta, datetime, time
from typing import Optional

from data.DbModels import PlanTask, Execution, Task, User, Plan
from data.database import SessionLocal

class Repository:
    def __init__(self, user: User, phase, phase_order, phase_order_start):
        self.session_maker = SessionLocal
        self.user = user
        self.phase = phase
        self.phase_order = phase_order
        self.phase_order_start = phase_order_start

    def create_plan_records(self, algorithm: str, planned_tasks: list, group_id: int, generation: int,
                            generating_time: float, disruption_time: Optional[datetime]) -> Plan:
        session = self.session_maker()
        try:
            plan = Plan(algorithm=algorithm, group=group_id, generation=generation, disruption_time=disruption_time,
                        generating_time=generating_time)
            session.add(plan)
            session.flush()

            # planned_tasks is a list of elements like: (task_id, start_time, end_time)
            for t_data in planned_tasks:
                if t_data.get("is_break"):
                    # saving just tasks, breaks are saved when they are actually performed
                    continue
                pt = PlanTask(
                    plan_id=plan.id,
                    user_id=self.user.id,
                    task_id=t_data['task_id'],
                    start_time=t_data['start_time'],
                    end_time=t_data['end_time']
                )
                session.add(pt)
            session.commit()
            return plan
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    def save_break(self, break_duration: float, plan_record: Plan, start_hour_float: float, current_day: datetime):
        session = self.session_maker()
        try:
            task = Task(name="Break", workhours=break_duration, priority="low", phase=self.phase,
                        phase_order=self.phase_order, is_break=True)
            session.add(task)
            session.flush()

            h = int(start_hour_float)
            m = int((start_hour_float - h) * 60)
            start_time = datetime(current_day.year, current_day.month, current_day.day, h, m)

            plan_task = PlanTask(
                plan_id=plan_record.id, task_id=task.id, user_id=self.user.id,
                start_time=start_time, end_time=start_time + timedelta(hours=break_duration)
            )
            session.add(plan_task)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    def save_plan_task(self, task_id, start_time, end_time):
        session = self.session_maker()
        try:
            plan_task = PlanTask(task_id=task_id, user_id=self.user.id, start_time=start_time, end_time=end_time)
            session.add(plan_task)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    def save_execution_to_db(self, plan_record, task, start_date, end_date, energy_used):
        session = self.session_maker()
        try:
            plan_task = session.query(PlanTask).filter(
                PlanTask.plan_id == plan_record.id,
                PlanTask.task_id == task.id,
                PlanTask.user_id == self.user.id,
            ).first()

            if plan_task:
                execution = Execution(
                    plan_task_id=plan_task.id,
                    start_time=start_date,
                    end_time=end_date,
                    energy=energy_used
                )
                session.add(execution)
                session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()


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
