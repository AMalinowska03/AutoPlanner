from datetime import timedelta

from data.DbModels import PlanTask, Execution
from data.database import SessionLocal


def save_execution_to_db(user, current_day, plan_record, task, abs_start_hour, end_hour_in_day, energy_used):
    session = SessionLocal()
    try:
        base_date = user.work_start_time
        start_dt = base_date + timedelta(hours=abs_start_hour)
        abs_end_hour = current_day * 24.0 + end_hour_in_day
        end_dt = base_date + timedelta(hours=abs_end_hour)

        plan_task = session.query(PlanTask).filter(
            PlanTask.plan_id == plan_record.id,
            PlanTask.task_id == task.id
        ).first()

        if plan_task:
            exec_rec = Execution(
                plan_task_id=plan_task.id,
                start_time=start_dt,
                end_time=end_dt,
                energy=energy_used
            )
            session.add(exec_rec)
            session.commit()
    except Exception as e:
        session.rollback()
    finally:
        session.close()