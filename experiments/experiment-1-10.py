import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from data.DbHelper import MonthSimulationSession
from data.DbModels import User, Task
from data.database import SessionLocal
from models.BaselinePlanner import BaselinePlanner
from models.NSGAPlanner import NSGAPlanner
from models.PPOPlanner import PPOPlanner


def load_finished_user_ids(file_path: str = "finished_user_ids.json") -> list[int]:
    """Load user ids from JSON."""
    path = Path(file_path)
    if not path.exists():
        print(f"File {file_path} not existent.")
        return []

    with open(path, "r", encoding="utf-8") as f:
        user_ids: list[int] = json.load(f)

    return user_ids


def prepare_data_for_online_phase(phase: str) -> tuple[list[User], dict[int, list[Task]]]:
    if phase != 'online':
        raise ValueError(f"Unavailable phase: '{phase}'. possible values: 'online'.")

    with SessionLocal() as session:
        user_ids = load_finished_user_ids()

        users = session.query(User).filter_by(is_training=False).filter(User.id.in_(user_ids[225:250])).all()

        tasks_records = (
            session.query(Task)
            .filter(Task.phase_order < 6, Task.phase == phase)  # only from a year
            .order_by(Task.phase_order, Task.deadline, Task.priority)
            .all()
        )

        tasks = defaultdict(list)
        for t in tasks_records:
            tasks[t.phase_order].append(t)

        session.expunge_all()

    return users, dict(tasks)


def run_experiment():
    group_id = 0
    users, tasks = prepare_data_for_online_phase('online')
    for algorithm in ['ppo', 'nsga', 'baseline']:  # do one set for algorithm
        print(f"\n\n------------------------------------------------------------------------------------------------")
        print(f"                                      {algorithm} ")
        print(f"------------------------------------------------------------------------------------------------")
        for user in users:  # iterate through 300 users
            print(f"-------------------------------------- USER {user.id} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} --------------------------------------")
            if algorithm == 'baseline':
                planner = BaselinePlanner(user)
            elif algorithm == 'ppo':
                planner = PPOPlanner(user)
            elif algorithm == 'nsga':
                planner = NSGAPlanner(user)
            else:
                raise ValueError(f"Unavailable algorithm: '{algorithm}'")

            start_date = datetime(year=2027, month=1, day=4)
            for phase_order, month_tasks in tasks.items():  # iterate through all months of tasks
                print(f"Month {phase_order} | tasks {len(month_tasks)}")
                sim_session = MonthSimulationSession(user, algorithm, 'online', phase_order, group_id)
                res = planner.plan_and_simulate_month(session=sim_session,  user=user, month_tasks=month_tasks,
                                                      group_id=group_id, phase="online", phase_order=phase_order,
                                                      start_date=start_date)

                sim_session.compute_and_save_to_db(SessionLocal, res["total_replans"], res["days_used"])
                start_date = start_date + timedelta(days=28)
                group_id += 1

            print(f"----------------------------------- USER {user.id}: END -----------------------------------")


if __name__ == '__main__':
    run_experiment()