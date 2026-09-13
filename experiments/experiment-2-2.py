import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Any, Dict

from data.DbHelper import build_disruptors_map, MonthSimulationSession
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


def prepare_data_for_disruptions_phase(phase: str) -> tuple[Any, dict[int, list], dict[int, list]]:
    if phase != 'disruptions':
        raise ValueError(f"Unavailable phase: '{phase}'. possible values: 'disruptions'.")

    with SessionLocal() as session:
        user_ids = load_finished_user_ids()
        users = session.query(User).filter_by(is_training=False).filter(User.id.in_(user_ids[25:50])).all()

        tasks_records = (
            session.query(Task)
            .filter(Task.phase_order < 3, Task.phase == phase)  # only from 3 months
            .filter_by(phase=phase, is_disruptor=False)
            .order_by(Task.phase_order, Task.deadline, Task.priority)
            .all()
        )

        tasks = defaultdict(list)
        for t in tasks_records:
            tasks[t.phase_order].append(t)

        disruptor_task_records = (
            session.query(Task)
            .filter(Task.phase_order < 3, Task.phase == phase)  # only from 3 months
            .filter_by(phase=phase, is_disruptor=True)
            .order_by(Task.phase_order, Task.deadline, Task.priority)
            .all()
        )
        disruptor_tasks = defaultdict(list)
        for dt in disruptor_task_records:
            disruptor_tasks[dt.phase_order].append(dt)

        session.expunge_all()

    return users, dict(tasks), dict(disruptor_tasks)


def run_experiment():
    group_id = 252001
    users, tasks, disruptors_tasks = prepare_data_for_disruptions_phase('disruptions')
    for algorithm in ['ppo', 'nsga', 'baseline']:
        print(f"\n\n------------------------------------------------------------------------------------------------")
        print(f"                                      {algorithm} ")
        print(f"------------------------------------------------------------------------------------------------")
        for user in users:
            print(f"-------------------------------------- USER {user.id} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} --------------------------------------")
            if algorithm == 'baseline':
                planner = BaselinePlanner(user)
            elif algorithm == 'ppo':
                planner = PPOPlanner(user)
            elif algorithm == 'nsga':
                planner = NSGAPlanner(user)
            else:
                raise ValueError(f"Unavailable algorithm: '{algorithm}'")

            start_date = datetime(year=2028, month=9, day=11)
            for phase_order, month_tasks in tasks.items():
                sim_session = MonthSimulationSession(user, algorithm, 'disruptions', phase_order, group_id)
                disruptors_map = build_disruptors_map(disruptors_tasks[phase_order], start_date)
                res = planner.plan_and_simulate_month(session=sim_session, user=user, month_tasks=month_tasks,
                                                      group_id=group_id, phase="disruptions", phase_order=phase_order,
                                                      start_date=start_date, disruptors_map=disruptors_map)

                sim_session.compute_and_save_to_db(SessionLocal, res["total_replans"], res["days_used"])
                start_date = start_date + timedelta(days=28)
                group_id += 1
            print(f"-------------------------------------- USER {user.id} :END {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} --------------------------------------")


if __name__ == '__main__':
    run_experiment()
