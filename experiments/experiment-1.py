from collections import defaultdict

from data.DbModels import User, Task
from data.database import SessionLocal
from models.BaselinePlanner import BaselinePlanner
from models.NSGAPlanner import NSGAPlanner
from models.PPOPlanner import PPOPlanner


def prepare_data_for_online_phase(phase: str) -> tuple[list[User], dict[int, list[Task]]]:
    if phase != 'online':
        raise ValueError(f"Unavailable phase: '{phase}'. possible values: 'online'.")

    with SessionLocal() as session:
        users = session.query(User).filter_by(is_training=False).all()

        tasks_records = (
            session.query(Task)
            .filter_by(phase=phase)
            .order_by(Task.phase_order, Task.deadline, Task.priority)
            .all()
        )

        tasks = defaultdict(list)
        for t in tasks_records:
            tasks[t.phase_order].append(t)

        session.expunge_all()

    return users, dict(tasks)


def run_experiment():
    users, tasks = prepare_data_for_online_phase('online')
    for algorithm in ['ppo', 'nsga', 'baseline']:
        planner = None
        for user in users:
            if algorithm == 'baseline':
                planner = BaselinePlanner(user)
            elif algorithm == 'ppo':
                planner = PPOPlanner(user)
            elif algorithm == 'nsga':
                planner = NSGAPlanner(user)