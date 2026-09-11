import random
from typing import Optional

from data.DbModels import User, Task
from data.database import SessionLocal
from models.NSGAPlanner import NSGAPlanner
from models.PPOPlanner import PPOPlanner
from collections import defaultdict


def prepare_data_for_training(
        phase: str,
        chronotype: Optional[str] = None,
        users_per_chronotype: int = 100,
        max_phase_order: int = 12,
        seed: int = 42
) -> tuple[list[User], dict[int, list[Task]]]:
    if phase not in ('pretrain', 'finetune'):
        raise ValueError(f"Unavailable phase: '{phase}'. possible values: 'pretrain', 'finetune'.")

    with SessionLocal() as session:
        is_training = (phase == 'pretrain')

        if phase == 'finetune':
            if chronotype is not None:
                candidate_users = (
                    session.query(User)
                    .filter_by(is_training=is_training, chronotype=chronotype)
                    .all()
                )
                rng = random.Random(seed)
                users = rng.sample(candidate_users, min(users_per_chronotype, len(candidate_users)))
            else:
                all_users = session.query(User).filter_by(is_training=is_training).all()
                users_by_chrono = defaultdict(list)
                for u in all_users:
                    users_by_chrono[u.chronotype].append(u)

                rng = random.Random(seed)
                users = []
                for chrono, group in sorted(users_by_chrono.items()):
                    sample_size = min(users_per_chronotype, len(group))
                    users.extend(rng.sample(group, sample_size))
        else:
            users = session.query(User).filter_by(is_training=is_training).all()

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

def run_ppo_training(phase: str, users: list[User], tasks: dict[int, list[Task]]):
    print(f"------------------------- PPO {phase} -------------------------")
    planner = PPOPlanner()
    # if phase == 'pretrain':
    #     planner.pretrain(users, tasks)
    if phase == 'finetune':
        for user in users:
            planner.finetune_user(user, tasks)

    print(f"------------------------- PPO {phase}: END -------------------------")


def choose_nsga_params(phase: str, users: list[User], tasks: dict[int, list[Task]]):
    print(f"------------------------- NSGA {phase} -------------------------")
    planner = NSGAPlanner()
    if phase == 'pretrain':
        planner.pretrain(users, tasks)
    elif phase == 'finetune':
        for user in users:
            planner.finetune(user, tasks)

    print(f"------------------------- NSGA {phase}: END -------------------------")


if __name__ == '__main__':
    for phase in ('pretrain', 'finetune'):
        users, tasks = prepare_data_for_training(phase, chronotype='morning_lark', users_per_chronotype=100)
        # users, tasks = prepare_data_for_training(phase, chronotype='intermediate', users_per_chronotype=100)
        # users, tasks = prepare_data_for_training(phase, chronotype='night_owl', users_per_chronotype=100)
        run_ppo_training(phase, users, tasks)
        # choose_nsga_params(phase, users, tasks)