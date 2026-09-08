from data.DbModels import User, Task
from data.database import SessionLocal
from models.NSGAPlanner import NSGAPlanner
from models.PPOPlanner import PPOPlanner
from collections import defaultdict


def prepare_data_for_training(phase: str) -> tuple[list[User], dict[int, list[Task]]]:
    if phase not in ('pretrain', 'finetune'):
        raise ValueError(f"Unavailable phase: '{phase}'. possible values: 'pretrain', 'finetune'.")

    with SessionLocal() as session:
        is_training = (phase == 'pretrain')
        users = session.query(User).filter_by(is_training=is_training).all()

        tasks_records = (
            session.query(Task)
            .filter_by(phase=phase)
            .order_by(Task.phase_order)
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
    if phase == 'pretrain':
        planner.pretrain(users, tasks)
    elif phase == 'finetune':
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
            planner.finetune(user)

    print(f"------------------------- NSGA {phase}: END -------------------------")


if __name__ == '__main__':
    for phase in ('pretrain', 'finetune'):
        users, tasks = prepare_data_for_training(phase)
        run_ppo_training(phase, users, tasks)
        choose_nsga_params(phase, users, tasks)