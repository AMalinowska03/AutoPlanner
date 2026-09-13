# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime
#
# from pymoo.algorithms.moo.nsga3 import NSGA3
# from pymoo.optimize import minimize
# from pymoo.util.ref_dirs import get_reference_directions
# from pymoo.termination.default import DefaultMultiObjectiveTermination
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
#
# from data.database import SessionLocal
# from data.DbModels import User, Task
# from data.DbHelper import get_user_work_hours
# from models.NSGAPlanner import PlanOptimizationProblem, PARETO_SELECTION_WEIGHTS
#
#
# def run_single_nsga_history_test(n_gen_max: int = 100, n_partitions: int = 6):
#     # 1. Pobierz 1 przykładowego użytkownika i pulę zadań
#     with SessionLocal() as session:
#         user = session.query(User).filter_by(is_training=False).first()
#         tasks = (
#             session.query(Task)
#             .filter_by(phase="online")
#             .filter(Task.phase_order == 0)
#             .all()
#         )
#
#     print(f"Test dla użytkownika #{user.id} ({user.chronotype}), liczba zadań: {len(tasks)}")
#
#     work_start_hour, work_end_hour = get_user_work_hours(user)
#     start_date = datetime(2027, 1, 4)
#
#     # 2. Skonfiguruj problem i algorytm z zapisem historii
#     ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
#     algorithm = NSGA3(
#         pop_size=len(ref_dirs),
#         ref_dirs=ref_dirs,
#         crossover=SBX(prob=0.998, eta=15),
#         mutation=PM(eta=24)
#     )
#
#     problem = PlanOptimizationProblem(
#         tasks=tasks, previous_plan=[], start_date=start_date,
#         sim_day=0, sim_time=work_start_hour, time_since_break=0.0, total_break_time=0.0,
#         last_task_type=None, work_start_hour=work_start_hour, work_end_hour=work_end_hour, total_days=20
#     )
#
#     print(f"Uruchamianie optymalizacji do {n_gen_max} generacji (pop_size={len(ref_dirs)})...")
#     res = minimize(
#         problem,
#         algorithm,
#         termination=DefaultMultiObjectiveTermination(n_max_gen=n_gen_max),
#         seed=1,
#         verbose=False,
#         save_history=True  # KLUCZOWE: zapisuje stan każdej generacji
#     )
#
#     # 3. Wyciągnij jakość najlepszego planu w każdej generacji
#     history = res.history
#     generations = []
#     best_scalar_scores = []
#     f1_deadlines = []
#     f2_efficiencies = []
#     f3_stabilities = []
#     f4_healths = []
#
#     for gen_idx, gen_pop in enumerate(history):
#         F = gen_pop.pop.get("F")
#         if F is None or len(F) == 0:
#             continue
#
#         # Normalizacja i wybór najlepszego osobnika dokładnie wg Twojej wagi Pareto
#         f_min = F.min(axis=0)
#         f_max = F.max(axis=0)
#         ranges = f_max - f_min
#         ranges[ranges == 0.0] = 1.0
#         norm_f = (F - f_min) / ranges
#         scores = norm_f @ PARETO_SELECTION_WEIGHTS
#         best_idx = np.argmin(scores)
#
#         generations.append(gen_idx + 1)
#         best_scalar_scores.append(scores[best_idx])
#         f1_deadlines.append(F[best_idx, 0])
#         f2_efficiencies.append(F[best_idx, 1])
#         f3_stabilities.append(F[best_idx, 2])
#         f4_healths.append(F[best_idx, 3])
#
#     # 4. Rysowanie wykresów zbieżności
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#
#     # Wykres A: Zbieżność zagregowanego wyniku jakości (im niższy, tym lepiej)
#     axes[0].plot(generations, best_scalar_scores, color="#1f77b4", lw=2)
#     axes[0].set_title("Zbieżność Jakości Planu (Skalaryzowany Score Pareto)")
#     axes[0].set_xlabel("Numer generacji (n_gen)")
#     axes[0].set_ylabel("Wskaźnik kosztu (niższy = lepszy)")
#     axes[0].grid(True, alpha=0.3)
#
#     # Zaznaczmy przykładowe punkty odcięcia
#     for gen_mark in [35, 50, 75]:
#         if gen_mark <= len(generations):
#             val = best_scalar_scores[gen_mark - 1]
#             axes[0].axvline(x=gen_mark, color="red", linestyle="--", alpha=0.6)
#             axes[0].text(gen_mark + 1, val, f"gen={gen_mark}\n({val:.2f})", fontsize=8)
#
#     # Wykres B: Zbieżność poszczególnych funkcji celu (F1: Deadline, F2: Efektywność, F4: Zdrowie)
#     axes[1].plot(generations, f1_deadlines, label="F1 (Kary za terminy)", lw=1.8)
#     axes[1].plot(generations, f2_efficiencies, label="F2 (Dopasowanie czasu)", lw=1.8)
#     axes[1].plot(generations, f4_healths, label="F4 (Zmęczenie / przerwy)", lw=1.8)
#     axes[1].set_title("Przebieg Celów Składowych w Kolejnych Generacjach")
#     axes[1].set_xlabel("Numer generacji (n_gen)")
#     axes[1].set_ylabel("Wartość funkcji celu (surowa)")
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig("zbieznosc_nsga_generacje.png", dpi=300)
#     plt.close()
#
#     print("Wykres zapisano do: zbieznosc_nsga_generacje.png")
#
#     # Wypisz tabelę logów w konsoli
#     print("\n" + "="*50)
#     print("ANALIZA PŁASKOWYŻU (PLATEAU):")
#     print(f"{'Generacja':<12} | {'Score Pareto':<15} | {'Zysk vs Poprzedni (%)':<20}")
#     print("-" * 50)
#     for g in [20, 35, 50, 75, 99]:
#         if g <= len(best_scalar_scores):
#             score_curr = best_scalar_scores[g - 1]
#             if g == 20:
#                 print(f"{g:<12} | {score_curr:<15.4f} | {'-':<20}")
#             else:
#                 score_prev = best_scalar_scores[prev_g - 1]
#                 gain = ((score_prev - score_curr) / max(abs(score_prev), 1e-6)) * 100.0
#                 print(f"{g:<12} | {score_curr:<15.4f} | {f'{gain:+.2f}%':<20}")
#             prev_g = g
#     print("="*50 + "\n")
#
#
# if __name__ == "__main__":
#     run_single_nsga_history_test(n_gen_max=100, n_partitions=6)

#
# from collections import defaultdict
# import numpy as np
# from data.database import SessionLocal
# from data.DbModels import User, Task
# from models.PPOEnv import PPOPlannerEnv, FlattenMultiDiscreteActionWrapper, GymnasiumToGymWrapper
#
#
# def diagnose_single_episode():
#     with SessionLocal() as session:
#         user = session.query(User).filter_by(is_training=True).first()
#         raw_tasks = session.query(Task).filter_by(phase="pretrain").all()
#         session.expunge_all()
#
#     # Create task pool dictionary identical to training
#     tasks_dict = {0: raw_tasks[:80]}
#
#     raw_env = PPOPlannerEnv(users_pool=[user], divided_tasks=tasks_dict, max_tasks_count=50)
#     env = FlattenMultiDiscreteActionWrapper(raw_env, num_time_bins=12)
#     env = GymnasiumToGymWrapper(env)
#
#     obs = env.reset()
#     done = False
#
#     reward_breakdown = defaultdict(float)
#     step_count = 0
#
#     # Monkey-patch raw_env step calculations to track reward sources
#     orig_break_rew = raw_env._calculate_break_reward
#     orig_deadline_rew = raw_env._calculate_deadline_reward
#     orig_time_rew = raw_env._calculate_time_allotment_reward
#     orig_disrupt_rew = raw_env._calculate_disruption_reward
#     orig_overtime_rew = raw_env._calculate_overtime_reward
#     orig_end_break_rew = raw_env._calculate_end_day_break_reward
#
#     tracked = defaultdict(float)
#
#     def wrap_fn(name, orig_fn):
#         def wrapped(*args, **kwargs):
#             val = orig_fn(*args, **kwargs)
#             tracked[name] += val
#             return val
#
#         return wrapped
#
#     raw_env._calculate_break_reward = wrap_fn("1. Przerwy (break_reward)", orig_break_rew)
#     raw_env._calculate_deadline_reward = wrap_fn("2. Terminy (deadline_reward)", orig_deadline_rew)
#     raw_env._calculate_time_allotment_reward = wrap_fn("3. Dopasowanie czasu (allotment)", orig_time_rew)
#     raw_env._calculate_disruption_reward = wrap_fn("4. Niestabilność (disruption)", orig_disrupt_rew)
#     raw_env._calculate_overtime_reward = wrap_fn("5. Nadgodziny (overtime)", orig_overtime_rew)
#     raw_env._calculate_end_day_break_reward = wrap_fn("6. Bilans przerw dnia (end_break)", orig_end_break_rew)
#
#     total_env_reward = 0.0
#
#     while not done:
#         # Sample legal random action (or insert policy action)
#         action = env.action_space.sample()
#         obs, rew, done, info = env.step(action)
#         total_env_reward += rew
#         step_count += 1
#
#     print("=" * 65)
#     print(f"DIAGNOSTYKA POJEDYNCZEGO EPIZODU (Kroki: {step_count})")
#     print(
#         f"Użytkownik #{user.id}, Pozostało niezrobionych zadań: {len(raw_env.remaining_tasks) + len(raw_env.backlog)}")
#     print("=" * 65)
#     print(f"{'Składowa Nagrody / Kary':<35} | {'Skumulowana Wartość':<15}")
#     print("-" * 65)
#
#     for name, val in sorted(tracked.items(), key=lambda x: x[1]):
#         print(f"{name:<35} | {val:<15.2f}")
#
#     print("-" * 65)
#     print(f"{'CAŁKOWITY ZWROT (EpRet)':<35} | {total_env_reward:<15.2f}")
#     print("=" * 65)
#
#
# if __name__ == "__main__":
#     diagnose_single_episode()


from data.database import SessionLocal
from data.DbModels import User, Task
from models.BaselinePlanner import BaselinePlanner
from models.NSGAPlanner import NSGAPlanner
from models.PPOPlanner import PPOPlanner

with SessionLocal() as session:
    user = session.query(User).filter_by(is_training=True, id=30).first()
    tasks = session.query(Task).filter_by(phase="online", phase_order=0).all()

planner = PPOPlanner(user)
planner.print_plan_after_train(user, tasks)

print("--------------------------------------------------")

planner = BaselinePlanner(user)
planner.print_plan_after_train(user, tasks)


print("--------------------------------------------------")

planner = NSGAPlanner(user)
planner.print_plan_after_train(user, tasks)
