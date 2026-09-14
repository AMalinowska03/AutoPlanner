import os
import copy
import io
import shelve
from datetime import datetime, timedelta
import torch

from data.database import SessionLocal
from data.DbModels import User, Task, BreakTask
from data.DbHelper import (
    get_user_work_hours,
    build_disruptors_map,
    sort_tasks_by_deadline_and_priority
)
from simulation.UserSimulator import UserSimulator
from models.PPOPlanner import PPOPlanner
from models.NSGAPlanner import NSGAPlanner
from models.BaselinePlanner import BaselinePlanner
from plot_results import load_data, get_sample_users_per_chronotype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ppo_load_model(planner, user):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

    pt_candidates = [
        os.path.join(project_root, f"PPOGenerated/ppo_user_{user.id}_active/pyt_save/model.pt"),
        os.path.join(project_root, f"PPOGenerated/finetune_u{user.id}/pyt_save/model.pt"),
        os.path.join(project_root, "PPOGenerated/pretrain/pyt_save/model.pt")
    ]
    for pt_path in pt_candidates:
        if os.path.isfile(pt_path):
            model = torch.load(pt_path, map_location=device, weights_only=False)
            model.eval()
            return model

    shelve_candidates = [
        os.path.join(project_root, f"experiments/db/models_store_u{user.id}.db"),
        os.path.join(project_root, "experiments/db/models_store.db"),
    ]
    target_keys = [f"ppo_user_{user.id}_active", f"ppo_user_{user.id}_finetuned", "ppo_base_pretrained"]

    for sh_path in shelve_candidates:
        base_clean = sh_path[:-3] if sh_path.endswith(".db") else sh_path
        if not (os.path.exists(sh_path) or os.path.exists(sh_path + ".dat") or os.path.exists(base_clean + ".dat")):
            continue
        try:
            with shelve.open(sh_path, flag='r') as db:
                for k in target_keys:
                    if k in db:
                        buffer = io.BytesIO(db[k])
                        model = torch.load(buffer, map_location=device, weights_only=False)
                        model.eval()
                        return model
        except Exception:
            continue

    raise FileNotFoundError(f"Nie znaleziono modelu dla użytkownika #{user.id}.")


def planner_instance_env(planner, user, start_date):
    from models.PPOEnv import PPOPlannerEnv
    return PPOPlannerEnv(users_pool=[user], divided_tasks={0: []}, max_tasks_count=50, planning_mode=True,
                         start_day=start_date)


def ppo_make_wrapped(raw_env):
    from models.PPOEnv import make_wrapped_env
    return make_wrapped_env(raw_env=raw_env)


PPOPlanner.make_wrapped = staticmethod(ppo_make_wrapped)
PPOPlanner.load_model = ppo_load_model


def run_full_disruption_generations(planner_instance, user, tasks_pool, disruptors_pool, start_date,
                                    target_replan_gens=5):
    """
    Prowadzi symulację do zarejestrowania Planu Bazowego (Gen 0) oraz
    dokładnie target_replan_gens (5) kolejnych planów po wstrzyknięciach zakłóceń.
    """
    work_start_hour, work_end_hour = get_user_work_hours(user)
    simulator = UserSimulator(user)
    disr_map = build_disruptors_map(disruptors_pool, start_date)

    remaining_tasks = sort_tasks_by_deadline_and_priority(copy.deepcopy(tasks_pool))
    executed_history = {}  # task_id -> {"actual_start", "actual_end"}
    plans_history = []  # gen -> plan
    disruption_times = {}  # gen -> datetime

    sim_day = 0
    sim_time = work_start_hour
    current_generation = 0
    disruption_occurrence_time = None

    time_since_last_break = 0.0
    total_break_time_today = 0.0
    last_task_type = None

    while remaining_tasks and sim_day < 20 and current_generation <= target_replan_gens:
        # Generowanie planu
        if isinstance(planner_instance, PPOPlanner):
            raw_env = planner_instance_env(planner_instance, user, start_date)
            env = planner_instance.make_wrapped(raw_env)
            ac_model = planner_instance.load_model(user)
            current_plan, _ = planner_instance._generate_plan(
                env, raw_env, ac_model, remaining_tasks, [], sim_day, sim_time,
                time_since_last_break, total_break_time_today, last_task_type
            )
        elif isinstance(planner_instance, NSGAPlanner):
            current_plan, _ = planner_instance._generate_plan(
                remaining_tasks, [], sim_day, sim_time,
                time_since_last_break, total_break_time_today, last_task_type,
                work_start_hour, work_end_hour, 20, start_date
            )
        else:
            current_plan, _ = planner_instance._generate_plan(
                remaining_tasks, sim_day, sim_time,
                work_start_hour, work_end_hour, start_date
            )

        plans_history.append(copy.deepcopy(current_plan))
        disruption_times[current_generation] = disruption_occurrence_time

        if current_generation == target_replan_gens:
            break

        replan_triggered = False
        while current_plan:
            plan_item = current_plan.pop(0)
            cal_days = sim_day + (sim_day // 5) * 2
            current_sim_dt = start_date + timedelta(days=cal_days, hours=int(sim_time),
                                                    minutes=int((sim_time % 1) * 60))

            if plan_item.get("is_break"):
                dur = plan_item["duration"]
                simulator.process_break(dur, sim_time)
                sim_time += dur
                time_since_last_break = 0.0
                total_break_time_today += dur
                last_task_type = None
            else:
                task = plan_item["task"]
                actual_dur, end_time, _ = simulator.execute_task(task, sim_time, last_task_type)
                actual_end_dt = current_sim_dt + timedelta(hours=actual_dur)
                executed_history[task.id] = {
                    "actual_start": current_sim_dt,
                    "actual_end": actual_end_dt
                }
                sim_time = end_time
                time_since_last_break += actual_dur
                last_task_type = task.type

            # Sprawdzenie disruptorów
            if disr_map and sim_day in disr_map:
                pending = disr_map[sim_day]
                appeared = []
                while pending and pending[0][0] <= sim_time:
                    d_time, d_task = pending.pop(0)
                    appeared.append((d_time, d_task))

                if appeared:
                    remaining_tasks = [it["task"] for it in current_plan if "task" in it]
                    remaining_tasks.extend(dt for _, dt in appeared)
                    remaining_tasks = sort_tasks_by_deadline_and_priority(remaining_tasks)

                    first_d_time = appeared[0][0]
                    dh = int(first_d_time)
                    dm = int((first_d_time - dh) * 60)
                    disruption_occurrence_time = start_date + timedelta(days=cal_days, hours=dh, minutes=dm)

                    current_generation += 1
                    replan_triggered = True
                    break

            if sim_time >= work_end_hour:
                sim_day += 1
                sim_time = work_start_hour
                time_since_last_break = 0.0
                total_break_time_today = 0.0
                last_task_type = None

        if not replan_triggered:
            break

    return plans_history, executed_history, disruption_times


def format_dt(dt):
    return dt.strftime("%d-%m %H:%M") if dt else "-"


def escape_tex(text):
    return str(text).replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")


def generate_case_study_tex(user, tasks_pool, disruptors_pool, output_dir="../results/case_studies"):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    if output_dir.startswith(".."):
        output_dir = os.path.join(project_root, "results", "case_studies")
    os.makedirs(output_dir, exist_ok=True)

    first_inj = min(t.injection_time for t in disruptors_pool)
    start_date = datetime(first_inj.year, first_inj.month, 3)

    planners = {
        "PPO": PPOPlanner(user),
        "NSGA-III": NSGAPlanner(user),
        "Baseline": BaselinePlanner(user)
    }

    tex_filename = os.path.join(output_dir, f"case_study_user_{user.id}_{user.chronotype}.tex")

    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write("% =========================================================================\n")
        f.write(f"% STUDIUM PRZYPADKU DLA PRACOWNIKA #{user.id} ({user.chronotype.upper()})\n")
        f.write("% =========================================================================\n\n")

        for algo_name, planner in planners.items():
            print(f"[{algo_name}] Generowanie planu i 5 przeplanowań...")
            plans_history, executed_map, disruption_times = run_full_disruption_generations(
                planner, user, copy.deepcopy(tasks_pool), copy.deepcopy(disruptors_pool), start_date,
                target_replan_gens=5
            )

            gen0 = plans_history[0] if len(plans_history) > 0 else []

            f.write(f"\\subsection{{Algorytm: {algo_name}}}\n\n")

            # -------------------------------------------------------------
            # 1. Pełny plan bazowy (Generacja 0) z przerwami i zadaniami
            # -------------------------------------------------------------
            f.write(f"\\subsubsection*{{1. Pełna Struktura Planu Początkowego (Generacja 0)}}\n")
            f.write("Poniżej przedstawiono kompletny harmonogram początkowy wygenerowany na start miesiąca, "
                    "uwzględniający zaplanowane zadania oraz interwały przerw ergonomicznych:\n\n")
            f.write("{\\footnotesize\n\\begin{enumerate}\n")
            for item in gen0:
                if item.get("is_break"):
                    f.write(
                        f"  \\item [\\textbf{{PRZERWA}}] Zaplanowano: {format_dt(item['start_time'])} -- {format_dt(item['end_time'])} "
                        f"(Czas: {item['duration']:.2f}h)\n")
                else:
                    t = item["task"]
                    dl_str = format_dt(t.deadline)
                    f.write(f"  \\item [\\textbf{{{t.id}}}] \\textbf{{{escape_tex(t.name)}}} | "
                            f"Prio: \\texttt{{{t.priority}}} | "
                            f"Deadline: {dl_str} | "
                            f"Nominalnie: {t.workhours:.2f}h | "
                            f"Planowany start: {format_dt(item['start_time'])} | "
                            f"Planowany koniec: {format_dt(item['end_time'])} | "
                            f"Zaplanowany czas: {item['duration']:.2f}h\n")
            f.write("\\end{enumerate}\n}\n\n")

            # -------------------------------------------------------------
            # 2. Zbiorcza Tabela: Plan bazowy + 5 przeplanowań
            # -------------------------------------------------------------
            f.write(f"\\subsubsection*{{2. Zbiorcza Ewolucja Planu (Plan Bazowy i 5 Przeplanowań)}}\n\n")
            f.write("W poniższej tabeli zestawiono terminy realizacji zadań w kolejnych iteracjach planowania. "
                    "Pogrubioną czcionką oznaczono zadania fizycznie zrealizowane (rzeczywisty czas trwania). "
                    "Zadania zakłócające (disruptory) umieszczono na dole tabeli (przed momentem ich wstrzyknięcia oznaczono je symbolem `$-$`):\n\n")

            # Budowa map zadań dla każdej generacji
            gen_maps = []
            for p in plans_history:
                g_map = {it["task_id"]: it for it in p if not it.get("is_break")}
                gen_maps.append(g_map)

            # Identyfikatory disruptorów
            disruptor_ids_ordered = [t.id for t in disruptors_pool]
            base_task_ids = [it["task_id"] for it in gen0 if not it.get("is_break")]
            # Uporządkowanie: zwykłe na górze, disruptory na dole
            normal_ids = [tid for tid in base_task_ids if tid not in disruptor_ids_ordered]

            num_gens = len(plans_history)
            cols_def = "|r|" + "c|" * num_gens

            f.write("\\begin{table}[htbp]\n\\centering\\scriptsize\n")
            f.write(
                f"\\caption{{Zbiorcze zestawienie przeplanowań dla algorytmu {algo_name} (Pracownik \\#{user.id})}}\n")
            f.write(f"\\label{{tab:case_study_evolution_{algo_name.lower().replace('-', '')}_u{user.id}}}\n")
            f.write(f"\\begin{{tabular}}{{{cols_def}}}\n\\hline\n")

            headers = ["\\textbf{ID Zadania}"]
            for g_idx in range(num_gens):
                if g_idx == 0:
                    headers.append("\\textbf{Gen 0 (Baza)}")
                else:
                    headers.append(f"\\textbf{{Gen {g_idx}}}")
            f.write(" & ".join(headers) + " \\\\ \\hline\n")

            def format_cell(tid, g_idx):
                # 1. Czy jest zaplanowane w tej generacji?
                if g_idx < len(gen_maps) and tid in gen_maps[g_idx]:
                    item = gen_maps[g_idx][tid]
                    return f"{format_dt(item['start_time'])}--{item['end_time'].strftime('%H:%M')}"

                # 2. Jeśli nie ma w planie, czy zostało już wykonane przed tą generacją?
                if tid in executed_map:
                    e = executed_map[tid]
                    return f"\\textbf{{{format_dt(e['actual_start'])}--{e['actual_end'].strftime('%H:%M')}}}"

                # 3. Jeśli to disruptor przed iniekcją lub zadanie odrzucone
                return "-"

            # Drukowanie zadań standardowych
            for tid in normal_ids:
                row = [str(tid)]
                for g_idx in range(num_gens):
                    row.append(format_cell(tid, g_idx))
                f.write(" & ".join(row) + " \\\\ \n")

            # Drukowanie disruptorów na dole tabeli
            f.write("\\hline\n")
            for tid in disruptor_ids_ordered:
                row = [f"\\textbf{{{tid}*}}"]
                for g_idx in range(num_gens):
                    row.append(format_cell(tid, g_idx))
                f.write(" & ".join(row) + " \\\\ \n")

            f.write("\\hline\n\\end{tabular}\n")
            f.write("\\\\ {\\tiny * -- Zadanie zakłócające (disruptor). "
                    "\\textbf{Pogrubiony tekst} = rzeczywisty czas wykonania ukończonego zadania.}\n")
            f.write("\\end{table}\n\n\\vspace{0.8cm}\n")

    print(f"Pomyślnie wygenerowano plik: {tex_filename}")


if __name__ == "__main__":

    df = load_data()
    sample_users_map = get_sample_users_per_chronotype(df)

    with SessionLocal() as session:
        tasks_pool = (
            session.query(Task)
            .filter(Task.phase == "disruptions", Task.phase_order == 0, Task.is_disruptor == False)
            .order_by(Task.deadline, Task.priority)
            .all()
        )
        disruptors_pool = (
            session.query(Task)
            .filter(Task.phase == "disruptions", Task.phase_order == 0, Task.is_disruptor == True)
            .order_by(Task.injection_time)
            .all()
        )
        session.expunge_all()

    print(f"Pobrano {len(tasks_pool)} zadań bazowych i {len(disruptors_pool)} disruptorów dla phase_order=0.")

    with SessionLocal() as session:
        for chronotype, u_id in sample_users_map.items():
            user = session.query(User).filter_by(id=u_id).first()
            print(f"Generowanie studium przypadku w LaTeX dla pracownika #{user.id} ({chronotype})...")
            generate_case_study_tex(user, tasks_pool, disruptors_pool, output_dir="../results/case_studies")