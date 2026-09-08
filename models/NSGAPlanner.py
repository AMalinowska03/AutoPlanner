import time
import math
import numpy as np
from datetime import timedelta, datetime
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


class TaskSchedulingProblem(ElementwiseProblem):
    def __init__(self, tasks, user_profile, current_time_start):
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.user = user_profile
        self.current_time_start = current_time_start

        # Zmienne: kolejność (0-1), czas przerw (0-1.5h), mnożnik czasu (indeks 0-3)
        super().__init__(
            n_var=self.n_tasks * 3,
            n_obj=3,  # f1: Deadlines, f2: Przerwy, f3: Overtime/Stabilność
            n_ieq_constr=0,
            xl=np.array([0.0] * self.n_tasks + [0.0] * self.n_tasks + [0] * self.n_tasks),
            xu=np.array([1.0] * self.n_tasks + [1.5] * self.n_tasks + [3] * self.n_tasks)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        sequence = np.argsort(x[0:self.n_tasks])
        break_genes = x[self.n_tasks:self.n_tasks * 2]
        time_genes = x[self.n_tasks * 2:self.n_tasks * 3]

        f1_deadlines, f2_breaks, f3_overtime = 0.0, 0.0, 0.0
        current_time = self.current_time_start

        for idx in sequence:
            task = self.tasks[idx]
            break_duration = break_genes[idx]

            # Przetwarzanie przerwy
            if break_duration > 0.15:  # Próg włączenia przerwy
                # TUTAJ: f2_breaks += obliczona kara za przerwę
                current_time += timedelta(hours=break_duration)

            # Czas trwania zadania na podstawie mnożnika
            planned_duration = (int(np.round(time_genes[idx])) + 1) * float(task.workhours)

            # TUTAJ: Logika przenoszenia zadania na kolejny dzień jeśli przekracza work_end_time[cite: 2]

            task_end_time = current_time + timedelta(hours=planned_duration)

            # TUTAJ: f1_deadlines += kara za deadline
            # TUTAJ: f3_overtime += kara za overtime / stability

            current_time = task_end_time

        out["F"] = [f1_deadlines, f2_breaks, f3_overtime]


# ==========================================
# CZĘŚĆ 2 & 3: KLASA NSGAPLANNER (LOGIKA)
# ==========================================
class NSGAPlanner:
    def __init__(self, user, repository):
        self.user = user
        self.repository = repository
        # Domyślne parametry ustalone empirycznie, podmieniane po pretrain/finetune
        self.nsga_params = {
            'pop_size': 100,
            'n_gen': 200,
            'ref_dirs': get_reference_directions("das-dennis", 3, n_partitions=12)
        }

    # --- KALIBRACJA ---
    def evaluate_pareto_front(self, res, weights=[0.6, 0.3, 0.1]):
        """Uśrednia wyniki frontu z naciskiem na deadliny (waga 0.6)."""
        if res.F is None: return float('inf')
        return np.mean(np.dot(res.F, weights))

    def pretrain(self, all_users, tasks_by_month_dict):
        """Uruchamia poszukiwanie optymalnych parametrów ogólnych dla wielu userów[cite: 2]."""
        best_score = float('inf')
        for pop in [50, 100]:
            for gen in [100, 200]:
                total_score = 0.0
                eval_count = 0
                for u in all_users:
                    for phase_order, tasks in tasks_by_month_dict.items():
                        alg = NSGA3(ref_dirs=self.nsga_params['ref_dirs'], pop_size=pop)
                        prob = TaskSchedulingProblem(tasks, u, u.work_start_time)
                        res = minimize(prob, alg, ('n_gen', gen), verbose=False)
                        total_score += self.evaluate_pareto_front(res)
                        eval_count += 1

                avg_score = total_score / max(1, eval_count)
                if avg_score < best_score:
                    best_score = avg_score
                    self.nsga_params.update({'pop_size': pop, 'n_gen': gen})
        print(f"Pretrain zakończony. Najlepsze parametry: {self.nsga_params}")

    def finetune(self, user_tasks):
        """Dostraja parametry tylko dla konkretnego użytkownika `self.user`."""
        # Podobna logika do pretrain, ale zawężona siatka poszukiwań i tylko self.user
        pass

        # --- GENEROWANIE PLANU ---

    def generate_plan(self, pending_tasks, current_sim_time):
        """Generuje plan, mierzy czas i wyciąga najlepsze rozwiązanie faworyzujące deadliny."""
        start_gen_time = time.time()

        problem = TaskSchedulingProblem(pending_tasks, self.user, current_sim_time)
        algorithm = NSGA3(
            ref_dirs=self.nsga_params['ref_dirs'],
            pop_size=self.nsga_params['pop_size']
        )

        # Algorytm nie korzysta z symulatora podczas generowania
        res = minimize(problem, algorithm, ('n_gen', self.nsga_params['n_gen']), verbose=False)
        generating_time = time.time() - start_gen_time

        # Wybieramy rozwiązanie z Frontu Pareto z najmniejszą karą dla f1 (Deadliny)
        best_idx = np.argmin(res.F[:, 0])
        best_genome = res.X[best_idx]

        # Ostatnie zdekodowanie najlepszego genomu na konkretne obiekty i czasy (jak w _evaluate)
        structured_plan = self._decode_genome_to_schedule(best_genome, pending_tasks, current_sim_time)

        return structured_plan, generating_time

    def _decode_genome_to_schedule(self, genome, tasks, start_time):
        """Zamienia genom w listę słowników `{'task_id': int, 'start_time': dt, 'end_time': dt, 'is_break': bool, ...}`"""
        # (Implementacja odtwarzająca pętlę z _evaluate i budująca listę)
        return []  # Zwraca gotową uporządkowaną listę

    # --- SYMULACJA I ZAPIS ---
    def execute_and_simulate(self, monthly_tasks, group_id, phase="online"):
        """Główna pętla wykonawcza z obsługą symulatora, przerw i zakłóceń."""
        sim = UserSimulator(self.user)
        current_sim_time = self.user.work_start_time
        generation_count = 0

        # Wyodrębnienie zakłóceń
        regular_tasks = [t for t in monthly_tasks if not t.is_disruptor]
        disruptors = [t for t in monthly_tasks if t.is_disruptor]

        pending_tasks = regular_tasks.copy()

        while pending_tasks:
            # 1. Generowanie planu (bez douczania[cite: 1, 2])
            planned_schedule, gen_time = self.generate_plan(pending_tasks, current_sim_time)

            # 2. Zapis głównego rekordu Planu i PlanTask
            plan_db_record = self.repository.create_plan_records(
                algorithm="nsga",
                planned_tasks=[t for t in planned_schedule if not t.get('is_break')],
                group_id=group_id,
                generation=generation_count,
                disruption_time=None
            )
            # Aktualizacja generating_time
            plan_db_record.generating_time = gen_time
            # Zapisanie do bazy (np. self.repository.session.commit())

            # 3. Pętla wykonawcza dla wygenerowanego planu
            for item in planned_schedule:
                # Obsługa zdarzeń losowych (Disruptions) w trakcie dnia
                if phase == "disruptions":
                    for d_task in disruptors:
                        if current_sim_time >= d_task.disruption_time and d_task not in pending_tasks:
                            print(f"Dodano zakłócenie: {d_task.name}")
                            pending_tasks.append(d_task)
                            generation_count += 1
                            # Zakończenie aktualnego zadania i przeplanowanie na nowo
                            break
                    else:
                        continue
                    break  # Wyjście z pętli wykonawczej, nastąpi replanowanie

                # Wykonanie przerwy
                if item.get('is_break'):
                    duration_hrs = (item['end_time'] - item['start_time']).total_seconds() / 3600
                    # Zapisuje nowe zadanie typu przerwa i odpowiedni PlanTask[cite: 2, 3]
                    self.repository.save_break(
                        break_duration=duration_hrs,
                        plan_record=plan_db_record,
                        start_hour_float=current_sim_time.hour + current_sim_time.minute / 60.0,
                        current_day=current_sim_time
                    )
                    # Odtwarzanie energii po przerwie[cite: 1]
                    sim.process_break(duration_hrs, current_sim_time.hour)
                    current_sim_time = item['end_time']
                    continue

                # Faktyczne wykonanie zwykłego zadania w symulatorze
                task_obj = next(t for t in pending_tasks if t.id == item['task_id'])
                sim_start_float = current_sim_time.hour + current_sim_time.minute / 60.0

                # Zwraca faktyczny czas trwania i zużytą energię[cite: 1]
                actual_dur, end_float_time, energy_used = sim.execute_task(task_obj, sim_start_float)

                sim_end_time = current_sim_time + timedelta(hours=actual_dur)

                # Zapis faktycznego wykonania (Execution)[cite: 3]
                self.repository.save_execution_to_db(
                    plan_record=plan_db_record,
                    task=task_obj,
                    start_date=current_sim_time,
                    end_date=sim_end_time,
                    energy_used=energy_used
                )

                current_sim_time = sim_end_time
                pending_tasks.remove(task_obj)

                # Jeśli koniec dnia pracy jest za mniej niż 5 minut, przewijamy czas do jutra i replanujemy[cite: 2]
                time_to_end = (self.user.work_end_time - current_sim_time).total_seconds() / 60
                if 0 <= time_to_end < 5:
                    current_sim_time = current_sim_time + timedelta(days=1)
                    current_sim_time = current_sim_time.replace(
                        hour=self.user.work_start_time.hour,
                        minute=self.user.work_start_time.minute
                    )
                    generation_count += 1
                    break  # Przerwij bieżący plan, wygeneruj nowy na resztę zadań