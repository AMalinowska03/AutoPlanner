import math
import copy
import io
import shelve
import time
import optuna
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Optional, List

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import Problem

from data.DbHelper import Repository, get_user_work_hours, sim_time_to_datetime
from data.DbModels import User, Task
from simulation.UserSimulator import UserSimulator, calculate_switch_lag

DisruptorsMap = dict[int, list[tuple[float, Task]]]

TIME_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
PENALTY_WEIGHT_DEADLINE = 6.0
PENALTY_WEIGHT_EFFICIENCY = 3.0
PENALTY_WEIGHT_STABILITY = 2.0
PENALTY_WEIGHT_HEALTH = 1.5

PARETO_SELECTION_WEIGHTS = np.array([
    6.0,  # deadline
    3.0,  # efficiency
    2.0,  # stability
    1.5,  # health
])

PRIO_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 3.0, "urgent": 6.0}

class PlanOptimizationProblem(Problem):
    def __init__(self, tasks, previous_plan, start_date, sim_day, sim_time,
                 time_since_break, total_break_time, last_task_type, work_start_hour, work_end_hour,
                 total_days):

        self.tasks = tasks
        self.n_tasks = len(tasks)

        # 3 parts: [tasks order, how much dime dedicated, break decision]
        super().__init__(n_var=self.n_tasks * 3, n_obj=4, n_ieq_constr=1, xl=0.0, xu=1.0)

        self.previous_plan = previous_plan
        self.start_date = start_date
        self.base_sim_day = sim_day
        self.base_sim_time = sim_time
        self.base_time_since_break = time_since_break
        self.base_total_break = total_break_time
        self.base_last_task_type = last_task_type

        self.work_start_hour = work_start_hour
        self.work_end_hour = work_end_hour
        self.total_days = total_days

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        G = np.zeros((X.shape[0], 1))

        for i in range(X.shape[0]):
            x_order = X[i, :self.n_tasks]  # first part holds tasks order
            x_time = X[i, self.n_tasks: 2 * self.n_tasks]  # second part holds how much each task should take
            x_breaks = X[i, 2 * self.n_tasks:]  # third part decides if there should be a break before task

            sequence = np.argsort(x_order)

            time_indices = np.floor(x_time * 12).astype(int)  # we have 12 different possible time multipliers
            time_indices = np.clip(time_indices, 0, 11)

            f1, f2, f3, f4, horizon = self._simulate_timeline(sequence, time_indices, x_breaks)

            F[i, 0] = f1
            F[i, 1] = f2
            F[i, 2] = f3
            F[i, 3] = f4
            G[i, 0] = horizon

        out["F"] = F
        out["G"] = G

    def _simulate_timeline(self, sequence, time_indices, break_genes):
        """
        Creates optimization conditions and returns objective components
        (f1_deadline, f2_efficiency, f3_disruptions, f4_health)
        The lower the value the better the chromosome.
        :param sequence: planned tasks in order
        :param time_indices: what time was allocated for tasks
        :param break_genes: whether break and what length should be added before task
        :return:
        """
        current_day = self.base_sim_day
        current_time = self.base_sim_time
        time_since_break = self.base_time_since_break
        total_break_time = self.base_total_break
        last_task_type = self.base_last_task_type
        max_used_day = current_day

        obj_deadline, obj_eff, obj_disrupt, obj_health = 0.0, 0.0, 0.0, 0.0

        for idx in sequence:
            task = self.tasks[idx]
            action_time_idx = time_indices[idx]
            break_gene = break_genes[idx]

            # BREAK: we add a break if break gene is at least 0.2
            if break_gene >= 0.2:
                break_duration = (5.0 / 60.0) + (break_gene - 0.2) / 0.8 * 0.75

                # if positive will be < 0
                obj_eff += self._calculate_break_reward(current_time, time_since_break, break_duration)

                current_time += break_duration
                total_break_time += break_duration
                time_since_break = 0.0
                last_task_type = None
                max_used_day = max(max_used_day, current_day)

                # end of day
                if current_time >= self.work_end_hour:
                    obj_health += self._calculate_overtime_reward(current_time)
                    obj_health += self._calculate_end_day_break_reward(total_break_time, current_time)

                    current_day += 1
                    current_time = self.work_start_hour
                    time_since_break = 0.0
                    total_break_time = 0.0
                    last_task_type = None

            # TASK
            time_multiplier = TIME_MULTIPLIERS[action_time_idx]
            planned_duration = float(task.workhours) * time_multiplier

            lag_cost, lag_dur = calculate_switch_lag(last_task_type, task.type)
            actual_duration = planned_duration * (1.0 + lag_cost * 0.5)
            end_time = current_time + actual_duration

            current_abs_start = current_day * 24.0 + current_time

            obj_deadline += self._calculate_deadline_reward(current_day, task, end_time)
            obj_eff += self._calculate_time_allotment_reward(task, planned_duration, actual_duration)
            obj_disrupt += self._calculate_disruption_reward(task, current_abs_start)

            current_time = end_time
            time_since_break += actual_duration
            last_task_type = task.type
            max_used_day = max(max_used_day, current_day)

            # end of day
            if current_time >= self.work_end_hour:
                obj_health += self._calculate_overtime_reward(current_time)
                obj_health += self._calculate_end_day_break_reward(total_break_time, current_time)

                current_day += 1
                current_time = self.work_start_hour
                time_since_break = 0.0
                total_break_time = 0.0
                last_task_type = None
        horizon_violation = max_used_day - (self.total_days - 1)
        return obj_deadline, obj_eff, obj_disrupt, obj_health, horizon_violation

    def _calculate_break_reward(self, current_time, time_since_break,  break_duration: float):
        break_reward = 0.0

        # penalty (beginning/end of day break, too many/few breaks)
        if current_time <= self.work_start_hour + 0.1:
            break_reward += break_duration * PENALTY_WEIGHT_EFFICIENCY * 2.0
        if current_time + break_duration >= self.work_end_hour:
            break_reward += break_duration * PENALTY_WEIGHT_EFFICIENCY * 2.0

        # break distribution
        if time_since_break >= 2.0:
            # greater the reward, the longer last break was
            break_reward -= (time_since_break * PENALTY_WEIGHT_HEALTH)
            # preferable short breaks except for lunch
            if break_duration > 0.35 and not (11.5 < current_time < 14.5):
                break_reward += (break_duration - 0.35) * PENALTY_WEIGHT_EFFICIENCY * 2.0
        elif time_since_break < 1.0:
            # penalty for stacking breaks
            break_reward += 2.0

        # reward (break for eating midday)
        if 11.5 < current_time < 14.5 and 0.25 <= break_duration <= 0.75:
            break_reward -= break_duration * PENALTY_WEIGHT_HEALTH

        return break_reward

    def _calculate_deadline_reward(self, current_day: int, task: Task, end_time: float):
        deadline_reward = 0.0
        end_date = sim_time_to_datetime(self.start_date, current_day, end_time)
        tardiness = (end_date - task.deadline).total_seconds() / 3600.0

        w_prio = PRIO_WEIGHTS.get(task.priority, 1.0)

        if tardiness > 0:
            tardiness_days = (tardiness / 24.0)  # if we are late 7 days it's as bad as beyond that
            # missing deadline is more crucial to correct than rewarding for doing task on time
            deadline_reward += (1.0 + tardiness_days) * PENALTY_WEIGHT_DEADLINE * w_prio
        else:
            deadline_reward -= PENALTY_WEIGHT_DEADLINE * w_prio
        return deadline_reward

    def _calculate_time_allotment_reward(self, task: Task, action_time, actual_duration: float):
        time_reward = 0.0
        time_multiplier = TIME_MULTIPLIERS[action_time % len(TIME_MULTIPLIERS)]
        planned_duration = time_multiplier * float(task.workhours)
        # penalty (task execution time exceeded/finished early - disruption, overtime)
        # we cap it at -8h +8h - standard work day length, already badly allocated, prevents reward from exploding
        planning_time_difference = max(-8.0, min(8.0, actual_duration - planned_duration))
        # 15min grace period
        if -0.25 < planning_time_difference < 0.25:
            time_reward -= 1.0 / (1.0 + abs(planning_time_difference) * PENALTY_WEIGHT_EFFICIENCY)
        elif planning_time_difference > 0.25:
            # the more time was actually needed to complete the task the more penalty
            time_reward += abs(planning_time_difference) * (2 * PENALTY_WEIGHT_EFFICIENCY)
        elif planning_time_difference < -0.25:  # finished before time
            # we could save plan time here but giving a bit more time is always better than not giving enough
            time_reward += abs(planning_time_difference) * PENALTY_WEIGHT_EFFICIENCY

        return time_reward

    def _calculate_disruption_reward(self, task: Task, current_abs_start: float):
        disruption_reward = 0.0
        prev_record = next((item for item in self.previous_plan if item.get("task_id") == task.id), None)
        if prev_record:
            prev_start = prev_record.get("_abs_start", current_abs_start)
            shift = abs(current_abs_start - prev_start)

            if shift > 0.1:  # we don't include small shifts of a few minutes
                time_to_event = max(0.5, prev_start - current_abs_start)
                # if bigger the shift the less penalty grows - it's a big change anyway (logarithmic)
                base_penalty = math.log(1.0 + shift) * PENALTY_WEIGHT_STABILITY

                # the further the event was originally planned the less impact it has
                proximity_multiplier = 1.0 / time_to_event

                disruption_reward += base_penalty * proximity_multiplier
            else:
                # small reward for keeping the task unmoved
                disruption_reward -= PENALTY_WEIGHT_STABILITY

        return disruption_reward

    def _calculate_overtime_reward(self, end_time: float):
        # overtime penalty - the longer task takes to end the bigger penalty
        overtime_reward = 0.0
        overtime = end_time - self.work_end_hour
        if overtime > 2.0:
            overtime_reward += overtime * PENALTY_WEIGHT_HEALTH * 3
        elif overtime > 0.0:
            overtime_reward += overtime * PENALTY_WEIGHT_HEALTH
        return overtime_reward

    def _calculate_end_day_break_reward(self, total_break_time, current_time):
        break_reward = 0.0
        worked_today = max(0.1, current_time - self.work_start_hour)
        break_pct = total_break_time / worked_today
        if 0.10 <= break_pct <= 0.15:
            break_reward -= 2 * PENALTY_WEIGHT_HEALTH
        elif break_pct < 0.10:
            break_reward += PENALTY_WEIGHT_HEALTH*(0.9 - break_pct)
        else:
            break_reward += 5 * PENALTY_WEIGHT_HEALTH * (break_pct - 0.15)
        return break_reward


class NSGAPlanner:
    def __init__(self, user: Optional[User] = None):
        self.user = user
        self.storage_path = "db/nsga_params.db"
        self.repository = None
        self.nsga_params = {"n_partitions": 4, "n_gen": 50, "prob_cross": 0.9, "eta_mut": 20}

    def pretrain(self, all_users: List[User], pretrain_tasks: dict, n_trials: int = 30):
        """
        Searches for globally optimal hiperparameters based on randomly chosen task groups and users.
        :param all_users:
        :param pretrain_tasks:
        :param n_trials:
        :return:
        """
        print("NSGA ------ Start pretraining (Optuna)...")
        scenarios = sample_scenarios(all_users, pretrain_tasks, max(1, int(len(pretrain_tasks)/2)))
        training_date = datetime(year=2027, month=2, day=1)

        def objective(trial):
            # Optuna chooses evolution params
            n_partitions = trial.suggest_int("n_partitions", 2, 6)  # pop_size (10 - 84)
            n_gen = trial.suggest_int("n_gen", 20, 100)
            prob_cross = trial.suggest_float("prob_cross", 0.5, 1.0)
            eta_mut = trial.suggest_int("eta_mut", 10, 30)

            scenario_scores = []

            params = {"n_partitions": n_partitions, "n_gen": n_gen, "prob_cross": prob_cross, "eta_mut": eta_mut}
            for user, tasks in scenarios:
                plan = self._optimization(user, tasks, params, training_date)
                if plan is None:
                    scenario_scores.append(float("inf"))
                    continue
                score = self._evaluate_plan_with_simulator(user, plan, training_date)
                scenario_scores.append(score)

            return np.mean(scenario_scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print(f"NSGA ------ Best global params: {study.best_params}")
        self._save_to_storage("base_nsga_params", study.best_params)

    def finetune(self, user: User, finetune_tasks: dict, n_trials: int = 15):
        """
        Dostraja hiperparametry dla konkretnego użytkownika, startując wokół bazy z Pretrain.
        """
        print(f"NSGA ------ Finetuning (Optuna) for user {user.id}...")

        base_params = {"n_partitions": 4, "n_gen": 50, "prob_cross": 0.9, "eta_mut": 20}
        with shelve.open(self.storage_path) as db:
            if "base_nsga_params" in db:
                base_params = db["base_nsga_params"]

        scenarios = sample_scenarios([user], finetune_tasks, len(finetune_tasks))
        training_date = datetime(year=2027, month=2, day=1)

        def objective(trial):
            # limit params to tuning globally found values (Fine-tuning)
            n_partitions = trial.suggest_int("n_partitions", max(2, base_params["n_partitions"] - 1),
                                             base_params["n_partitions"] + 1)
            n_gen = trial.suggest_int("n_gen", max(20, base_params["n_gen"] - 20), base_params["n_gen"] + 20)
            prob_cross = trial.suggest_float("prob_cross", max(0.5, base_params["prob_cross"] - 0.2),
                                             min(1.0, base_params["prob_cross"] + 0.1))

            scenario_scores = []

            params = {"n_partitions": n_partitions, "n_gen": n_gen, "prob_cross": prob_cross,
                      "eta_mut": base_params["eta_mut"]}
            for scenario_user, tasks in scenarios:
                plan = self._optimization(scenario_user, tasks, params, training_date)
                if plan is None:
                    scenario_scores.append(float("inf"))
                    continue
                score = self._evaluate_plan_with_simulator(scenario_user, plan, training_date)
                scenario_scores.append(score)

            return np.mean(scenario_scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = {**study.best_params, "eta_mut": base_params["eta_mut"]}
        self._save_to_storage(f"user_{user.id}_nsga_params", best_params)

    def _select_from_pareto(self, res):
        if res.F is None or res.X is None:
            return None

        F = np.atleast_2d(np.asarray(res.F, dtype=float))

        X = np.atleast_2d(res.X)

        if F.shape[0] == 1:
            return X[0]

        f_min = F.min(axis=0)
        f_max = F.max(axis=0)

        ranges = f_max - f_min
        ranges[ranges == 0.0] = 1.0

        normalized_f = (F - f_min) / ranges

        scores = (normalized_f @ PARETO_SELECTION_WEIGHTS)

        return X[int(np.argmin(scores))]

    def _optimization(self, sample_user: User, sample_tasks: list[Task],
                      params: dict, start_date: datetime = datetime(2027, 2, 1)):
        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=params["n_partitions"])
        algorithm = NSGA3(
            pop_size=len(ref_dirs),
            ref_dirs=ref_dirs,
            crossover=SBX(prob=params["prob_cross"], eta=15),
            mutation=PM(eta=params["eta_mut"])
        )

        work_start_hour, work_end_hour = get_user_work_hours(sample_user)
        problem = PlanOptimizationProblem(
            tasks=sample_tasks, previous_plan=[], start_date=start_date,
            sim_day=0, sim_time=work_start_hour, time_since_break=0.0, total_break_time=0.0,
            last_task_type=None, work_start_hour=work_start_hour, work_end_hour=work_end_hour, total_days=20
        )

        res = minimize(problem, algorithm, termination=DefaultMultiObjectiveTermination(n_max_gen=params["n_gen"]),
                       seed=1, verbose=False)

        best_X = self._select_from_pareto(res)

        if best_X is None:
            return None

        return self._decode_x_to_plan(best_X, sample_tasks, start_date, 0,
                                      work_start_hour, work_start_hour, work_end_hour)\


    def _evaluate_plan_with_simulator(
            self,
            user: User,
            plan: list,
            start_date: datetime,
            total_days: int = 20
    ) -> float:
        """
        Evaluates a plan using the simulated real user.
        Used only during NSGA pretrain/finetune Optuna calibration.
        Lower score is better.
        """
        simulator = UserSimulator(user)

        work_start_hour, work_end_hour = (get_user_work_hours(user))

        sim_day = 0
        sim_time = work_start_hour
        last_task_type = None

        completed = 0
        deadline_delay_sum = 0.0
        estimation_error_sum = 0.0
        overtime_sum = 0.0

        task_count = sum(1 for item in plan if not item.get("is_break"))

        for index, item in enumerate(plan):
            if sim_day >= total_days:
                break

            if item.get("is_break"):
                duration = float(item["duration"])

                simulator.process_break(duration, sim_time)

                sim_time += duration
                last_task_type = None

            else:
                task = item["task"]

                planned_duration = float(item["duration"])

                actual_duration, end_time, _ = simulator.execute_task(task=task, start_time=sim_time,
                                                                      prev_task_type=last_task_type)

                actual_end_date = sim_time_to_datetime(start_date, sim_day, end_time)

                if task.deadline is not None:
                    delay = (actual_end_date - task.deadline).total_seconds() / 3600.0

                    if delay > 0:
                        priority_weight = PRIO_WEIGHTS.get(task.priority, 1.0)
                        deadline_delay_sum += delay * priority_weight

                estimation_error_sum += abs(actual_duration - planned_duration) / max(actual_duration, 1e-6)
                completed += 1

                sim_time = end_time
                last_task_type = task.type

            if sim_time >= work_end_hour:
                overtime_sum += max(0.0, sim_time - work_end_hour) ** 2

                simulator.reset(sim_time, work_end_hour, weekly=(sim_day + 1) % 5 == 0)
                sim_day += 1
                sim_time = work_start_hour
                last_task_type = None

        if task_count == 0:
            return 0.0

        completion_loss = (task_count - completed) / task_count

        completed_denominator = max(1, completed)

        mean_deadline_delay = deadline_delay_sum / completed_denominator
        mean_estimation_error = estimation_error_sum / completed_denominator

        used_days = max(1, min(sim_day + 1, total_days))
        mean_overtime = overtime_sum / used_days

        return float(
            50.0 * completion_loss
            + 6.0 * mean_deadline_delay
            + 3.0 * mean_estimation_error
            + 1.5 * mean_overtime
        )

    def plan_and_simulate_month(self, user: User, month_tasks: List[Task], group_id: int,
                                disruptors_map: Optional[DisruptorsMap] = None,
                                phase='online', phase_order=0, start_date=datetime(2027, 1, 4)):
        disr_map = copy.deepcopy(disruptors_map)
        self.repository = Repository(user, phase, phase_order, start_date)
        work_start_hour, work_end_hour = get_user_work_hours(user)

        # load user params nor NSGA
        self.nsga_params = {"n_partitions": 4, "n_gen": 50, "prob_cross": 0.9, "eta_mut": 20}
        with shelve.open(self.storage_path) as db:
            if f"user_{user.id}_nsga_params" in db:
                self.nsga_params = db[f"user_{user.id}_nsga_params"]

        simulator = UserSimulator(user)
        current_generation = 0
        remaining_to_plan = copy.deepcopy(month_tasks)
        previous_plan_state = []

        sim_day = 0
        sim_time = work_start_hour

        disruption_occurrence_time = None
        time_since_last_break = 0.0
        total_break_time_today = 0.0
        last_task_type = None
        while remaining_to_plan and sim_day < 20:
            print(f"NSGA ------ Planning: user {user.id} | generation: {current_generation} | start date: {start_date}")
            current_plan, gen_time = self._generate_plan(remaining_to_plan, previous_plan_state, sim_day, sim_time,
                                                         time_since_last_break, total_break_time_today, last_task_type,
                                                         work_start_hour, work_end_hour, 20, start_date)

            # save plan to db
            plan_record = self.repository.create_plan_records(
                algorithm="nsga", planned_tasks=current_plan, group_id=group_id,
                generation=current_generation, disruption_time=disruption_occurrence_time, generating_time=gen_time
            )

            print(f"NSGA ------ Simulating ------")
            replan_needed = False

            # go through all planned tasks until they are possible to be completed
            while current_plan:
                plan_item = current_plan.pop(0)
                calendar_days_passed = sim_day + (sim_day // 5) * 2

                # execute plan item
                if plan_item.get("is_break"):
                    simulator.process_break(duration=plan_item["duration"], time=sim_time)
                    self.repository.save_break(plan_item["duration"], plan_record, sim_time, plan_item["start_time"])
                    sim_time += plan_item["duration"]
                    time_since_last_break = 0.0
                    total_break_time_today += plan_item["duration"]
                    last_task_type = None
                else:
                    task = plan_item["task"]
                    actual_dur, end_time, energy = simulator.execute_task(task, sim_time, last_task_type)
                    current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                            minutes=int((sim_time % 1) * 60))
                    self.repository.save_execution_to_db(
                        plan_record, task, current_sim_dt,
                        current_sim_dt + timedelta(hours=actual_dur), energy
                    )
                    last_task_type = task.type
                    sim_time = end_time
                    time_since_last_break += actual_dur

                #  check if disruptor is supposed to appear
                if disr_map and sim_day in disr_map:
                    pending_disruptors = disr_map[sim_day]
                    disruptors_appeared = []
                    while pending_disruptors and pending_disruptors[0][0] <= sim_time:
                        disrupt_time, disruptor_task = pending_disruptors.pop(0)
                        disruptors_appeared.append((disrupt_time, disruptor_task))

                    if disruptors_appeared:
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        remaining_to_plan.extend(disruptor_task for _, disruptor_task in disruptors_appeared)
                        first_disruption_time = min(disrupt_time for disrupt_time, _ in disruptors_appeared)
                        dh = int(first_disruption_time)
                        dm = int((first_disruption_time - dh) * 60)
                        disruption_occurrence_time = start_date + timedelta(days=calendar_days_passed, hours=dh,
                                                                            minutes=dm)
                        previous_plan_state = copy.deepcopy(current_plan)
                        replan_needed = True
                        print(f"NSGA ------ Disruptor occurred: RE-PLANNING ------")
                        break
                disruption_occurrence_time = None

                current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                        minutes=int((sim_time % 1) * 60))

                # if we finish tasks for the day earlier we might want to re-plan to not waste work day
                if current_plan:
                    next_item = current_plan[0]
                    next_start_dt = next_item["start_time"]

                    # re-plan if next task is in next day, and we still have over 0.5h of work day
                    if next_start_dt.date() > current_sim_dt.date() and work_end_hour - sim_time >= 0.5:
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        previous_plan_state = copy.deepcopy(current_plan)
                        replan_needed = True
                        print(f"NSGA ------ Have time left: RE-PLANNING ------")
                        break

                # end of day
                if sim_time >= work_end_hour:
                    sim_day += 1
                    if sim_day >= 20:
                        if current_plan:
                            print(f"NSGA ------ End of month with tasks left ------")
                        remaining_to_plan = []
                        break
                    simulator.reset(sim_time, work_end_hour, weekly=(sim_day % 5 == 0))
                    sim_time = work_start_hour
                    time_since_last_break = 0.0
                    total_break_time_today = 0.0
                    last_task_type = None

                    if current_plan:  # still have tasks in plan
                        next_item = current_plan[0]
                        next_start_dt = next_item["start_time"]
                        if next_start_dt.date() <= current_sim_dt.date():
                            remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                            previous_plan_state = copy.deepcopy(current_plan)
                            replan_needed = True
                            print(f"NSGA ------ Tasks left from day: RE-PLANNING ------")
                            break

            # if we moved through tasks without re-planning we finish month
            if not replan_needed:
                remaining_to_plan = []
                print(f"NSGA ------ Simulation END ------")
            else:
                current_generation += 1

        return {
            "total_replans": current_generation,
            "days_used": sim_day,
        }

    def _generate_plan(self, remaining_tasks, previous_plan, sim_day, sim_time, time_since_last_break, total_break_time,
                       last_task_type, work_start_hour, work_end_hour, total_days, start_date):
        t0 = time.time()

        # Odtworzenie parametrów z Optuny
        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=self.nsga_params["n_partitions"])
        algorithm = NSGA3(
            pop_size=len(ref_dirs),
            ref_dirs=ref_dirs,
            crossover=SBX(prob=self.nsga_params["prob_cross"], eta=15),
            mutation=PM(eta=self.nsga_params["eta_mut"])
        )

        problem = PlanOptimizationProblem(
            tasks=remaining_tasks, previous_plan=previous_plan, start_date=start_date,
            sim_day=sim_day, sim_time=sim_time, time_since_break=time_since_last_break,
            total_break_time=total_break_time, last_task_type=last_task_type,
            work_start_hour=work_start_hour, work_end_hour=work_end_hour, total_days=total_days
        )

        termination = DefaultMultiObjectiveTermination(n_max_gen=self.nsga_params["n_gen"])
        res = minimize(problem, algorithm, termination, seed=1, verbose=False)

        # choose best option from pareto front based on weights
        best_X = self._select_from_pareto(res)
        if best_X is None:
            raise RuntimeError("NSGA did not produce a feasible solution.")

        current_plan = self._decode_x_to_plan(best_X, remaining_tasks, start_date, sim_day, sim_time,
                                              work_start_hour, work_end_hour)

        generation_time = round(time.time() - t0, 4)
        return current_plan, generation_time

    def _save_to_storage(self, key: str, params: dict):
        with shelve.open(self.storage_path) as db:
            db[key] = params

    def _decode_x_to_plan(self, X, tasks, start_date, sim_day, sim_time, work_start_hour, work_end_hour):
        n_tasks = len(tasks)
        x_order = X[:n_tasks]
        x_time = X[n_tasks: 2 * n_tasks]
        x_breaks = X[2 * n_tasks:]

        sequence = np.argsort(x_order)
        time_indices = np.clip(np.floor(x_time * 12).astype(int), 0, 11)

        current_plan = []
        current_day, current_time = sim_day, sim_time
        last_task_type = None

        for idx in sequence:
            task = tasks[idx]

            if x_breaks[idx] >= 0.2:
                break_dur = (5.0 / 60.0) + (x_breaks[idx] - 0.2) / 0.8 * 0.75
                cal_days = current_day + (current_day // 5) * 2
                start_dt = start_date + timedelta(days=cal_days, hours=int(current_time),
                                                  minutes=int((current_time % 1) * 60))
                end_dt = start_dt + timedelta(hours=break_dur)

                current_plan.append({
                    "is_break": True, "duration": break_dur,
                    "start_time": start_dt, "end_time": end_dt
                })
                current_time += break_dur
                last_task_type = None
                if current_time >= work_end_hour:  # work_end_hour
                    current_day += 1
                    current_time = work_start_hour  # work_start_hour

            t_mult = TIME_MULTIPLIERS[time_indices[idx]]
            lag_cost, _ = calculate_switch_lag(last_task_type, task.type)
            scheduled_duration = float(task.workhours) * t_mult * (1.0 + lag_cost * 0.5)

            cal_days = current_day + (current_day // 5) * 2
            start_dt = start_date + timedelta(days=cal_days, hours=int(current_time),
                                              minutes=int((current_time % 1) * 60))
            end_dt = start_dt + timedelta(hours=scheduled_duration)

            current_plan.append({
                "task_id": task.id, "task": task, "duration": scheduled_duration,
                "start_time": start_dt, "end_time": end_dt,
                "_abs_start": current_day * 24.0 + current_time
            })

            current_time += scheduled_duration
            last_task_type = task.type
            if current_time >= work_end_hour:
                current_day += 1
                current_time = work_start_hour
                last_task_type = None

        return current_plan


def sample_scenarios(
        users: list[User],
        divided_tasks: dict[int, list[Task]],
        n_scenarios: int = 8,
        seed: int = 42,
):
    rng = random.Random(seed)

    pairs = [
        (user, group_id)
        for user in users
        for group_id in divided_tasks.keys()
    ]

    selected = rng.sample(
        pairs,
        k=min(n_scenarios, len(pairs)),
    )

    return [
        (user, divided_tasks[group_id])
        for user, group_id in selected
    ]
