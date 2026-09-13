from datetime import datetime
from typing import List, Dict, Any
import math

from simulation.UserSimulator import CHRONOTYPES, SKILL_ATTR_MAP, SWITCH_MATRIX, calculate_switch_lag
from collections import defaultdict
from data.DbModels import User

PRIORITY_WEIGHTS = {"low": 0.25, "medium": 0.5, "high": 0.75, "urgent": 1.0}


def get_user_work_hours(user):
    work_start_hour = (
        user.work_start_time.hour + user.work_start_time.minute / 60.0
        if isinstance(user.work_start_time, datetime) else 8.0
    )
    work_end_hour = (
        user.work_end_time.hour + user.work_end_time.minute / 60.0
        if isinstance(user.work_end_time, datetime) else 16.0
    )
    return work_start_hour, work_end_hour


def get_task_difficulty(task, user: User) -> float:
    param = SKILL_ATTR_MAP.get(task.type)
    skill = getattr(user, param["attr_name"], 0.5) if param else 0.5
    return float(task.workhours) * (1.5 - skill)


def get_expected_attention(chronotype: str, hour_decimal: float) -> float:
    peak = CHRONOTYPES[chronotype]["peak_attention_factor"]
    attention = 0.15 * math.cos(2 * math.pi * (hour_decimal - peak) / 24)
    return (attention + 0.15) / 0.3


def daily_task_completion_score(plans_history: List[dict], executed_tasks: List[Dict[str, Any]]) -> float:
    """
    Checks plan realisation day by day.
    For each day we choose plan day was started with.
    Added tasks when time left are counted onl as bonus not lowering the score when they are not finished by EoD.
    """
    if not plans_history:
        return 1.0

    # tasks completed (not breaks)
    executed_by_day = defaultdict(set)
    for exc in executed_tasks:
        task = exc["task"]
        if getattr(task, "is_break", False):
            continue
        day_date = exc["actual_start"].date()
        executed_by_day[day_date].add(task.id)

    # get unique dates plans were created for
    all_dates = set()
    for plan in plans_history:
        for t_info in plan["tasks"].values():
            if not t_info.get("is_break"):
                all_dates.add(t_info["start_time"].date())

    if not all_dates:
        return 1.0

    daily_scores = []

    # sort plans by generation
    sorted_plans = sorted(plans_history, key=lambda p: p["generation"])

    for current_date in sorted(all_dates):
        # A. Wyznaczamy plan poranny (baza dnia):
        # morning plan of the day - last known generation created before or at the start of that day
        morning_plan_tasks = set()
        for plan in sorted_plans:
            tasks_for_today = {
                t_id for t_id, t_info in plan["tasks"].items()
                if t_info["start_time"].date() == current_date and not t_info.get("is_break")
            }
            if tasks_for_today:
                morning_plan_tasks = tasks_for_today
                # stop at first generation that created day's plan
                break

        if not morning_plan_tasks:
            continue

        completed_today = executed_by_day.get(current_date, set())

        # how many were executed from morning plan on that day
        base_completed = morning_plan_tasks.intersection(completed_today)
        # how many extras were added
        extra_completed = completed_today.difference(morning_plan_tasks)

        # completed tasks for the day compared to planned
        total_success = len(base_completed) + len(extra_completed)
        day_score = min(1.0, total_success / max(len(morning_plan_tasks), 1))

        daily_scores.append(day_score)

    if not daily_scores:
        return 1.0

    return round(float(sum(daily_scores) / len(daily_scores)), 4)


def task_time_estimation_score(executed_tasks: List[Dict[str, Any]]) -> float:
    score = 0.0
    weight_sum = 0.0
    for execution in executed_tasks:
        task = execution["task"]
        if getattr(task, "is_break", False):
            continue
        expected_time = execution["planned_duration_sec"]
        execution_time = execution["actual_duration_sec"]
        difference = abs(execution_time - expected_time)
        weight = PRIORITY_WEIGHTS.get(task.priority, 0.5)
        weight_sum += weight
        score += (difference/max(expected_time, 1.0))*weight
    return round(score/weight_sum, 4) if weight_sum > 0.0 else 0.0


def task_execution_delay_score(executed_tasks: List[Dict[str, Any]]):
    priority_weighed_delays = 0.0
    weight_sum = 0.0
    for execution in executed_tasks:
        task = execution["task"]
        if getattr(task, "is_break", False) or task.deadline is None:
            continue
        delay = (execution["actual_end"] - task.deadline).total_seconds() / 3600.0
        if delay > 0:
            priority_weighed_delays += PRIORITY_WEIGHTS.get(task.priority, 0.5) * delay
        weight_sum += PRIORITY_WEIGHTS[task.priority]
    return round(priority_weighed_delays/weight_sum, 4) if weight_sum > 0.0 else 0.0


def energy_distribution_score(executed_tasks: List[Dict[str, Any]], user) -> float:
    """
    Score of how well plan is adjusted to user attention distribution throughout the day
    Score of 1 means the assignment is perfect and most demanding tasks are assigned in peak attention time
    :param executed_tasks:
    :return:
    """
    if not executed_tasks:
        return 1.0

    total_weight = 0.0
    weighted_alignment = 0.0

    for execution in executed_tasks:
        task = execution["task"]
        if getattr(task, "is_break", False):
            continue

        # middle time of planned task (np. 14:30 -> 14.5)
        mid_dt = execution["actual_start"] + (execution["actual_end"] - execution["actual_start"]) / 2
        decimal_hour = mid_dt.hour + mid_dt.minute / 60.0 + mid_dt.second / 3600.0

        # normalized attention
        norm_attention = get_expected_attention(user.chronotype, decimal_hour)

        difficulty = get_task_difficulty(task, user)

        weighted_alignment += difficulty * norm_attention
        total_weight += difficulty

    return round(weighted_alignment / total_weight, 4) if total_weight > 0.0 else 1.0


def context_switch_score(executed_tasks: List[Dict[str, Any]]) -> dict[str, float]:
    """
    Checked on final plan that was executed,
    how much time loss there was the result of context switching between task types

    :param plan:
    :return:
    """
    if not executed_tasks:
        return {"total_switch_hours": 0.0, "avg_daily_switch_hours": 0.0, "efficiency_ratio": 1.0}

    tasks_by_day = defaultdict(list)
    for execution in executed_tasks:
        tasks_by_day[execution["actual_start"].date()].append(execution)

    total_switch_hours = 0.0
    total_work_hours = 0.0

    for day, day_tasks in tasks_by_day.items():
        # tasks sorted by start_time just in case
        day_tasks.sort(key=lambda x: x["actual_start"])

        prev_type = None
        for exc in day_tasks:
            task = exc["task"]
            # when break there is no context switch
            if getattr(task, "is_break", False):
                prev_type = None
                continue

            curr_type = task.type
            total_work_hours += exc["actual_duration_sec"] / 3600.0

            if prev_type is not None:
                cost, duration = calculate_switch_lag(prev_type, curr_type)
                total_switch_hours += cost * duration

            prev_type = curr_type

    num_days = max(1, len(tasks_by_day))
    avg_daily_switch = total_switch_hours / num_days

    # ratio of how much work time there was compared to time with included switch lag -> 1.0 no time lost on switch
    efficiency_ratio = (
        total_work_hours / (total_work_hours + total_switch_hours)
        if (total_work_hours + total_switch_hours) > 0 else 1.0
    )

    return {
        "total_switch_hours": round(total_switch_hours, 4),
        "avg_daily_switch_hours": round(avg_daily_switch, 4),
        "efficiency_ratio": round(efficiency_ratio, 4)
    }


def break_distribution_score(executed_tasks: List[Dict[str, Any]], user: User) -> Dict[str, float]:
    """
    Break statistics: summary of time, count of breaks,
    how many breaks compared to work time and long stretch without break penalty (>3.5h)
    """
    if not executed_tasks:
        return {"break_count": 0, "total_break_hours": 0.0, "break_ratio": 0.0, "long_stretch_penalty": 0.0}

    tasks_by_day = defaultdict(list)
    for exc in executed_tasks:
        tasks_by_day[exc["actual_start"].date()].append(exc)

    total_work_sec = 0.0
    total_break_sec = 0.0
    break_count = 0
    long_stretch_penalty = 0.0

    for day, day_tasks in tasks_by_day.items():
        day_tasks.sort(key=lambda x: x["actual_start"])
        time_since_last_break = 0.0

        for exc in day_tasks:
            is_break = getattr(exc["task"], "is_break", False)
            dur_sec = exc["actual_duration_sec"]

            if is_break:
                total_break_sec += dur_sec
                break_count += 1
                time_since_last_break = 0.0
            else:
                total_work_sec += dur_sec
                dur_hours = dur_sec / 3600.0
                time_since_last_break += dur_hours
                # Ergonomia: praca ciągła powyżej 3.5h bez odpoczynku
                if time_since_last_break > 3.5:
                    long_stretch_penalty += (time_since_last_break - 3.5)

    total_work_hours = total_work_sec / 3600.0
    total_break_hours = total_break_sec / 3600.0
    break_ratio = (total_break_hours / max(0.1, total_work_hours)) * 100.0

    return {
        "break_count": break_count,
        "total_break_hours": round(total_break_hours, 2),
        "break_ratio": round(break_ratio, 2),
        "long_stretch_penalty": round(long_stretch_penalty, 2)
    }


def overtime_score(executed_tasks: List[Dict[str, Any]], user: User) -> Dict[str, float]:
    """
    Summary of overtime work as well as average daily overtime and count of overtime days
    """
    _, work_end_hour = get_user_work_hours(user)

    tasks_by_day = defaultdict(list)
    for exc in executed_tasks:
        tasks_by_day[exc["actual_start"].date()].append(exc)

    total_overtime_hours = 0.0
    days_with_overtime = 0

    for day, day_tasks in tasks_by_day.items():
        # Najpóźniejszy koniec zadania w danym dniu
        latest_end_dt = max(t["actual_end"] for t in day_tasks)
        day_end_hour = latest_end_dt.hour + latest_end_dt.minute / 60.0 + latest_end_dt.second / 3600.0

        if day_end_hour > work_end_hour:
            overtime = day_end_hour - work_end_hour
            total_overtime_hours += overtime
            days_with_overtime += 1

    return {
        "total_overtime_hours": round(total_overtime_hours, 2),
        "avg_daily_overtime": round(total_overtime_hours / max(1, len(tasks_by_day)), 2),
        "days_with_overtime": days_with_overtime
    }


def instability_score(plans: List[dict]):
    """
    For generations of each plan we generate how much disruptions influenced the plan stability
    and take the average of that measure where disruptor task
    :param plans: plans of same group
    :return:
    """
    sorted_plans = sorted(plans, key=lambda p: p["generation"])
    previous_plan = None
    distance_factor = 0.3
    disruption_impact = 0.0
    tasks_count = 0
    for plan in sorted_plans:
        disr_time = plan.get("disruption_time")
        if previous_plan is None or disr_time is None:
            previous_plan = plan
            continue
        curr_tasks = plan["tasks"]
        prev_tasks = previous_plan["tasks"]
        for task_id, cur_t in curr_tasks.items():
            if cur_t.get("is_break"):
                continue
            # check just impact on the ones after disruption cause those before are not changing anymore
            if cur_t["start_time"] < disr_time:
                continue

            # we are not counting the added task
            if task_id in prev_tasks:
                prev_t = prev_tasks[task_id]
                old = prev_t["start_time"]
                new = cur_t["start_time"]

                delay_diff = abs((old - new).total_seconds()) / 3600.0
                dist_to_disrupt = max(0.0, (old - disr_time).total_seconds() / 3600.0)
                disruption_impact += delay_diff / ((dist_to_disrupt + 1.0) ** distance_factor)
                tasks_count += 1
        previous_plan = plan

    return round(disruption_impact/tasks_count, 4) if tasks_count > 0 else 0.0
