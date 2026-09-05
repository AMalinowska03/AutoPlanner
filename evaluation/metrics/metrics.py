from typing import List
import math
from simulation.UserSimulator import CHRONOTYPES, SKILL_ATTR_MAP, SWITCH_MATRIX, calculate_switch_lag
from collections import defaultdict
from data.DbModels import Execution, PlanTask, Plan, User

PRIORITY_WEIGHTS = {"low": 0.25, "medium": 0.5, "high": 0.75, "urgent": 1.0}


def daily_task_completion_score(planned_tasks, executed_tasks):
    """
    Fraction of tasks completed in a day compared to planned tasks
    :param planned_tasks: tasks that were planned for that day
    :param executed_tasks: tasks actually completed
    :return:
    """
    planned_tasks = [t for t in planned_tasks if t.task_id != 0]
    executed_tasks = [e for e in executed_tasks if e.plan_task and e.plan_task.task_id != 0]
    if len(planned_tasks) == 0 or len(executed_tasks) == 0:
        return 1
    return round(len(executed_tasks)/len(planned_tasks), 2)


def task_time_estimation_score(executed_tasks: List[Execution]) -> float:
    score = 0.0
    weight_sum = 0.0
    for execution in executed_tasks:
        planned_task = execution.plan_task
        if planned_task.task_id == 0 or planned_task.task is None:
            continue
        expected_time = (planned_task.end_time - planned_task.start_time).total_seconds()
        execution_time = (execution.end_time - execution.start_time).total_seconds()
        difference = abs(execution_time - expected_time)
        weight = PRIORITY_WEIGHTS[execution.plan_task.task.priority]
        weight_sum += weight
        score += (difference/max(expected_time, 1.0))*weight
    if weight_sum == 0.0:
        return 0.0
    return round(score/weight_sum, 4)


def task_execution_delay_score(executed_tasks: List[Execution]):
    priority_weighed_delays = 0.0
    weight_sum = 0.0
    for execution in executed_tasks:
        if execution.plan_task.task_id == 0 or execution.plan_task.task is None:
            continue
        task = execution.plan_task.task
        delay = (execution.end_time - task.deadline).total_seconds() / 3600.0
        if delay > 0:
            priority_weighed_delays += PRIORITY_WEIGHTS[task.priority] * delay
        weight_sum += PRIORITY_WEIGHTS[task.priority]
    if weight_sum == 0.0:
        return 0.0
    return round(priority_weighed_delays/weight_sum, 4)


def get_task_difficulty(task, user: User) -> float:
    param = SKILL_ATTR_MAP.get(task.type)
    skill = getattr(user, param["attr_name"], 0.5) if param else 0.5
    return float(task.workhours) * (1.5 - skill)


def get_expected_attention(chronotype: str, hour_decimal: float) -> float:
    peak = CHRONOTYPES[chronotype]["peak_attention_factor"]
    return 0.7 + 0.3 * math.sin(2 * math.pi * (hour_decimal - peak) / 24)


def energy_distribution_score(planned_tasks: List[PlanTask]) -> float:
    """
    Score of how well plan is adjusted to user attention distribution throughout the day
    Score of 1 means the assignment is perfect and most demanding tasks are assigned in peak attention time
    :param planned_tasks:
    :return:
    """
    if not planned_tasks:
        return 1.0

    total_weight = 0.0
    weighted_alignment = 0.0

    for pt in planned_tasks:
        if pt.task_id == 0 or pt.task is None:
            continue
        user = pt.user
        task = pt.task

        # middle time of planned task (np. 14:30 -> 14.5)
        mid_time = pt.start_time + (pt.end_time - pt.start_time) / 2
        decimal_hour = mid_time.hour + mid_time.minute / 60.0 + mid_time.second / 3600.0

        # estimated attention at given time [0.4, 1.0]
        attention = get_expected_attention(user.chronotype, decimal_hour)
        # normalization to [0, 1]
        norm_attention = (attention - 0.4) / 0.6

        difficulty = get_task_difficulty(task, user)

        weighted_alignment += difficulty * norm_attention
        total_weight += difficulty

    if total_weight == 0:
        return 1.0

    return round(weighted_alignment / total_weight, 4)


def context_switch_score(plan: Plan) -> dict[str, float]:
    """
    Checked on final plan that was executed,
    how much time loss there was the result of context switching between task types

    :param plan:
    :return:
    """
    tasks = list(plan.plan_tasks.values())
    if not tasks:
        return {"total_switch_hours": 0.0, "avg_daily_switch_hours": 0.0, "efficiency_ratio": 1.0}

    tasks_by_day = defaultdict(list)
    for pt in tasks:
        tasks_by_day[pt.start_time.date()].append(pt)

    total_switch_hours = 0.0
    total_work_hours = 0.0

    for day, day_tasks in tasks_by_day.items():
        # tasks sorted by start_time just in case
        day_tasks.sort(key=lambda x: x.start_time)

        prev_type = None
        for pt in day_tasks:
            # when break there is no context switch
            if pt.task_id == 0 or pt.task is None:
                prev_type = None
                continue

            curr_type = pt.task.type
            total_work_hours += float(pt.task.workhours)

            if prev_type is not None:
                total_switch_hours += calculate_switch_lag(prev_type, curr_type)

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


def instability_score(plans: List[Plan]):
    """
    For generations of each plan we generate how much disruptions influenced the plan stability
    and take the average of that measure where disruptor task
    :param plans: plans of same group
    :return:
    """
    previous_plan = None
    distance_factor = 0.3
    disruption_impact = 0.0
    tasks_count = 0
    for plan in plans:
        if previous_plan is None or plan.disruption_time is None:
            previous_plan = plan
            continue
        for current_plan_task_id in plan.plan_tasks:
            if current_plan_task_id == 0:
                continue
            current_plan_task = plan.plan_tasks.get(current_plan_task_id)
            # check just impact on the ones after disruption cause those before are not changing anymore
            if current_plan_task.start_time < plan.disruption_time:
                continue
            previous_plan_task = previous_plan.plan_tasks.get(current_plan_task_id)
            # we are not counting the added task
            if previous_plan_task:
                old = previous_plan_task.start_time
                new = current_plan_task.start_time
                disruption_impact += (
                        (abs(old - new).total_seconds()/3600.0)/(max(0.0, (old - plan.disruption_time).total_seconds()) / 3600.0 + 1.0)**distance_factor
                )
                tasks_count += 1
    if tasks_count == 0:
        return 0.0
    return round(disruption_impact/tasks_count, 4)
