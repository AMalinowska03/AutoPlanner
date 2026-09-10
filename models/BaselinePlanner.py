import copy
import shelve
import time
from datetime import datetime, timedelta
from typing import List, Optional

from data.DbHelper import get_user_work_hours, Repository
from data.DbModels import User, Task
from simulation.UserSimulator import UserSimulator

DisruptorsMap = dict[int, list[tuple[float, Task]]]

class BaselinePlanner:
    def __init__(self, user: User):
        self.user = user
        self.work_start_hour, self.work_end_hour = get_user_work_hours(user)
        self.repository = None

    def plan_and_simulate_month(self, month_tasks: List[Task], group_id: int, disruptors_map: Optional[DisruptorsMap] = None,
                                phase='online', phase_order=0, start_date=datetime(2027, 1, 4)):
        disr_map = copy.deepcopy(disruptors_map)
        self.repository = Repository(self.user, phase, phase_order, start_date)

        simulator = UserSimulator(self.user)
        current_generation = 0
        remaining_to_plan = copy.deepcopy(month_tasks)
        previous_plan_state = []

        sim_day = 0
        sim_time = self.work_start_hour

        disruption_occurrence_time = None
        time_since_last_break = 0.0
        total_break_time_today = 0.0
        last_task_type = None
        while remaining_to_plan and sim_day < 20:
            print(f"Base ------ Planning: user {self.user.id} | generation: {current_generation} | start date: {start_date}")
            current_plan, gen_time = self._generate_plan(remaining_to_plan, sim_day, sim_time,
                                                         self.work_start_hour, self.work_end_hour,
                                                         start_date
                                                         )

            # save plan to db
            plan_record = self.repository.create_plan_records(
                algorithm="baseline", planned_tasks=current_plan, group_id=group_id,
                generation=current_generation, disruption_time=disruption_occurrence_time, generating_time=gen_time
            )

            print(f"Base ------ Simulating ------")
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
                    if pending_disruptors and pending_disruptors[0][0] <= sim_time:
                        disrupt_time, disruptor_task = pending_disruptors.pop(0)
                        dh = int(disrupt_time)
                        dm = int((disrupt_time - dh) * 60)
                        disruption_occurrence_time = start_date + timedelta(days=calendar_days_passed, hours=dh,
                                                                            minutes=dm)
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        remaining_to_plan.append(disruptor_task)

                        # set to history to check instability
                        previous_plan_state = copy.deepcopy(current_plan)
                        replan_needed = True
                        print(f"Base ------ Disruptor occurred: RE-PLANNING ------")
                        break
                disruption_occurrence_time = None

                current_sim_dt = start_date + timedelta(days=calendar_days_passed, hours=int(sim_time),
                                                        minutes=int((sim_time % 1) * 60))
                # if we finish task earlier we might want to re-plan
                # because other task might be more efficiently performed in that gap
                if current_plan:
                    next_item = current_plan[0]
                    next_start_dt = next_item["start_time"]

                    # time to next task
                    gap_seconds = (next_start_dt - current_sim_dt).total_seconds()

                    # re-plan if:
                    # - there is more than one hour to next task start
                    # - next task is in next day, and we still have over 0.5h of work day
                    if gap_seconds > 15*60 or (
                            next_start_dt.date() > current_sim_dt.date() and self.work_end_hour - sim_time > 0.5):
                        remaining_to_plan = [item["task"] for item in current_plan if "task" in item]
                        previous_plan_state = copy.deepcopy(current_plan)
                        replan_needed = True
                        print(f"Base ------ Have time left: RE-PLANNING ------")
                        break

                # end of day
                if sim_time >= self.work_end_hour:
                    sim_day += 1
                    if sim_day >= 20:
                        if current_plan:
                            print(f"Base ------ End of month with tasks left ------")
                        remaining_to_plan = []
                        break
                    simulator.reset(sim_time, self.work_end_hour, weekly=(sim_day % 5 == 0))
                    sim_time = self.work_start_hour
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
                            print(f"Base ------ Tasks left from day: RE-PLANNING ------")
                            break

            # if we moved through tasks without re-planning we finish month
            if not replan_needed:
                remaining_to_plan = []
                print(f"Base ------ Simulation END ------")
            else:
                current_generation += 1


        return {
            "total_replans": current_generation,
            "days_used": sim_day,
        }

    def _generate_plan(self, remaining_tasks, sim_day, sim_time,
                       work_start_hour, work_end_hour, start_date):
        t0 = time.time()
        prio_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}

        def sort_key(sorted_task) -> tuple[datetime, int]:
            """
            Sorts tasks by deadline and priority from highest to lowest
            :param sorted_task: task being sorted
            :return:
            """
            dl = sorted_task.deadline if sorted_task.deadline else datetime.max  # no deadline means end of month
            prio = prio_map.get(sorted_task.priority, 1)
            return dl, -prio

        sorted_tasks = sorted(remaining_tasks, key=sort_key)  # sorting

        current_plan = []
        c_day = sim_day
        c_time = sim_time
        lunch_taken_today = c_time >= 14.0  # if re-plan starts after 2pm then skip break

        for task in sorted_tasks:
            duration = float(task.workhours)

            cal_days = c_day + (c_day // 5) * 2
            start_dt = start_date + timedelta(days=cal_days, hours=int(c_time), minutes=int((c_time % 1) * 60))
            end_dt = start_dt + timedelta(hours=duration)

            current_plan.append({
                "task_id": task.id,
                "task": task,
                "duration": duration,
                "start_time": start_dt,
                "end_time": end_dt,
                "_abs_start": c_day * 24.0 + c_time
            })

            c_time += duration

            # add break between 11:00-14:30
            if 11.0 <= c_time <= 14.5 and not lunch_taken_today:
                break_dur = 0.75  # 45 minutes
                b_start_dt = end_dt
                b_end_dt = b_start_dt + timedelta(hours=break_dur)

                current_plan.append({
                    "is_break": True,
                    "duration": break_dur,
                    "start_time": b_start_dt,
                    "end_time": b_end_dt
                })
                c_time += break_dur
                lunch_taken_today = True

            # move to next day if overtime
            if c_time >= work_end_hour:
                c_day += 1
                c_time = work_start_hour
                lunch_taken_today = False

        generation_time = round(time.time() - t0, 4)
        return current_plan, generation_time

