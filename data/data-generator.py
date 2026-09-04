from typing import List, Tuple, Dict, Union

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.DbModels import Base, User, Task

DATABASE_URL = "sqlite:///planner_experiment_data.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(engine)


def load_raw_csvs():
    """
    Retrieves raw data from CSV files.
    :return:
    """
    df_cog = pd.read_csv("kaggle_datasets/cognitive_biometric_optimization.csv")
    df_hr = pd.read_csv("kaggle_datasets/hr_allocation_dataset.csv")
    df_tasks = pd.read_csv("kaggle_datasets/taskdata.csv")
    return df_cog, df_hr, df_tasks

def process_users(df_hr, df_cog):
    """
    Creates base for users with normalized skill levels, procrastination probability and added chronotypes
    Added from hr_allocation_dataset with inclusions based on cognitive_biometric_optimization dataset
    """
    users = df_hr[['employee_id', 'technical_skill_score', 'communication_score',
                   'problem_solving_score', 'idle_time_hours', 'attendance_rate']].drop_duplicates(subset=['employee_id']).copy()

    users.rename(columns={'employee_id': 'id'}, inplace=True)

    # scaling skills to normalized values
    users['technical_skill'] = (users['technical_skill_score'] / 100.0).round(2)
    users['communication_skill'] = (users['communication_score'] / 100.0).round(2)
    users['analytical_skill'] = (users['problem_solving_score'] / 100.0).round(2)

    # generate missing skill levels randomly
    np.random.seed(42)
    users['creativity_skill'] = np.round(np.random.beta(3, 3, size=len(users)), 2)
    users['routine_skill'] = np.round(np.random.beta(4, 2, size=len(users)), 2)

    # probability that this user will procrastinate based on db data
    users['procrastination_probability'] = np.clip(
        (users['idle_time_hours'] / 8.0) * (1.0 - (users['attendance_rate'] / 100.0)) * 2.5,
        0.05, 0.95
    ).round(2)

    # chronotypes distributed as in cognitive_biometric_optimization limited to 3 types
    chrono_map = {
        'morning_lark': 'morning_lark',
        'lark': 'morning_lark',
        'evening_owl': 'night_owl',
        'night_owl': 'night_owl',
        'intermediate': 'intermediate'
    }

    cog_chronos = df_cog['chronotype'].str.lower().map(
        lambda x: next((v for k, v in chrono_map.items() if k in str(x)), 'intermediate'))
    chrono_probs = cog_chronos.value_counts(normalize=True)

    users['chronotype'] = np.random.choice(
        chrono_probs.index,
        size=len(users),
        p=chrono_probs.values
    )

    # random shift time assigned to user so different chronotypes realistically have to adapt to various conditions
    shift_starts = [7, 8, 9]
    random_starts = np.random.choice(shift_starts, size=len(users))
    users['work_start_time'] = [datetime(2027, 2, 1, h, 0) for h in random_starts]
    users['work_end_time'] = [datetime(2027, 2, 1, h + 8, 0) for h in random_starts]

    cols_to_drop = ['technical_skill_score', 'communication_score',
                    'problem_solving_score', 'idle_time_hours', 'attendance_rate']
    return users.drop(columns=cols_to_drop)

def process_and_combine_tasks(df_hr, df_tasks):
    """
    Combines tasks from 2 sources, removes duplicates and classifies predefined types.
    :param df_hr:
    :param df_tasks:
    :return:
    """

    # tasks from taskdata dataset
    td_tasks = df_tasks.dropna(subset=['task_description']).copy()
    td_tasks['name'] = td_tasks['task_description'] + " (" + td_tasks['project'] + ")"
    td_tasks['workhours'] = pd.to_numeric(td_tasks['estimated_duration'].str.replace(',', '.'), errors='coerce').fillna(2).astype(int).clip(1, 16)
    td_tasks['department'] = td_tasks['dept']
    td_tasks['task_priority'] = td_tasks['priority']

    # tasks from hr dataset
    hr_tasks = df_hr[['task_id', 'department', 'task_complexity', 'workload_hours', 'task_priority']].drop_duplicates(subset=['task_id']).copy()
    def hr_task_name(row):
        return get_task_name_for_hr_dataset(df_tasks, row)
    hr_tasks['name'] = hr_tasks.apply(hr_task_name, axis=1)
    hr_tasks['workhours'] = hr_tasks['workload_hours'].astype(float)

    # concatenating tasks from two sources
    combined = pd.concat([hr_tasks[['name', 'workhours', 'task_priority', 'department']],
                          td_tasks[['name', 'workhours', 'task_priority', 'department']]], ignore_index=True)
    combined = process_and_split_tasks(combined)

    # map types and priority for db enum values
    def get_type(dept):
        d = str(dept).lower()
        if any(k in d for k in ['tech', 'it', 'eng', 'code']): return 'technical'
        if any(k in d for k in ['market', 'sales', 'talk', 'client']): return 'communication'
        if any(k in d for k in ['analytic', 'finance', 'data']): return 'analytical'
        if any(k in d for k in ['design', 'media', 'creative']): return 'creativity'
        return 'routine'

    def adjust_priority_distribution(p):
        """
        Demotes priority of high/urgent tasks to make priority proportion closer to 10/20/45/15 (urgent->low)
        """
        p_str = str(p).lower()
        rand = np.random.rand()

        if 'critical' in p_str or 'urgent' in p_str:
            if rand < 0.50:
                return 'medium'
            elif rand < 0.55:
                return 'high'
            return 'urgent'

        if 'high' in p_str:
            if rand < 0.45:
                return 'medium'
            elif rand < 0.50:
                return 'low'
            return 'high'

        if 'medium' in p_str:
            return 'medium'

        return 'low'

    combined['type'] = combined['department'].apply(get_type)
    np.random.seed(42)
    combined['priority'] = combined['task_priority'].apply(adjust_priority_distribution)

    return combined[['name', 'workhours', 'priority', 'type']].reset_index(drop=True)


def get_task_name_for_hr_dataset(df_tasks, row):
    dept_description_samples = {
        'IT': ['Code feature', 'Bug fixing', 'API Integration', 'Code review', 'Deploy to staging'],
        'Engineering': ['Code feature', 'Bug fixing', 'System refactoring', 'DB optimization'],
        'Analytics': ['Write report', 'Design dashboard', 'Data cleanup', 'Metrics review'],
        'Marketing': ['Write proposal', 'Campaign review', 'Create ad copy', 'Social media update'],
        'Sales': ['Write proposal', 'Client demo', 'Follow up calls', 'CRM Update'],
        'HR': ['Team meeting', 'Interview candidate', 'Update employee records', 'Onboarding'],
        'Finance': ['Write report', 'Budget audit', 'Invoice processing', 'Monthly review']
    }
    all_td_descriptions = df_tasks['task_description'].dropna().unique().tolist()
    dept = str(row['department'])
    possible_descriptions = dept_description_samples.get(dept, all_td_descriptions)
    desc = np.random.choice(possible_descriptions)
    return f"{desc} #{row['task_id']} ({dept})"


def process_and_split_tasks(df_input):
    processed_records = []
    group_counter = 1
    for _, row in df_input.iterrows():
        total_hours = row['workhours']
        name = row['name']
        priority = row['task_priority']
        dept = row['department']

        # if task is longer than 3h we divide it into subtasks as simulator does not include it
        # division by model would be another complexity layer and not easily measurable
        if total_hours > 3.0:
            current_group_id = group_counter
            group_counter += 1
            remaining = total_hours
            part = 1

            while remaining > 0.0:
                # choosing time to allocate for subtask for 0.5-3h in 0.5h increments or rest of remaining time
                sub_duration = min(remaining, np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
                sub_duration = round(float(sub_duration), 2)

                processed_records.append({
                    'task_group_id': current_group_id,
                    'name': f"{name} (Part {part})",
                    'workhours': sub_duration,
                    'task_priority': priority,
                    'department': dept
                })

                remaining -= sub_duration
                part += 1

        # if task can be completed in 3h finish it in one
        else:
            work_h = float(np.clip(total_hours, 0.08, 3.0))

            # shorten some short tasks to under an hour to diversify
            if work_h <= 1.0 and np.random.rand() < 0.3:
                work_h = float(np.random.choice([0.08, 0.17, 0.25, 0.5]))

            processed_records.append({
                'task_group_id': None,
                'name': name,
                'workhours': round(work_h, 2),
                'task_priority': priority,
                'department': dept
            })

    return pd.DataFrame(processed_records)


# def generate_deadlines_for_phase(df_tasks, phase, start_date, end_date, seed=42):
#     """
#     Generates deadlines
#     :param df_tasks: list of all tasks
#     :param phase: experiment phase name
#     :param start_date:
#     :param end_date:
#     :param seed: random seed
#     :return:
#     """
#     np.random.seed(seed)
#     full_df = []
#     for sub_phase_tasks in df_tasks:
#         df = sub_phase_tasks.copy().reset_index(drop=True)
#         if len(df) == 0:
#             return df
#
#         business_days = pd.date_range(start=start_date, end=end_date, freq='B')
#
#         # available deadline times, preferably end of day/midday
#         possible_times, probabilities = [], []
#         for h in range(8, 17):
#             for m in range(0, 60, 5):
#                 if h == 8 and m < 30:
#                     continue
#                 possible_times.append((h, m))
#                 weight = 3.0 if h in [15, 16] else (1.5 if h in [11, 12] else 1.0)
#                 probabilities.append(weight)
#
#         probabilities = np.array(probabilities) / sum(probabilities)
#
#         deadlines = []
#         for _ in range(len(df)):
#             day = np.random.choice(business_days)
#             time_idx = np.random.choice(len(possible_times), p=probabilities)
#             h, m = possible_times[time_idx]
#             deadlines.append(pd.Timestamp(day).replace(hour=h, minute=m, second=0))
#
#         df['phase'] = phase
#         df['deadline'] = deadlines
#         df['is_disruptor'] = False
#         full_df.append(df)
#     return full_df

def generate_deadlines_for_phase(
        df_tasks: List[pd.DataFrame],
        phase: str,
        start_date: Union[str, pd.Timestamp, datetime, List],
        end_date: Union[str, pd.Timestamp, datetime, List] = None,
        shift_per_subphase: bool = True,
        seed: int = 42
) -> List[pd.DataFrame]:
    """
    Generuje deadliny dla podfaz.
    - Jeśli start_date to lista: każda podfaza i bierze start i end z listy[i].
    - Jeśli start_date to pojedyncza data i shift_per_subphase=True:
      kolejne podfazy dostają kolejne okna po dokładnie 4 tygodnie (20 dni roboczych).
    - Jeśli start_date to pojedyncza data i shift_per_subphase=False:
      wszystkie podfazy mają dokładnie takie samo 4-tygodniowe okno (dla pretreningu).
    """
    np.random.seed(seed)
    full_df = []

    # 1. Rozkład godzinowy (preferowany koniec dnia i okolice lunchu)
    possible_times, time_weights = [], []
    for h in range(8, 17):
        for m in range(0, 60, 5):
            if h == 8 and m < 30:
                continue
            possible_times.append((h, m))
            weight = 3.0 if h in [15, 16] else (1.5 if h in [11, 12] else 1.0)
            time_weights.append(weight)
    time_probs = np.array(time_weights) / sum(time_weights)

    # 2. Przetwarzanie podfaz
    for idx, sub_phase_tasks in enumerate(df_tasks):
        df = sub_phase_tasks.copy().reset_index(drop=True)
        if len(df) == 0:
            full_df.append(df)
            continue

        # Ustalenie zakresu 20 dni roboczych (dokładnie 4 tygodnie)
        if isinstance(start_date, list):
            sub_start = pd.to_datetime(start_date[idx])
            sub_end = pd.to_datetime(end_date[idx]) if end_date is not None else None
            if sub_end is None:
                business_days = pd.date_range(start=sub_start, periods=20, freq='B')
            else:
                business_days = pd.date_range(start=sub_start, end=sub_end, freq='B')
        else:
            base_start = pd.to_datetime(start_date)
            if shift_per_subphase:
                # Kolejne podfazy to kolejne miesiące (+ idx * 4 tygodnie)
                sub_start = base_start + timedelta(weeks=4 * idx)
            else:
                # Każda podfaza w tym samym oknie czasowym
                sub_start = base_start

            # Dokładnie 4 tygodnie robocze = 20 dni
            business_days = pd.date_range(start=sub_start, periods=20, freq='B')

        deadlines = []
        # Rozkład dni w zależności od priorytetu
        # (urgent: dni 0-4, high: dni 2-9, medium: dni 5-15, low: dni 8-19)
        for _, row in df.iterrows():
            prio = getattr(row, 'priority', 'medium')

            if prio == 'urgent':
                day_pool = business_days[:5]
            elif prio == 'high':
                day_pool = business_days[2:10]
            elif prio == 'medium':
                day_pool = business_days[5:16]
            else:  # low
                day_pool = business_days[8:]

            chosen_day = np.random.choice(day_pool)
            time_idx = np.random.choice(len(possible_times), p=time_probs)
            h, m = possible_times[time_idx]

            deadlines.append(pd.Timestamp(chosen_day).replace(hour=h, minute=m, second=0))

        df['phase'] = phase
        df['deadline'] = deadlines
        df['is_disruptor'] = False
        full_df.append(df)

    return full_df


def create_disruptor_tasks(count, start_date, end_date, seed=999):
    """
    Generates disruptor tasks that will be added in experiment 2 during execution
    :param count: number of tasks to generate
    :param start_date: deadline start date
    :param end_date: deadline end date
    :param seed: random seed
    :return:
    """
    np.random.seed(seed)
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')

    types = ['communication', 'routine', 'technical', 'analytical', 'creativity']
    type_probs = [0.35, 0.35, 0.10, 0.10, 0.10]  # most disruptors are communication or routine usually

    disruptor_templates = {
        'communication': ['Urgent client call', 'Ad-hoc sync with lead', 'Emergency mail response',
                          'Slack escalations'],
        'routine': ['Quick System fix', 'Approve urgent invoice', 'Access permission grant', 'Status update'],
        'technical': ['Hotfix server crash', 'DB connection drop', 'Critical bug patch'],
        'analytical': ['Quick metrics verification', 'Data drop anomaly check'],
        'creativity': ['Urgent banner revision', 'Copywriting fix']
    }

    disruptor_records = []

    for i in range(1, count + 1):
        t_type = np.random.choice(types, p=type_probs)
        t_name = f"[DISRUPTOR] {np.random.choice(disruptor_templates[t_type])} #{i}"

        # 5 to 60 minutes (0.08h - 1.0h)
        duration = float(np.random.choice([round((x * 5.0 / 60.0), 2) for x in range(1, 13)]))

        day = np.random.choice(business_days)
        hour = np.random.choice(range(9, 17))
        minute = np.random.choice([0, 15, 30, 45])
        deadline = pd.Timestamp(day).replace(hour=hour, minute=minute, second=0)

        disruptor_records.append({
            'task_group_id': None,
            'name': t_name,
            'workhours': duration,
            'task_priority': np.random.choice(['high', 'urgent'], p=[0.4, 0.6]),
            'department': t_type.capitalize(),
            'type': t_type,
            'phase': 'test_week',
            'deadline': deadline,
            'is_disruptor': True
        })

    return pd.DataFrame(disruptor_records)


def divide_tasks_to_phases(
        task_pool: List[Task],
        min_sub_workhours: float = 90.0,
        max_sub_workhours: float = 120.0,
        target_base_task_count: int = 100
) -> Tuple[List[List[Task]], List[List[Task]], List[List[Task]], List[List[Task]]]:
    """
    Dzieli pulę zadań na fazy: pretrain (60%), finetune (10%), phase1 (20%), phase2 (10%).
    Wewnątrz każdej fazy tworzy podfazy o łącznym czasie w przedziale [90, 120] roboczogodzin,
    zachowując zadane proporcje priorytetów i oznaczając zadania polem `phase_order`.

    Zwraca: (pretrain_subs, finetune_subs, phase1_subs, phase2_subs), gdzie każdy element
    to lista podfaz: [[task1, task2, ...], [task1, task2, ...], ...]
    """
    total_tasks = len(task_pool)
    shuffled_pool = list(task_pool)
    random.shuffle(shuffled_pool)

    phase_targets = [
        ('pretrain', int(total_tasks * 0.60)),
        ('finetune', int(total_tasks * 0.10)),
        ('phase1', int(total_tasks * 0.20)),
        ('phase2', None)  # rest
    ]

    prio_weights = {
        "urgent": 0.10,
        "high": 0.20,
        "medium": 0.45,
        "low": 0.25
    }
    remaining_pool = list(shuffled_pool)

    df: Dict[str, List[List[Task]]] = {}

    # monthly task lists for training epochs/executions
    for phase_name, target_count in phase_targets:
        if target_count is not None:
            pool = remaining_pool[:target_count]
            remaining_pool = remaining_pool[target_count:]
        else:
            pool = remaining_pool
            remaining_pool = []
        by_prio = {
            "urgent": [t for t in pool if t.priority == "urgent"],
            "high": [t for t in pool if t.priority == "high"],
            "medium": [t for t in pool if t.priority == "medium"],
            "low": [t for t in pool if t.priority == "low"]
        }

        sub_phases = []
        phase_order = 0

        while True:
            total_available = sum(len(v) for v in by_prio.values())
            # Zakończ podział, gdy pula jest zbyt mała, aby utworzyć nową podfazę
            if total_available < int(target_base_task_count * 0.6):
                break

            # Obliczenie wstępnej liczby zadań per priorytet
            counts = {prio: int(target_base_task_count * w) for prio, w in prio_weights.items()}

            selected_batch = []
            # Wyciągamy zadania ze zbioru (bez zwracania)
            for prio, count in counts.items():
                available = by_prio[prio]
                take_cnt = min(count, len(available))
                taken = random.sample(available, take_cnt)
                selected_batch.extend(taken)
                for t in taken:
                    available.remove(t)

            total_hours = sum(float(t.workhours) for t in selected_batch)

            # add tasks if there is still time to fill
            attempts = 0
            while total_hours < min_sub_workhours and attempts < 5:
                available_prios = [p for p, tasks in by_prio.items() if len(tasks) > 0]
                if not available_prios:
                    break  # no more tasks for phase

                weights_subset = [prio_weights[p] for p in available_prios]
                chosen_prio = random.choices(available_prios, weights=weights_subset, k=1)[0]

                new_t = by_prio[chosen_prio].pop(random.randrange(len(by_prio[chosen_prio])))
                selected_batch.append(new_t)
                total_hours += float(new_t.workhours)

            # take out tasks if they exceed max hours
            attempts = 0
            while total_hours > max_sub_workhours and len(selected_batch) > 0 and attempts < 5:
                # take out random tasks and put back to pool
                drop_idx = random.randrange(len(selected_batch))
                dropped_t = selected_batch.pop(drop_idx)
                by_prio[dropped_t.priority].append(dropped_t)
                total_hours -= float(dropped_t.workhours)

            # when correct subset ready save it
            if total_hours >= min_sub_workhours:
                for t in selected_batch:
                    t.phase_order = phase_order

                sub_phases.append(selected_batch)
                phase_order += 1
            else:
                # not enough hours and not enough tasks to choose from - return to pool
                for t in selected_batch:
                    by_prio[t.priority].append(t)
                break

        df[phase_name] = sub_phases
        for tasks in by_prio.values():
            remaining_pool.extend(tasks)

    return df['pretrain'], df['finetune'], df['phase1'], df['phase2']

def run_pipeline():
    print("Loading CSV...")
    df_cog, df_hr, df_tasks = load_raw_csvs()

    print("Processing users...")
    users_df = process_users(df_hr, df_cog)

    print("Combining and cleaning up tasks...")
    clean_tasks_pool = process_and_combine_tasks(df_hr, df_tasks)

    print("Dividing tasks and generating deadlines...")
    pretrain_tasks_lists_df, finetune_tasks_lists_df, phase1_tasks_lists_df, phase2_tasks_lists_df = divide_tasks_to_phases(clean_tasks_pool)
    pretrain_tasks_df = generate_deadlines_for_phase(pretrain_tasks_lists_df, "pretrain", '2027-02-01', '2027-02-28', 100)
    finetune_tasks_df = generate_deadlines_for_phase(finetune_tasks_lists_df, "finetune", '2027-02-01', '2027-02-28', 101)
    phase1_tasks_df = generate_deadlines_for_phase(phase1_tasks_lists_df, "online", '2027-02-01', '2027-02-28', 102)
    phase2_tasks_df = generate_deadlines_for_phase(phase2_tasks_lists_df, "disruptors", '2027-02-01', '2027-02-28', 103)
    disruptor_tasks = create_disruptor_tasks(20, '2027-05-31', '2027-06-27', 103)

    print("Saving to SQLite...")
    session = SessionLocal()
    try:
        # save users
        users_df.to_sql('user', con=engine, if_exists='append', index=False)

        # save all tasks
        all_tasks = pd.concat([pretrain_tasks_df, finetune_tasks_df, phase1_tasks_df, phase2_tasks_df, disruptor_tasks], ignore_index=True)
        all_tasks.to_sql('task', con=engine, if_exists='append', index=False)

        session.commit()
        print("Success! Database if filled.")
    except Exception as e:
        session.rollback()
        print(f"Error when saving to database: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    run_pipeline()