import pandas as pd
import numpy as np
from datetime import datetime
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

    def get_priority(p):
        p_str = str(p).lower()
        if 'critical' in p_str or 'urgent' in p_str: return 'urgent'
        if 'high' in p_str: return 'high'
        if 'medium' in p_str: return 'medium'
        return 'low'

    combined['type'] = combined['department'].apply(get_type)
    combined['priority'] = combined['task_priority'].apply(get_priority)

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
        total_hours = row['workload']
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


def generate_deadlines_for_phase(df_tasks, phase, start_date, end_date, seed=42):
    """
    Generates deadlines
    :param df_tasks: list of all tasks
    :param phase: experiment phase name
    :param start_date:
    :param end_date:
    :param seed: random seed
    :return:
    """
    np.random.seed(seed)
    df = df_tasks.copy().reset_index(drop=True)
    if len(df) == 0:
        return df

    business_days = pd.date_range(start=start_date, end=end_date, freq='B')

    # available deadline times, preferably end of day/midday
    possible_times, probabilities = [], []
    for h in range(8, 17):
        for m in range(0, 60, 5):
            if h == 8 and m < 30: continue
            possible_times.append((h, m))
            weight = 3.0 if h in [15, 16] else (1.5 if h in [11, 12] else 1.0)
            probabilities.append(weight)

    probabilities = np.array(probabilities) / sum(probabilities)

    deadlines = []
    for _ in range(len(df)):
        day = np.random.choice(business_days)
        time_idx = np.random.choice(len(possible_times), p=probabilities)
        h, m = possible_times[time_idx]
        deadlines.append(pd.Timestamp(day).replace(hour=h, minute=m, second=0))

    df['phase'] = phase
    df['deadline'] = deadlines
    df['is_disruptor'] = False
    return df


def create_disruptor_tasks(count, start_date='2027-03-01', end_date='2027-03-07', seed=999):
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
        duration = float(np.random.choice([round((x * 5.0 /60.0), 2) for x in range(1, 13)]))

        # second phase of experiment takes place for first week of march
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


def build_full_experiment_dataset(combined_tasks_df):
    """
    Dzieli bazową pulę na fazy i generuje zestaw zadań typu disruptor.
    """
    df_all = combined_tasks_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    total_base_count = len(df_all)

    # 1. Obliczenie proporcji
    # np. 60% Pretraining, 25% Online Training (Luty), 15% Test Week (Marzec)
    n_pretrain = int(total_base_count * 0.60)
    n_online = int(total_base_count * 0.25)

    df_pretrain = df_all.iloc[:n_pretrain].copy()
    df_online = df_all.iloc[n_pretrain: n_pretrain + n_online].copy()
    df_test_week_regular = df_all.iloc[n_pretrain + n_online:].copy()

    # 2. Przypisanie okien czasowych i deadlinów
    # Pretraining: szerokie okno styczeń/luty 2027
    df_pretrain = generate_deadlines_for_phase(df_pretrain, 'pretraining', '2027-01-15', '2027-02-28', seed=101)

    # Online Training: Luty 2027 (01.02 - 28.02)
    df_online = generate_deadlines_for_phase(df_online, 'online_training', '2027-02-01', '2027-02-28', seed=102)

    # Test Week - Zwykłe zadania: 1. tydzień Marca 2027 (01.03 - 07.03)
    df_test_week_regular = generate_deadlines_for_phase(df_test_week_regular, 'test_week', '2027-03-01', '2027-03-07',
                                                        seed=103)

    # 3. Wygenerowanie DISRUPTORÓW (~10% łącznej liczby zadań w systemie)
    n_disruptors = int(total_base_count * 0.10)
    df_disruptors = create_disruptor_tasks(count=n_disruptors, start_date='2027-03-01', end_date='2027-03-07', seed=202)

    # 4. Połączenie wszystkich zbiorów w jeden eksperymentalny DataFrame
    full_dataset = pd.concat([
        df_pretrain,
        df_online,
        df_test_week_regular,
        df_disruptors
    ], ignore_index=True)

    return full_dataset

# 3. GLÓWNY SKRYPT WYKONAWCZY
def run_pipeline():
    print("Wczytywanie CSV...")
    df_cog, df_hr, df_tasks = load_raw_csvs()

    print("Przetwarzanie Użytkowników...")
    users_df = process_users(df_hr, df_cog)

    print("Łączenie i czyszczenie Zadań...")
    clean_tasks_pool = process_and_combine_tasks(df_hr, df_tasks)

    print("Generowanie deadlinów na Luty 2027 (Zbioru testowego i treningowego)...")
    train_tasks_df = generate_deadlines_feb_2027(clean_tasks_pool, count=300)
    test_tasks_df = generate_deadlines_feb_2027(clean_tasks_pool, count=1000)

    # Zapis do bazy za pomocą Pandas to_sql
    print("Zapisywanie danych do SQLite...")
    session = SessionLocal()
    try:
        # Wstawienie użytkowników
        users_df.to_sql('user', con=engine, if_exists='append', index=False)

        # Wstawienie zadań (połączone zbiory)
        all_tasks = pd.concat([train_tasks_df, test_tasks_df], ignore_index=True)
        all_tasks.to_sql('task', con=engine, if_exists='append', index=False)

        session.commit()
        print("Zakończono sukcesem! Baza została wypełniona.")
    except Exception as e:
        session.rollback()
        print(f"Błąd podczas zapisu do bazy: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    run_pipeline()