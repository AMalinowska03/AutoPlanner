from sqlalchemy import Column, Integer, String, Text, Enum, ForeignKey, Float, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.orm.collections import attribute_keyed_dict
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    chronotype = Column(Enum("morning_lark", "intermediate", "night_owl"))
    communication_skill = Column(Float)
    creativity_skill = Column(Float)
    technical_skill = Column(Float)
    routine_skill = Column(Float)
    analytical_skill = Column(Float)
    procrastination_probability = Column(Float)
    work_start_time = Column(DateTime)
    work_end_time = Column(DateTime)
    is_training = Column(Boolean, default=False)


class Task(Base):
    __tablename__ = 'task'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    workhours = Column(Float)
    priority = Column(Enum("low", "medium", "high", "urgent"))
    type = Column(Enum("communication", "creativity", "technical", "routine", "analytical"))
    deadline = Column(DateTime)
    phase = Column(Enum("pretrain", "finetune", "online", "disruptions"), default="online")
    phase_order = Column(Integer)  # number of sub phase as a monthly task set available
    is_disruptor = Column(Boolean, default=False)
    injection_time = Column(DateTime, nullable=True)  # by that time disruptor is supposed to be added to plan
    is_break = Column(Boolean, default=False)


class ExperimentMetric(Base):
    __tablename__ = 'experiment_metric'

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_type = Column(Enum('online', 'disruptions'))
    algorithm = Column(Enum('ppo', 'nsga', 'baseline'))
    user_id = Column(Integer, index=True)
    phase_order = Column(Integer, index=True)
    group_id = Column(Integer)

    total_replans = Column(Integer)
    days_used = Column(Integer)
    avg_generating_time = Column(Float)

    monthly_completion_score = Column(Float)
    daily_completion_score = Column(Float)
    time_estimation_error = Column(Float)
    delay_score = Column(Float)
    energy_score = Column(Float)
    switch_efficiency = Column(Float)
    instability = Column(Float)
    total_overtime_hours = Column(Float, default=0.0)
    break_ratio = Column(Float, default=0.0)
    break_count = Column(Integer, default=0)
    long_stretch_penalty = Column(Float, default=0.0)
    urgent_delayed_count = Column(Integer, default=0)


class BreakTask:
    def __init__(self, duration_hours: float):
        self.id = -1
        self.name = "Break"
        self.workhours = duration_hours
        self.priority = "low"
        self.type = "break"
        self.deadline = None
        self.is_break = True


def init_db():
    from data.database import engine
    Base.metadata.create_all(bind=engine)
