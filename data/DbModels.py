from sqlalchemy import Column, Integer, String, Text, Enum, ForeignKey, Float, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import VARCHAR
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


class Task(Base):
    __tablename__ = 'task'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(Integer, nullable=True)
    name = Column(String)
    workhours = Column(Integer)
    priority = Column(Enum("low", "medium", "high", "urgent"))
    type = Column(Enum("communication", "creativity", "technical", "routine", "analytical"))
    deadline = Column(DateTime)
    phase = Column(Enum("pretrain", "online", "disruptions"), default="online")
    is_disruptor = Column(Boolean, default=False)

class Plan(Base):
    __tablename__ = 'plan'
    id = Column(Integer, primary_key=True)
    generation = Column(Integer)


class PlanTask(Base):
    __tablename__ = 'plan_task'
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey('plan.id'))
    user_id = Column(Integer, ForeignKey('user.id'))
    task_id = Column(Integer, ForeignKey('task.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)

class Execution(Base):
    __tablename__ = 'execution'
    plan_task_id = Column(Integer, ForeignKey('plan_task.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    energy = Column(Float)
