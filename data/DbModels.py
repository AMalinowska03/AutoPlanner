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


class Plan(Base):
    __tablename__ = 'plan'
    id = Column(Integer, primary_key=True)
    algorithm = Column(Enum("ppo", "nsga", "baseline"))  # for filtering in data collection
    group = Column(Integer)  # for same month and user to group all generations for later measuring
    generation = Column(Integer)  # for same month and user, increased with each needed re-plan
    generating_time = Column(Float)  # how long this plan version was generated for
    disruption_time = Column(DateTime)  # how long this plan version was generated for


    plan_tasks = relationship(
        "PlanTask",
        back_populates="plan",
        primaryjoin="PlanTask.plan_id == Plan.id",
        collection_class=attribute_keyed_dict("task_id")
    )


class PlanTask(Base):
    __tablename__ = 'plan_task'
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey('plan.id'))
    user_id = Column(Integer, ForeignKey('user.id'))
    task_id = Column(Integer, ForeignKey('task.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    plan = relationship(
        "Plan",
        back_populates="plan_task",
        primaryjoin="PlanTask.plan_id == Plan.id",
    )
    task = relationship(
        "Task",
        back_populates="plan_task",
        primaryjoin="PlanTask.task_id == Task.id",
    )
    user = relationship(
        "User",
        back_populates="plan_task",
        primaryjoin="PlanTask.user_id == User.id",
    )


class Execution(Base):
    __tablename__ = 'execution'
    id = Column(Integer, primary_key=True)
    plan_task_id = Column(Integer, ForeignKey('plan_task.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    energy = Column(Float)

    plan_task = relationship(
        "PlanTask",
        back_populates="execution",
        primaryjoin="PlanTask.id == Execution.plan_task_id",
    )


def init_db():
    from data.database import engine
    Base.metadata.create_all(bind=engine)
