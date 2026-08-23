from sqlalchemy import Column, Integer, String, Text, Enum, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import VARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    chronotype = Column(Enum("morning_lark", "intermediate", "night_owl"))

