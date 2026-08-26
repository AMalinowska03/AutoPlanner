# app/db_config.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database URL format: mysql+pymysql://username:password@localhost/db_name
DATABASE_URL = "mysql+pymysql://root@localhost:3306/auto_planner"

# Create the database engine
engine = create_engine(DATABASE_URL, connect_args={
    "host": "localhost", "user": "root", "password": "", "database": "auto_planner"
})

# SessionLocal will be used to create a session to interact with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
