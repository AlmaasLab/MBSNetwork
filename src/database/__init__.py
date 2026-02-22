import os

from sqlalchemy import NullPool, create_engine
from sqlalchemy.orm import Session as _Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists

from database.datamodel import Base

engine_kwargs = {"poolclass": NullPool}

PG_HOST = os.environ["PG_HOST"]
PG_PORT = os.environ["PG_PORT"]
PG_DB = os.environ["PG_DB"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]

DB_URL = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(DB_URL, **engine_kwargs)

if not database_exists(engine.url):
    create_database(engine.url)

Base.metadata.create_all(engine)
Session: sessionmaker[_Session] = sessionmaker(engine)
