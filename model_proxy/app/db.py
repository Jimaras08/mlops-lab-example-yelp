import logging

from databases import Database
from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table, create_engine, \
    DateTime, func

from app.constants import DATABASE_URL, POSTGRES_TABLE

logger = logging.getLogger()


# SQLAlchemy
engine = create_engine(DATABASE_URL)
metadata = MetaData()

predicts = Table(
    POSTGRES_TABLE,
    metadata,
    Column("id", Integer, primary_key=True),
    Column("text", String(length=1024 * 5)),
    Column("is_positive_user_answered", Boolean),
    Column("is_positive_model_answered", Boolean),
    Column("model_run_id", String(length=32)),
    Column("timestamp", DateTime, default=func.now(), nullable=False),
)

# databases query builder
database = Database(DATABASE_URL)
