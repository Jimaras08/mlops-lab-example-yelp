from app.api.models import PredictResponseSchema
from app.db import database, predicts


async def post(payload: PredictResponseSchema):
    query = predicts.insert().values(
        text=payload.text,
        is_positive_user_answered=payload.is_positive_user_answered,
        is_positive_model_answered=payload.is_positive_model_answered,
        mlflow_run_id=payload.mlflow_run_id,
    )
    return await database.execute(query=query)


async def get(id: int):
    query = predicts.select().where(id == predicts.c.id)
    return await database.fetch_one(query=query)


async def get_all():
    query = predicts.select()
    return await database.fetch_all(query=query)


async def get_correctness_rate():
    query_total = predicts.count()
    total = await database.fetch_val(query=query_total)
    if not total:
        return None
    query_total_correct = predicts.count().where(
        predicts.c.is_positive_model_answered == predicts.c.is_positive_user_answered
    )
    total_correct = await database.fetch_val(query=query_total_correct)
    return total_correct / total
