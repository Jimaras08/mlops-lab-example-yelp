from app.api.models import PredictRequestSchema,PredictResponseSchema
from app.db import database, predicts


async def post(payload: PredictResponseSchema):
    query = predicts.insert().values(
        text=payload.text,
        is_positive_user_answered=payload.is_positive_user_answered,
        is_positive_model_answered=payload.is_positive_model_answered,
        model_run_id=payload.model_run_id,
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
        return 0.0
    query_total_correct = predicts.count().where(
        # predicts.c.model_output_is_positive ==
        predicts.c.ground_truth_is_positive
    )
    total_correct = await database.fetch_val(query=query_total_correct)
    return total_correct / total
