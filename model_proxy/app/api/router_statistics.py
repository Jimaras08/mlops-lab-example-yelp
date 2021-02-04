from fastapi import APIRouter

from app.api import crud

router = APIRouter()


@router.get("")
async def get_statistics():
    rate = await crud.get_correctness_rate()
    response_object = {
        "statistics": {
            "correctness_rate": rate,
        }
    }
    return response_object
