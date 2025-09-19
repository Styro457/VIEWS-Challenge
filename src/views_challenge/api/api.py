from typing import List

from fastapi import APIRouter, Query

router = APIRouter()


@router.get(path="/all_months")
def get_all_months():
    """Returns all available months"""
    return "All Months!"


@router.get(path="/all_cells")
def get_all_cells():
    """Returns all available cells"""
    return "All Cells!"


# ... = required
@router.get(path="/cells")
def get_cells_by(
    ids: List[int] = Query(None, description="List of Cells IDs"),
    month_range_start: int = None,
    month_range_end: int = None,
    country: str = None,
):
    """Returns all cells that match the criteria passed as query parameters"""
    return (f"IDs: {ids}\nMonth Range:{month_range_start} -"
            f" {month_range_end}\nCountry: {country}")
