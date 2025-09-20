from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Query

from views_challenge.data.data import (
    get_all_months,
    get_all_cells,
    get_all_countries,
    get_cells_with_filters
)
from views_challenge.data.models import CellsResponse


class ViolenceType(str, Enum):
    os = "os"
    ns = "ns"
    sb = "sb"


router = APIRouter()


@router.get("/months")
def get_available_months():
    """Get all available month IDs."""
    months = get_all_months()
    return {"months": months, "count": len(months)}


@router.get("/countries")
def get_available_countries():
    """Get all available country IDs."""
    countries = get_all_countries()
    return {"countries": countries, "count": len(countries)}


@router.get("/cells", response_model=CellsResponse)
def get_cells_by_filters(
    ids: Optional[List[int]] = Query(None, description="List of grid cell IDs"),
    month_range_start: Optional[int] = Query(None, description="Start month ID"),
    month_range_end: Optional[int] = Query(None, description="End month ID"),
    country_id: Optional[int] = Query(None, description="Country ID"),
    violence_types: Optional[List[ViolenceType]] = Query(None, description="Violence types to include"),
    limit: int = Query(10, description="Maximum number of cells to return")
):
    """
    Get grid cells with comprehensive forecast data.

    Default behavior (no query params):
    - Returns first 10 cells
    - Includes all violence types (sb, ns, os)
    - Includes all available months for those cells

    With query params: Filters as specified
    """

    # Set defaults when no filters provided
    if not any([ids, month_range_start, month_range_end, country_id]):
        # Get first N cells as default
        all_cells = get_all_cells()
        ids = all_cells[:limit]

    # Convert enum values to strings
    violence_type_strings = None
    if violence_types:
        violence_type_strings = [vt.value for vt in violence_types]

    return get_cells_with_filters(
        priogrid_ids=ids,
        month_range_start=month_range_start,
        month_range_end=month_range_end,
        country_id=country_id,
        violence_types=violence_type_strings
    )


@router.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        months = get_all_months()
        cells = get_all_cells()
        countries = get_all_countries()

        return {
            "status": "healthy",
            "data_loaded": True,
            "total_months": len(months),
            "total_cells": len(cells),
            "total_countries": len(countries)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "data_loaded": False,
            "error": str(e)
        }