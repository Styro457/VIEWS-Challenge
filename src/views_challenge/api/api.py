from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Query
from starlette.responses import JSONResponse

from views_challenge.data.data import (
    get_all_months,
    get_all_cells,
    get_all_countries,
    get_cells_with_filters,
)
from views_challenge.data.models import CellsResponse


class ViolenceType(str, Enum):
    os = "os"
    ns = "ns"
    sb = "sb"


class ReturnParameters(str, Enum):
    grid_id = "grid_id"
    lat_lon = "lat_lon"
    country_id = "country_id"
    country_name = "country_name"
    map_value = "map_value"
    ci_50 = "ci_50"
    ci_90 = "ci_90"
    ci_99 = "ci_99"
    prob_above_10 = "prob_above_10"
    prob_above_20 = "prob_above_20"
    prob_above_30 = "prob_above_30"
    prob_above_40 = "prob_above_40"
    prob_above_50 = "prob_above_50"
    prob_above_60 = "prob_above_60"


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


@router.get("/all_cells")
def get_all_cells_endpoint():
    """Get all available cell IDs."""
    cells = get_all_cells()
    return {"cells": cells, "count": len(cells)}


@router.get("/cells", response_model=CellsResponse)
def get_cells_by_filters(
    ids: Optional[List[int]] = Query(None, description="List of grid cell IDs"),
    month_range_start: Optional[int] = Query(None, description="Start month ID"),
    month_range_end: Optional[int] = Query(None, description="End month ID"),
    country_id: Optional[int] = Query(None, description="Country ID"),
    violence_types: Optional[List[ViolenceType]] = Query(
        None, description="Violence types to include"
    ),
    limit: int = Query(10, description="Maximum number of cells to return"),
    offset: Optional[int] = Query(
        None, description="Offset from where to start returning cells"
    ),
    return_params: Optional[List[ReturnParameters]] = Query(
        None,
        description="Specify which data fields to return"
                    " (e.g., map_value, ci_90, ci_99)",
    ),
):
    """
    Get grid cells with selective forecast data based on ReturnParameters.

    Default behavior (no query params):
    - Returns first 10 cells
    - Includes all violence types (sb, ns, os)
    - Includes all available months for those cells
    - Only computes basic cell info and MAP estimates

    ReturnParameters control what data is computed and returned:
    - grid_id, lat_lon, country_id: Basic cell information (always included)
    - map_value: MAP estimates
    - ci_50, ci_90, ci_99: Confidence intervals
    - prob_above_10 through prob_above_60: Probability thresholds

    Examples:
    - /cells?return_params=map_value,ci_90
    - /cells?ids=123&return_params=map_value,ci_50,ci_99
    """

    # Set defaults when no filters provided
    # if not any([ids, month_range_start, month_range_end, country_id]):
    #     # Get first N cells as default
    #     all_cells = get_all_cells()
    #     ids = all_cells[:limit]

    # Convert enum values to strings
    violence_type_strings = None
    if violence_types:
        violence_type_strings = [vt.value for vt in violence_types]

    # Convert ReturnParameters to field inclusion flags
    if return_params is None:
        # Default: include all fields
        grid_id = lat_lon = country_id_field = country_name_field = True
        map_value = ci_50 = ci_90 = ci_99 = True
        prob_thresholds = True
    else:
        # Selective: include essentials + specified fields
        grid_id = True  # Always include when return_params specified
        lat_lon = ReturnParameters.lat_lon in return_params
        country_id_field = True  # Always include when return_params specified
        country_name_field = ReturnParameters.country_name in return_params
        map_value = ReturnParameters.map_value in return_params
        ci_50 = ReturnParameters.ci_50 in return_params
        ci_90 = ReturnParameters.ci_90 in return_params
        ci_99 = ReturnParameters.ci_99 in return_params
        prob_thresholds = any(
            getattr(ReturnParameters, f"prob_above_{i}", None) in return_params
            for i in [10, 20, 30, 40, 50, 60]
        )

    # Note: grid_id, month_id are always
    # included when return_params specified

    # Note: Probability thresholds are included if any
    # statistical computation is requested

    filtered_cells = get_cells_with_filters(
        priogrid_ids=ids,
        month_range_start=month_range_start,
        month_range_end=month_range_end,
        country_id=country_id,
        country_name_field=country_name_field,
        violence_types=violence_type_strings,
        include_grid_id=grid_id,
        include_lat_lon=lat_lon,
        include_country_id=country_id_field,
        map_value=map_value,
        ci_50=ci_50,
        ci_90=ci_90,
        ci_99=ci_99,
        include_prob_thresholds=prob_thresholds,
        limit=limit,
        offset=offset
    )

    response = JSONResponse(content=filtered_cells.model_dump(exclude_none=True))
    return response


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
            "total_countries": len(countries),
        }
    except Exception as e:
        return {"status": "unhealthy", "data_loaded": False, "error": str(e)}
