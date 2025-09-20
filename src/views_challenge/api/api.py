import json
import os
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
    
country_list_path = "./m49-list.json"
# pandas reads the file relative to location from where the program is executed
parquet_filepath = os.path.join(os.path.dirname(__file__), "preds_001.parquet")


class ReturnParameters(str, Enum):
    grid_id = "grid_id"
    lat_lon = "lat_lon"
    country_id = "country_id"
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


def decode_country(country_id):
    """Returns country name based on M49 ID"""
    with open(country_list_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        countries_list = data["countries"]
        for country in countries_list:
            if int(country["m49code"]) == country_id:
                return country["name"]
        return None


def decode_month(month_id):
    return


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

@router.get(path="/all_cells")
def get_all_cells():
    """Get all available cell IDs."""
    cells = get_all_cells()
    return {"countries": cells, "count": len(cells)}

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
    Applies filters to the dataframe
    kwargs - filtering parameters e.g. country_id"""

    filtered_df = df.copy()
    if priogrid_ids:
        filtered_df = filtered_df[filtered_df["priogrid_id"].isin(priogrid_ids)]
    if month_range_end and month_range_start:
        filtered_df = filtered_df[
            (filtered_df["month_id"] >= month_range_start) & (filtered_df["month_id"] <= month_range_end)]
    if country_id:
        filtered_df = filtered_df[filtered_df["country_id"] == country_id]
    return filtered_df


@router.get(path="/cells")
def get_cells_by(ids: List[int] = Query(None, description="List of Cells IDs"), month_range_start: int = None,
                 month_range_end: int = None, country_id: int = None, violence_type: ViolenceType = Query(None),
                 return_param_selection: List[ReturnParameters] = Query(...)):
    """Returns all cells that match the criteria passed as query parameters"""
    print(
        f"IDs: {ids}\nMonth Range:{month_range_start} - {month_range_end}\nCountry: {country_id}\nViolence Type: {violence_type}\nParams requested: {return_param_selection}")
    df = pandas.read_parquet(parquet_filepath, engine="pyarrow")
    df = df.reset_index()
    filtered_df = filter_file(df, priogrid_ids=ids, month_range_start=month_range_start,
                              month_range_end=month_range_end,
                              country_id=country_id)

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
