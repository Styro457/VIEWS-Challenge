from enum import Enum
from typing import List

import pandas
from fastapi import APIRouter, Query


class ViolenceType(str, Enum):
    os = "os",
    ns = "ns",
    sb = "sb"


filepath = "./preds_001.parquet"

router = APIRouter()


# @router.get(path="/all_months")
# def get_all_months():
#     """Returns all available months"""
#     return "All Months!"
#
#
# @router.get(path="/all_cells")
# def get_all_cells():
#     """Returns all available cells"""
#     return "All Cells!"


def filter_file(df, priogrid_ids, month_range_start, month_range_end, country_id):
    """
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
                 month_range_end: int = None, country_id: int = None, violence_type: ViolenceType = Query(None)):
    """Returns all cells that match the criteria passed as query parameters"""
    print(
        f"IDs: {ids}\nMonth Range:{month_range_start} - {month_range_end}\nCountry: {country_id}\nViolence Type: {violence_type}")
    df = pandas.read_parquet(filepath, engine="pyarrow")
    df = df.reset_index()
    filtered_df = filter_file(df, priogrid_ids=ids, month_range_start=month_range_start,
                              month_range_end=month_range_end,
                              country_id=country_id)
    print(filtered_df)