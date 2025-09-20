import json
import os
from enum import Enum
from typing import List

import pandas
from fastapi import APIRouter, Query

country_list_path = "./m49-list.json"
# pandas reads the file relative to location from where the program is executed
parquet_filepath = os.path.join(os.path.dirname(__file__), "preds_001.parquet")
print(parquet_filepath)


class ViolenceType(str, Enum):
    os = "os"
    ns = "ns"
    sb = "sb"


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


@router.get(path="/all_months")
def get_all_months():
    """Returns all available months"""
    df = pandas.read_parquet(parquet_filepath, engine="pyarrow")
    df = df.reset_index()
    unique_months = df['month_id'].unique()
    return unique_months.tolist()


@router.get(path="/all_cells")
def get_all_cells():
    """Returns all available cells"""
    df = pandas.read_parquet(parquet_filepath, engine="pyarrow")
    df = df.reset_index()
    unique_cells = df['priogrid_id'].unique()
    return unique_cells.tolist()


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
