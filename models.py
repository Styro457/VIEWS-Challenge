from typing import Tuple, List

from pydantic import BaseModel, Field


class MonthForecast(BaseModel):
    map_value: float = Field(..., description="Mean value (MAP)")
    ci_50: Tuple[float, float] = Field(..., description="50% confidence interval (lower, upper)")
    ci_90: Tuple[float, float] = Field(..., description="90% confidence interval (lower, upper)")
    ci_99: Tuple[float, float] = Field(..., description="99% confidence interval (lower, upper)")
    prob_above_10: float = Field(..., ge=0, le=1)
    prob_above_20: float = Field(..., ge=0, le=1)
    prob_above_30: float = Field(..., ge=0, le=1)
    prob_above_40: float = Field(..., ge=0, le=1)
    prob_above_50: float = Field(..., ge=0, le=1)
    prob_above_60: float = Field(..., ge=0, le=1)


class Cell(BaseModel):
    """Cell Model"""
    id: int = Field(...)
    centroid_lat: float = Field(...)
    centroid_lon: float = Field(...)
    country_id: int = Field(...)
    months: List[MonthForecast] = Field(...)
