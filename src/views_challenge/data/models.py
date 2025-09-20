from typing import Tuple, List, Optional
from pydantic import BaseModel, Field


class ViolenceTypeForecast(BaseModel):
    """Forecast data for a specific violence type (sb, ns, or os)"""
    map_value: float = Field(..., description="MAP (Maximum A Posteriori) estimate")
    ci_50: Tuple[float, float] = Field(..., description="50% confidence interval (lower, upper)")
    ci_90: Tuple[float, float] = Field(..., description="90% confidence interval (lower, upper)")
    ci_99: Tuple[float, float] = Field(..., description="99% confidence interval (lower, upper)")
    prob_above_10: float = Field(..., ge=0, le=1, description="Probability above threshold 10")
    prob_above_20: float = Field(..., ge=0, le=1, description="Probability above threshold 20")
    prob_above_30: float = Field(..., ge=0, le=1, description="Probability above threshold 30")
    prob_above_40: float = Field(..., ge=0, le=1, description="Probability above threshold 40")
    prob_above_50: float = Field(..., ge=0, le=1, description="Probability above threshold 50")
    prob_above_60: float = Field(..., ge=0, le=1, description="Probability above threshold 60")


class MonthForecast(BaseModel):
    """Complete forecast data for all violence types for a specific month"""
    month_id: int = Field(..., description="Month identifier")

    # State-based conflict (government vs rebels)
    sb: Optional[ViolenceTypeForecast] = Field(None, description="State-based conflict forecast")

    # Non-state conflict (organized groups fighting)
    ns: Optional[ViolenceTypeForecast] = Field(None, description="Non-state conflict forecast")

    # One-sided violence (attacks on civilians)
    os: Optional[ViolenceTypeForecast] = Field(None, description="One-sided violence forecast")


class Cell(BaseModel):
    """Grid cell with complete forecast data"""
    priogrid_id: int = Field(..., description="PRIO-GRID cell identifier")
    centroid_lat: float = Field(..., description="Latitude of cell centroid")
    centroid_lon: float = Field(..., description="Longitude of cell centroid")
    country_id: int = Field(..., description="UN M49 country identifier")
    months: List[MonthForecast] = Field(..., description="Monthly forecasts")


class CellsResponse(BaseModel):
    """API response containing multiple cells"""
    cells: List[Cell] = Field(..., description="List of grid cells with forecasts")
    count: int = Field(..., description="Number of cells returned")
    filters_applied: dict = Field(..., description="Summary of filters applied")