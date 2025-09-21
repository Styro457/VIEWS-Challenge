"""
Pydantic models for data.py calculations
"""


from typing import Tuple, List, Optional
from pydantic import BaseModel, Field


class ViolenceTypeForecast(BaseModel):
    """Forecast data for a specific violence type (sb, ns, or os)"""

    map_value: Optional[float] = Field(
        None, description="MAP (Maximum A Posteriori) estimate"
    )
    ci_50: Optional[Tuple[float, float]] = Field(
        None, description="50% confidence interval (lower, upper)"
    )
    ci_90: Optional[Tuple[float, float]] = Field(
        None, description="90% confidence interval (lower, upper)"
    )
    ci_99: Optional[Tuple[float, float]] = Field(
        None, description="99% confidence interval (lower, upper)"
    )
    prob_above_001: Optional[float] = Field(
        None, ge=0, le=1, description="Probability above threshold 0.01"
    )
    prob_above_005: Optional[float] = Field(
        None, ge=0, le=1, description="Probability above threshold 0.05"
    )
    prob_above_010: Optional[float] = Field(
        None, ge=0, le=1, description="Probability above threshold 0.10"
    )
    prob_above_025: Optional[float] = Field(
        None, ge=0, le=1, description="Probability above threshold 0.25"
    )
    prob_above_050: Optional[float] = Field(
        None, ge=0, le=1, description="Probability above threshold 0.50"
    )
    prob_above_080: Optional[float] = Field(
        None, ge=0, le=1, description="Probability above threshold 0.80"
    )

    model_config = {"extra": "forbid"}

    def is_empty(self) -> bool:
        """Check if all forecast values are None (empty object)"""
        return all(
            getattr(self, field) is None
            for field in [
                'map_value', 'ci_50', 'ci_90', 'ci_99',
                'prob_above_001', 'prob_above_005', 'prob_above_010',
                'prob_above_025', 'prob_above_050', 'prob_above_080'
            ]
        )


class MonthForecast(BaseModel):
    """Complete forecast data for all violence types for a specific month"""

    month_id: int = Field(..., description="Month identifier")

    # State-based conflict (government vs rebels)
    sb: Optional[ViolenceTypeForecast] = Field(
        None, description="State-based conflict forecast"
    )

    # Non-state conflict (organized groups fighting)
    ns: Optional[ViolenceTypeForecast] = Field(
        None, description="Non-state conflict forecast"
    )

    # One-sided violence (attacks on civilians)
    os: Optional[ViolenceTypeForecast] = Field(
        None, description="One-sided violence forecast"
    )


class Cell(BaseModel):
    """Grid cell with selective forecast data based on ReturnParameters"""

    priogrid_id: Optional[int] = Field(None, description="PRIO-GRID cell identifier")
    centroid_lat: Optional[float] = Field(None, description="Latitude of cell centroid")
    centroid_lon: Optional[float] = Field(
        None, description="Longitude of cell centroid"
    )
    country_id: Optional[int] = Field(None, description="UN M49 country identifier")
    country_name: Optional[str] = Field(None, description="Human-readable country name")
    months: List[MonthForecast] = Field(..., description="Monthly forecasts")

    model_config = {"extra": "forbid"}

    def model_dump(self, **kwargs):
        """Override to exclude None values from JSON output"""
        data = super().model_dump(**kwargs)
        return {k: v for k, v in data.items() if v is not None}


class CellsResponse(BaseModel):
    """API response containing multiple cells"""

    cells: List[Cell] = Field(..., description="List of grid cells with forecasts")
    count: int = Field(..., description="Number of cells returned")
    filters_applied: dict = Field(..., description="Summary of filters applied")