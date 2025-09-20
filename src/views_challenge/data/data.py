"""
Data processing functions using views_pipeline_core for VIEWS conflict forecasting.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from views_challenge.data.handler import PGMDataset

from .models import Cell, MonthForecast, ViolenceTypeForecast, CellsResponse
from views_challenge.utils.utils import decode_country


class ViewsDataProcessor:
    """Handles processing of VIEWS forecast data using views_pipeline_core."""

    def __init__(self, data_dir: str = "env"):
        """
        Initialize the data processor.

        Args:
            data_dir: Directory containing the parquet files
        """
        self.data_dir = Path(data_dir)
        self.raw_df = None
        self._load_raw_data()

    def _load_raw_data(self):
        """Load raw data without processing (fast startup)."""
        preds_file = self.data_dir / "preds_001.parquet"

        if not preds_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {preds_file}")

        print("Loading raw VIEWS data...")
        self.raw_df = pd.read_parquet(preds_file)
        print(f"Raw data loaded! Shape: {self.raw_df.shape}")

    def _apply_filters(
        self,
        df: pd.DataFrame,
        priogrid_ids: Optional[List[int]] = None,
        month_range_start: Optional[int] = None,
        month_range_end: Optional[int] = None,
        country_id: Optional[int] = None,
        country_name_field: bool = None,
    ) -> pd.DataFrame:
        """Apply filters to the dataframe (fast pandas operations)."""
        filtered_df = df.copy()

        # Apply filters
        if country_id is not None:
            filtered_df = filtered_df[filtered_df["country_id"] == country_id]

        if priogrid_ids is not None:
            grid_mask = filtered_df.index.get_level_values("priogrid_id").isin(
                priogrid_ids
            )
            filtered_df = filtered_df[grid_mask]

        if month_range_start is not None:
            month_mask = (
                filtered_df.index.get_level_values("month_id") >= month_range_start
            )
            filtered_df = filtered_df[month_mask]

        if month_range_end is not None:
            month_mask = (
                filtered_df.index.get_level_values("month_id") <= month_range_end
            )
            filtered_df = filtered_df[month_mask]

        return filtered_df

    def _compute_statistics_for_filtered_data(
        self,
        filtered_df: pd.DataFrame,
        map_value: bool = True,
        ci_50: bool = False,
        ci_90: bool = False,
        ci_99: bool = False,
    ) -> pd.DataFrame:
        """Compute statistics only for the filtered subset (efficient)."""
        if len(filtered_df) == 0:
            return filtered_df

        print(f"Computing statistics for {len(filtered_df)} rows...")

        # Separate predictions from metadata
        pred_cols = [col for col in filtered_df.columns if col.startswith("pred_")]
        meta_cols = [col for col in filtered_df.columns if not col.startswith("pred_")]

        predictions_df = filtered_df[pred_cols]
        metadata_df = filtered_df[meta_cols]

        # Create PGMDataset for VIEWS calculations on filtered data only
        dataset = PGMDataset(source=predictions_df)

        # Combine everything into comprehensive dataframe
        comprehensive_df = metadata_df.copy()

        # Conditionally calculate statistics based on requested parameters
        if map_value:
            print("  Calculating MAP estimates...")
            map_estimates = dataset.calculate_map()
            for col in map_estimates.columns:
                comprehensive_df[col] = map_estimates[col]

        # Calculate confidence intervals only if requested
        confidence_levels = []
        if ci_50:
            confidence_levels.append((0.5, "50"))
        if ci_90:
            confidence_levels.append((0.9, "90"))
        if ci_99:
            confidence_levels.append((0.99, "99"))

        if confidence_levels:
            print("  Calculating confidence intervals...")
            for alpha, pct_str in confidence_levels:
                hdi_df = dataset.calculate_hdi(alpha=alpha)
                for col in hdi_df.columns:
                    new_col_name = col.replace("_hdi_", f"_hdi_{pct_str}_")
                    comprehensive_df[new_col_name] = hdi_df[col]

        print("  Statistics computation complete!")
        return comprehensive_df

    def _extract_violence_type_forecast(
        self,
        row: pd.Series,
        violence_type: str,
        month_id: int,
        map_value: bool = True,
        ci_50: bool = False,
        ci_90: bool = False,
        ci_99: bool = False,
        include_prob_thresholds: bool = True,
    ) -> ViolenceTypeForecast:
        """
        Extract forecast data for a specific violence type from a dataframe row.

        Args:
            row: Row from comprehensive dataframe
            violence_type: One of 'sb', 'ns', 'os'
            month_id: Month identifier
            map_value: Include MAP estimates
            ci_50: Include 50% confidence intervals
            ci_90: Include 90% confidence intervals
            ci_99: Include 99% confidence intervals

        Returns:
            ViolenceTypeForecast with only requested values
        """
        pred_col = f"pred_ln_{violence_type}_best"

        # Extract values based on what was computed
        extracted_map_value = None
        if map_value:
            map_col = f"{pred_col}_map"
            if map_col in row and pd.notna(row[map_col]):
                extracted_map_value = float(row[map_col])

        # Confidence intervals (only if computed)
        extracted_ci_50 = None
        if ci_50:
            ci_50_lower_col = f"{pred_col}_hdi_50_lower"
            ci_50_upper_col = f"{pred_col}_hdi_50_upper"
            if ci_50_lower_col in row and ci_50_upper_col in row:
                lower = (
                    float(row[ci_50_lower_col])
                    if pd.notna(row[ci_50_lower_col])
                    else 0.0
                )
                upper = (
                    float(row[ci_50_upper_col])
                    if pd.notna(row[ci_50_upper_col])
                    else 0.0
                )
                extracted_ci_50 = (lower, upper)

        extracted_ci_90 = None
        if ci_90:
            ci_90_lower_col = f"{pred_col}_hdi_90_lower"
            ci_90_upper_col = f"{pred_col}_hdi_90_upper"
            if ci_90_lower_col in row and ci_90_upper_col in row:
                lower = (
                    float(row[ci_90_lower_col])
                    if pd.notna(row[ci_90_lower_col])
                    else 0.0
                )
                upper = (
                    float(row[ci_90_upper_col])
                    if pd.notna(row[ci_90_upper_col])
                    else 0.0
                )
                extracted_ci_90 = (lower, upper)

        extracted_ci_99 = None
        if ci_99:
            ci_99_lower_col = f"{pred_col}_hdi_99_lower"
            ci_99_upper_col = f"{pred_col}_hdi_99_upper"
            if ci_99_lower_col in row and ci_99_upper_col in row:
                lower = (
                    float(row[ci_99_lower_col])
                    if pd.notna(row[ci_99_lower_col])
                    else 0.0
                )
                upper = (
                    float(row[ci_99_upper_col])
                    if pd.notna(row[ci_99_upper_col])
                    else 0.0
                )
                extracted_ci_99 = (lower, upper)

        # TODO: Add actual probability threshold calculations
        # For now, using placeholder values only if requested
        if include_prob_thresholds:
            prob_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        else:
            prob_thresholds = [None] * 6

        # Build forecast object with only requested fields
        forecast_data = {}

        if map_value and extracted_map_value is not None:
            forecast_data["map_value"] = extracted_map_value

        if ci_50 and extracted_ci_50 is not None:
            forecast_data["ci_50"] = extracted_ci_50

        if ci_90 and extracted_ci_90 is not None:
            forecast_data["ci_90"] = extracted_ci_90

        if ci_99 and extracted_ci_99 is not None:
            forecast_data["ci_99"] = extracted_ci_99

        if include_prob_thresholds:
            if prob_thresholds[0] is not None:
                forecast_data["prob_above_10"] = prob_thresholds[0]
            if prob_thresholds[1] is not None:
                forecast_data["prob_above_20"] = prob_thresholds[1]
            if prob_thresholds[2] is not None:
                forecast_data["prob_above_30"] = prob_thresholds[2]
            if prob_thresholds[3] is not None:
                forecast_data["prob_above_40"] = prob_thresholds[3]
            if prob_thresholds[4] is not None:
                forecast_data["prob_above_50"] = prob_thresholds[4]
            if prob_thresholds[5] is not None:
                forecast_data["prob_above_60"] = prob_thresholds[5]

        return ViolenceTypeForecast(**forecast_data)

    def get_filtered_cells(
        self,
        priogrid_ids: Optional[List[int]] = None,
        month_range_start: Optional[int] = None,
        month_range_end: Optional[int] = None,
        country_id: Optional[int] = None,
        violence_types: Optional[List[str]] = None,
        include_grid_id: bool = True,
        include_lat_lon: bool = True,
        include_country_id: bool = True,
        include_country_name: bool = True,
        map_value: bool = True,
        ci_50: bool = False,
        ci_90: bool = False,
        ci_99: bool = False,
        include_prob_thresholds: bool = True,
        limit: int = None,
        offset: int = None
    ) -> CellsResponse:
        """
        Get filtered cell data with comprehensive forecasts.

        EFFICIENT APPROACH: Filter first, then compute statistics
        only for filtered data.

        Args:
            priogrid_ids: List of grid cell IDs to include
            month_range_start: Start month ID
            month_range_end: End month ID
            country_id: Country ID to filter by
            violence_types: List of violence types to include ('sb', 'ns', 'os')

        Returns:
            CellsResponse with filtered and processed data
        """
        # Step 1: Apply filters to raw data (FAST - just pandas filtering)
        filtered_df = self._apply_filters(
            self.raw_df,
            priogrid_ids=priogrid_ids,
            month_range_start=month_range_start,
            month_range_end=month_range_end,
            country_id=country_id,
        )

        # Step 2: Apply limit and offset to the filtered values
        selected_ids = filtered_df.index.get_level_values("priogrid_id").unique()
        if offset:
            selected_ids = selected_ids[offset:]
        if limit:
            selected_ids = selected_ids[:limit]
        filtered_df = filtered_df[filtered_df.index.get_level_values("priogrid_id").isin(selected_ids)]

        if len(filtered_df) == 0:
            return CellsResponse(cells=[], count=0, filters_applied={})

        # Step 2: Compute statistics only for filtered data (EFFICIENT)
        df = self._compute_statistics_for_filtered_data(
            filtered_df, map_value=map_value, ci_50=ci_50, ci_90=ci_90, ci_99=ci_99
        )

        # Default to all violence types if none specified
        if violence_types is None:
            violence_types = ["sb", "ns", "os"]

        # Group by grid cell
        cells = []
        for priogrid_id in selected_ids:
            cell_data = df[filtered_df.index.get_level_values("priogrid_id") == priogrid_id]

            # Get cell metadata from first row
            first_row = cell_data.iloc[0]

            # Group by month for this cell
            months = []
            for month_id in cell_data.index.get_level_values("month_id").unique():
                month_row = cell_data[
                    cell_data.index.get_level_values("month_id") == month_id
                ].iloc[0]

                # Create violence type forecasts based on requested types
                violence_forecasts = {}
                for vtype in violence_types:
                    forecast = self._extract_violence_type_forecast(
                        month_row,
                        vtype,
                        month_id,
                        map_value=map_value,
                        ci_50=ci_50,
                        ci_90=ci_90,
                        ci_99=ci_99,
                        include_prob_thresholds=include_prob_thresholds,
                    )
                    # Only include violence types that have actual data
                    forecast_dict = forecast.model_dump()
                    if forecast_dict:  # If the dumped dict is not empty
                        violence_forecasts[vtype] = forecast

                month_forecast = MonthForecast(
                    month_id=month_id,
                    sb=violence_forecasts.get("sb"),
                    ns=violence_forecasts.get("ns"),
                    os=violence_forecasts.get("os"),
                )
                months.append(month_forecast)

            # Sort months by month_id
            months.sort(key=lambda x: x.month_id)

            country_id_val = int(first_row["country_id"])
            country_name = decode_country(country_id_val)

            # Build cell with only requested fields
            cell_data = {"months": months}

            if include_grid_id:
                cell_data["priogrid_id"] = priogrid_id
            if include_lat_lon:
                cell_data["centroid_lat"] = float(first_row["lat"])
                cell_data["centroid_lon"] = float(first_row["lon"])
            if include_country_id:
                cell_data["country_id"] = country_id_val
            if include_country_name:
                cell_data["country_name"] = country_name

            cell = Cell(**cell_data)
            cells.append(cell)

        # Create response
        filters_applied = {
            "priogrid_ids": priogrid_ids,
            "month_range_start": month_range_start,
            "month_range_end": month_range_end,
            "country_id": country_id,
            "violence_types": violence_types,
        }

        return CellsResponse(
            cells=cells, count=len(cells), filters_applied=filters_applied
        )

    def get_available_months(self) -> List[int]:
        """Get list of available month IDs."""
        return sorted(self.raw_df.index.get_level_values("month_id").unique().tolist())

    def get_available_cells(self) -> List[int]:
        """Get list of available grid cell IDs."""
        return sorted(
            self.raw_df.index.get_level_values("priogrid_id").unique().tolist()
        )

    def get_available_countries(self) -> List[int]:
        """Get list of available country IDs."""
        return sorted(self.raw_df["country_id"].unique().tolist())


# Global instance (singleton pattern)
_data_processor: Optional[ViewsDataProcessor] = None


def get_data_processor() -> ViewsDataProcessor:
    """Get the global data processor instance."""
    global _data_processor
    if _data_processor is None:
        _data_processor = ViewsDataProcessor()
    return _data_processor


# Convenience functions for API endpoints
def get_cells_with_filters(
    priogrid_ids: Optional[List[int]] = None,
    month_range_start: Optional[int] = None,
    month_range_end: Optional[int] = None,
    country_id: Optional[int] = None,
    violence_types: Optional[List[str]] = None,
    include_grid_id: bool = True,
    include_lat_lon: bool = True,
    include_country_id: bool = True,
    country_name_field: bool = True,
    map_value: bool = True,
    ci_50: bool = True,
    ci_90: bool = True,
    ci_99: bool = True,
    include_prob_thresholds: bool = True,
    limit: int = 10,
    offset: int = None
) -> CellsResponse:
    """Get filtered cells using the global data processor."""
    processor = get_data_processor()
    return processor.get_filtered_cells(
        priogrid_ids=priogrid_ids,
        month_range_start=month_range_start,
        month_range_end=month_range_end,
        country_id=country_id,
        violence_types=violence_types,
        include_grid_id=include_grid_id,
        include_lat_lon=include_lat_lon,
        include_country_id=include_country_id,
        include_country_name=country_name_field,
        map_value=map_value,
        ci_50=ci_50,
        ci_90=ci_90,
        ci_99=ci_99,
        include_prob_thresholds=include_prob_thresholds,
        limit=limit,
        offset=offset,
    )


def get_all_months() -> List[int]:
    """Get all available months."""
    processor = get_data_processor()
    return processor.get_available_months()


def get_all_cells() -> List[int]:
    """Get all available cells."""
    processor = get_data_processor()
    return processor.get_available_cells()


def get_all_countries() -> List[int]:
    """Get all available countries."""
    processor = get_data_processor()
    return processor.get_available_countries()
