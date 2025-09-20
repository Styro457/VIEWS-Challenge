"""
Data processing functions using views_pipeline_core for VIEWS conflict forecasting.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from views_pipeline_core.data.handlers import PGMDataset

from .models import Cell, MonthForecast, ViolenceTypeForecast, CellsResponse


class ViewsDataProcessor:
    """Handles processing of VIEWS forecast data using views_pipeline_core."""

    def __init__(self, data_dir: str = "hack_data"):
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
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> pd.DataFrame:
        """Apply filters to the dataframe (fast pandas operations)."""
        filtered_df = df.copy()

        # Apply filters
        if country_id is not None:
            filtered_df = filtered_df[filtered_df['country_id'] == country_id]

        if priogrid_ids is not None:
            grid_mask = filtered_df.index.get_level_values('priogrid_id').isin(priogrid_ids)
            filtered_df = filtered_df[grid_mask]

        if month_range_start is not None:
            month_mask = filtered_df.index.get_level_values('month_id') >= month_range_start
            filtered_df = filtered_df[month_mask]

        if month_range_end is not None:
            month_mask = filtered_df.index.get_level_values('month_id') <= month_range_end
            filtered_df = filtered_df[month_mask]

        # Apply offset and limit
        if offset:
            filtered_df = filtered_df.iloc[offset:]
        if limit is not None:
            filtered_df = filtered_df.iloc[:limit]

        return filtered_df

    def _compute_statistics_for_filtered_data(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistics only for the filtered subset (efficient)."""
        if len(filtered_df) == 0:
            return filtered_df

        print(f"Computing statistics for {len(filtered_df)} rows...")

        # Separate predictions from metadata
        pred_cols = [col for col in filtered_df.columns if col.startswith('pred_')]
        meta_cols = [col for col in filtered_df.columns if not col.startswith('pred_')]

        predictions_df = filtered_df[pred_cols]
        metadata_df = filtered_df[meta_cols]

        # Create PGMDataset for VIEWS calculations on filtered data only
        dataset = PGMDataset(source=predictions_df)

        # Calculate all statistics for filtered data
        print("  Calculating MAP estimates...")
        map_estimates = dataset.calculate_map()

        print("  Calculating confidence intervals...")
        hdi_50 = dataset.calculate_hdi(alpha=0.5)
        hdi_90 = dataset.calculate_hdi(alpha=0.9)
        hdi_99 = dataset.calculate_hdi(alpha=0.99)

        # Combine everything into comprehensive dataframe
        comprehensive_df = metadata_df.copy()

        # Add MAP estimates
        for col in map_estimates.columns:
            comprehensive_df[col] = map_estimates[col]

        # Add HDI bounds for each confidence level
        for alpha, hdi_df in [(0.5, hdi_50), (0.9, hdi_90), (0.99, hdi_99)]:
            confidence_pct = int(alpha * 100)
            for col in hdi_df.columns:
                new_col_name = col.replace('_hdi_', f'_hdi_{confidence_pct}_')
                comprehensive_df[new_col_name] = hdi_df[col]

        print("  Statistics computation complete!")
        return comprehensive_df

    def _extract_violence_type_forecast(
        self,
        row: pd.Series,
        violence_type: str,
        month_id: int
    ) -> ViolenceTypeForecast:
        """
        Extract forecast data for a specific violence type from a dataframe row.

        Args:
            row: Row from comprehensive dataframe
            violence_type: One of 'sb', 'ns', 'os'
            month_id: Month identifier

        Returns:
            ViolenceTypeForecast with all 13 values
        """
        pred_col = f"pred_ln_{violence_type}_best"

        # MAP estimate
        map_col = f"{pred_col}_map"
        map_value = float(row[map_col]) if pd.notna(row[map_col]) else 0.0

        # Confidence intervals
        ci_50_lower = float(row[f"{pred_col}_hdi_50_lower"]) if f"{pred_col}_hdi_50_lower" in row else 0.0
        ci_50_upper = float(row[f"{pred_col}_hdi_50_upper"]) if f"{pred_col}_hdi_50_upper" in row else 0.0

        ci_90_lower = float(row[f"{pred_col}_hdi_90_lower"]) if f"{pred_col}_hdi_90_lower" in row else 0.0
        ci_90_upper = float(row[f"{pred_col}_hdi_90_upper"]) if f"{pred_col}_hdi_90_upper" in row else 0.0

        ci_99_lower = float(row[f"{pred_col}_hdi_99_lower"]) if f"{pred_col}_hdi_99_lower" in row else 0.0
        ci_99_upper = float(row[f"{pred_col}_hdi_99_upper"]) if f"{pred_col}_hdi_99_upper" in row else 0.0

        # TODO: Add actual probability threshold calculations
        # For now, using placeholder values
        prob_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        return ViolenceTypeForecast(
            map_value=map_value,
            ci_50=(ci_50_lower, ci_50_upper),
            ci_90=(ci_90_lower, ci_90_upper),
            ci_99=(ci_99_lower, ci_99_upper),
            prob_above_10=prob_thresholds[0],
            prob_above_20=prob_thresholds[1],
            prob_above_30=prob_thresholds[2],
            prob_above_40=prob_thresholds[3],
            prob_above_50=prob_thresholds[4],
            prob_above_60=prob_thresholds[5]
        )

    def get_filtered_cells(
        self,
        priogrid_ids: Optional[List[int]] = None,
        month_range_start: Optional[int] = None,
        month_range_end: Optional[int] = None,
        country_id: Optional[int] = None,
        violence_types: Optional[List[str]] = None,
        limit: Optional[int] = 1000,
        offset: Optional[int] = 0,
    ) -> CellsResponse:
        """
        Get filtered cell data with comprehensive forecasts.

        EFFICIENT APPROACH: Filter first, then compute statistics only for filtered data.

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
            limit=limit,
            offset=offset
        )

        if len(filtered_df) == 0:
            return CellsResponse(cells=[], count=0, filters_applied={})

        # Step 2: Compute statistics only for filtered data (EFFICIENT)
        df = self._compute_statistics_for_filtered_data(filtered_df)

        # Default to all violence types if none specified
        if violence_types is None:
            violence_types = ['sb', 'ns', 'os']

        # Group by grid cell
        cells = []
        for priogrid_id in df.index.get_level_values('priogrid_id').unique():
            cell_data = df[df.index.get_level_values('priogrid_id') == priogrid_id]

            # Get cell metadata from first row
            first_row = cell_data.iloc[0]

            # Group by month for this cell
            months = []
            for month_id in cell_data.index.get_level_values('month_id').unique():
                month_row = cell_data[cell_data.index.get_level_values('month_id') == month_id].iloc[0]

                # Create violence type forecasts based on requested types
                violence_forecasts = {}
                for vtype in violence_types:
                    violence_forecasts[vtype] = self._extract_violence_type_forecast(
                        month_row, vtype, month_id
                    )

                month_forecast = MonthForecast(
                    month_id=month_id,
                    sb=violence_forecasts.get('sb'),
                    ns=violence_forecasts.get('ns'),
                    os=violence_forecasts.get('os')
                )
                months.append(month_forecast)

            # Sort months by month_id
            months.sort(key=lambda x: x.month_id)

            cell = Cell(
                priogrid_id=priogrid_id,
                centroid_lat=float(first_row['lat']),
                centroid_lon=float(first_row['lon']),
                country_id=int(first_row['country_id']),
                months=months
            )
            cells.append(cell)

        # Create response
        filters_applied = {
            'priogrid_ids': priogrid_ids,
            'month_range_start': month_range_start,
            'month_range_end': month_range_end,
            'country_id': country_id,
            'violence_types': violence_types
        }

        return CellsResponse(
            cells=cells,
            count=len(cells),
            filters_applied=filters_applied
        )

    def get_available_months(self) -> List[int]:
        """Get list of available month IDs."""
        return sorted(
            self.raw_df.index.get_level_values('month_id').unique().tolist()
        )

    def get_available_cells(self) -> List[int]:
        """Get list of available grid cell IDs."""
        return sorted(
            self.raw_df.index.get_level_values('priogrid_id').unique().tolist()
        )

    def get_available_countries(self) -> List[int]:
        """Get list of available country IDs."""
        return sorted(self.raw_df['country_id'].unique().tolist())


# Global instance (singleton pattern)
_data_processor: Optional[ViewsDataProcessor] = None


def get_data_processor() -> ViewsDataProcessor:
    """Get the global data processor instance."""
    global _data_processor
    if _data_processor is None:
        _data_processor = ViewsDataProcessor()
    return _data_processor


# Convenience functions for API endpoints
def get_cells_with_filters(**kwargs) -> CellsResponse:
    """Get filtered cells using the global data processor."""
    processor = get_data_processor()
    return processor.get_filtered_cells(**kwargs)


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