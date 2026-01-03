"""
Projection Model for Nuclear Energy Generation to 2050

This module projects nuclear energy generation by region from current year to 2050
based on the nuclear pipeline (operating, construction, announced plants).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class NuclearProjectionModel:
    """Project nuclear energy generation to 2050"""

    # Capacity factors by status
    CAPACITY_FACTORS = {
        'operating': 0.90,
        'construction': 0.85,
        'pre-construction': 0.85,
        'announced': 0.85,
        'default': 0.85
    }

    # Default plant lifetime (years)
    DEFAULT_LIFETIME = 60

    def __init__(self, data_path: str = None):
        """
        Initialize projection model

        Args:
            data_path: Path to cleaned nuclear tracker CSV
        """
        if data_path is None:
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "processed" / "nuclear_tracker_cleaned.csv"
        else:
            self.data_path = Path(data_path)

        self.df = None
        self.projections = None
        self.current_year = datetime.now().year

    def load_data(self):
        """Load cleaned nuclear tracker data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f" Loaded data: {len(self.df):,} rows")
            return self.df
        except FileNotFoundError:
            print(f" Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f" Error loading data: {e}")
            return None

    def prepare_projection_data(self):
        """
        Prepare data for projection by extracting start years and retirement years

        Returns:
            DataFrame with cleaned dates and years
        """
        if self.df is None:
            print("  No data loaded.")
            return None


        # Clean capacity
        self.df['capacity_numeric'] = pd.to_numeric(self.df['capacity_numeric'], errors='coerce')

        # Extract start year (when plant becomes operational)
        # Priority: Start Year > Commercial Operation Date > First Grid Connection
        self.df['start_year_extracted'] = pd.to_numeric(self.df['Start Year'], errors='coerce')

        # Try to extract year from Commercial Operation Date if Start Year is missing
        def extract_year_from_date(date_str):
            try:
                if pd.isna(date_str):
                    return np.nan
                date_str = str(date_str)
                # Try to extract year (assuming format YYYY-MM-DD or similar)
                if '-' in date_str:
                    year = int(date_str.split('-')[0])
                    return year if 1950 <= year <= 2100 else np.nan
                return np.nan
            except:
                return np.nan

        # Fill missing start years from operation date
        mask = self.df['start_year_extracted'].isna()
        self.df.loc[mask, 'start_year_extracted'] = self.df.loc[mask, 'Commercial Operation Date'].apply(extract_year_from_date)

        # Extract retirement year
        self.df['retirement_year_extracted'] = pd.to_numeric(self.df['Retirement Year'], errors='coerce')

        # For operating plants without retirement year, estimate based on start year + lifetime
        operating_mask = self.df['Status'].str.lower() == 'operating'
        no_retirement = self.df['retirement_year_extracted'].isna()

        self.df.loc[operating_mask & no_retirement, 'retirement_year_extracted'] = (
            self.df.loc[operating_mask & no_retirement, 'start_year_extracted'] + self.DEFAULT_LIFETIME
        )

        # Assign capacity factor
        def get_capacity_factor(status):
            status_lower = str(status).lower()
            for key in self.CAPACITY_FACTORS:
                if key in status_lower:
                    return self.CAPACITY_FACTORS[key]
            return self.CAPACITY_FACTORS['default']

        self.df['capacity_factor'] = self.df['Status'].apply(get_capacity_factor)

        # Summary
        print(f"\nData preparation summary:")
        print(f"  Plants with start year: {self.df['start_year_extracted'].notna().sum():,}")
        print(f"  Plants with retirement year: {self.df['retirement_year_extracted'].notna().sum():,}")
        print(f"  Start year range: {self.df['start_year_extracted'].min():.0f} - {self.df['start_year_extracted'].max():.0f}")

        # Status breakdown
        print(f"\nPlants by status:")
        status_counts = self.df['Status'].value_counts()
        for status, count in status_counts.head(10).items():
            print(f"  {status:30s}: {count:,}")

        return self.df

    def project_regional_generation(self, start_year: int = None, end_year: int = 2050) -> pd.DataFrame:
        """
        Project regional nuclear generation from start_year to end_year

        Args:
            start_year: Starting year for projection (default: current year)
            end_year: Ending year for projection (default: 2050)

        Returns:
            DataFrame with yearly projections by region
        """
        if self.df is None:
            print("WARNING -  No data loaded.")
            return None

        if start_year is None:
            start_year = self.current_year

        print("\n" + "="*80)
        print(f"PROJECTING REGIONAL GENERATION ({start_year} - {end_year})")
        print("="*80)

        # Create year range
        years = range(start_year, end_year + 1)
        regions = self.df['Region'].dropna().unique()

        # Initialize projections dictionary
        projections_data = []

        for region in regions:
            region_plants = self.df[self.df['Region'] == region].copy()

            for year in years:
                # Filter plants operating in this year
                # Plant operates if: start_year <= year < retirement_year
                operating_plants = region_plants[
                    (region_plants['start_year_extracted'] <= year) &
                    (
                        (region_plants['retirement_year_extracted'] > year) |
                        (region_plants['retirement_year_extracted'].isna())
                    )
                ].copy()

                # Calculate generation
                # TWh = MW × 8760 hours × CF / 1,000,000
                operating_plants['generation_twh'] = (
                    operating_plants['capacity_numeric'] *
                    8760 *
                    operating_plants['capacity_factor'] /
                    1_000_000
                )

                total_capacity = operating_plants['capacity_numeric'].sum() / 1000  # GW
                total_generation = operating_plants['generation_twh'].sum()
                unit_count = len(operating_plants)

                projections_data.append({
                    'year': year,
                    'region': region,
                    'capacity_gw': total_capacity,
                    'generation_twh': total_generation,
                    'unit_count': unit_count
                })

        self.projections = pd.DataFrame(projections_data)

        # Display summary
        print(f"\nDATA - Projection Summary ({end_year}):")
        final_year = self.projections[self.projections['year'] == end_year].copy()
        final_year = final_year.sort_values('generation_twh', ascending=False)

        print(f"\n{'Region':<35s} {'Capacity (GW)':>15s} {'Generation (TWh)':>18s} {'Units':>8s}")
        print("-" * 80)

        for _, row in final_year.iterrows():
            print(f"{row['region']:<35s} {row['capacity_gw']:>15,.1f} "
                  f"{row['generation_twh']:>18,.1f} {row['unit_count']:>8,}")

        total_gen = final_year['generation_twh'].sum()
        total_cap = final_year['capacity_gw'].sum()
        print("-" * 80)
        print(f"{'TOTAL':<35s} {total_cap:>15,.1f} {total_gen:>18,.1f}")

        return self.projections

    def project_global_generation(self, start_year: int = None, end_year: int = 2050) -> pd.DataFrame:
        """
        Project global nuclear generation (aggregated across all regions)

        Args:
            start_year: Starting year for projection
            end_year: Ending year for projection

        Returns:
            DataFrame with yearly global projections
        """
        if self.projections is None:
            print("WARNING -  Run project_regional_generation() first.")
            return None

        print("\n" + "="*80)
        print(f"GLOBAL GENERATION PROJECTION ({start_year or self.current_year} - {end_year})")
        print("="*80)

        # Aggregate by year
        global_proj = self.projections.groupby('year').agg({
            'capacity_gw': 'sum',
            'generation_twh': 'sum',
            'unit_count': 'sum'
        }).reset_index()

        # Display trends
        print(f"\nGlobal Generation Trends (5-year intervals):")
        print(f"{'Year':>6s} {'Capacity (GW)':>15s} {'Generation (TWh)':>18s} {'Units':>8s} {'YoY Growth':>12s}")
        print("-" * 65)

        prev_gen = None
        for _, row in global_proj.iterrows():
            if row['year'] % 5 == 0 or row['year'] == end_year:
                yoy = ""
                if prev_gen is not None and prev_gen > 0:
                    growth = ((row['generation_twh'] - prev_gen) / prev_gen) * 100
                    yoy = f"{growth:+.1f}%"

                print(f"{row['year']:>6.0f} {row['capacity_gw']:>15,.1f} "
                      f"{row['generation_twh']:>18,.1f} {row['unit_count']:>8,.0f} {yoy:>12s}")

                if row['year'] % 5 == 0:
                    prev_gen = row['generation_twh']

        return global_proj

    def analyze_pipeline_contribution(self, target_year: int = 2050):
        """
        Analyze contribution of pipeline projects (construction, announced) to target year

        Args:
            target_year: Year to analyze

        Returns:
            DataFrame with pipeline analysis
        """
        if self.df is None:
            print("WARNING -  No data loaded.")
            return None

        print("\n" + "="*80)
        print(f"PIPELINE CONTRIBUTION ANALYSIS (Target: {target_year})")
        print("="*80)

        # Categorize plants by status
        pipeline_statuses = ['construction', 'pre-construction', 'announced']
        operating_status = 'operating'

        results = []

        for region in self.df['Region'].dropna().unique():
            region_plants = self.df[self.df['Region'] == region].copy()

            # Operating plants
            operating = region_plants[region_plants['Status'].str.lower() == operating_status]
            operating_active_2050 = operating[
                (operating['start_year_extracted'] <= target_year) &
                (
                    (operating['retirement_year_extracted'] > target_year) |
                    (operating['retirement_year_extracted'].isna())
                )
            ]

            # Pipeline plants
            pipeline = region_plants[
                region_plants['Status'].str.lower().apply(
                    lambda x: any(s in str(x) for s in pipeline_statuses)
                )
            ]
            pipeline_active_2050 = pipeline[
                (pipeline['start_year_extracted'] <= target_year) &
                (
                    (pipeline['retirement_year_extracted'] > target_year) |
                    (pipeline['retirement_year_extracted'].isna())
                )
            ]

            # Calculate generation
            operating_gen = (operating_active_2050['capacity_numeric'] * 8760 * 0.90 / 1_000_000).sum()
            pipeline_gen = (pipeline_active_2050['capacity_numeric'] * 8760 * 0.85 / 1_000_000).sum()

            results.append({
                'region': region,
                'operating_generation_twh': operating_gen,
                'pipeline_generation_twh': pipeline_gen,
                'total_generation_twh': operating_gen + pipeline_gen,
                'pipeline_percentage': (pipeline_gen / (operating_gen + pipeline_gen) * 100) if (operating_gen + pipeline_gen) > 0 else 0,
                'operating_units': len(operating_active_2050),
                'pipeline_units': len(pipeline_active_2050)
            })

        pipeline_df = pd.DataFrame(results)
        pipeline_df = pipeline_df.sort_values('total_generation_twh', ascending=False)

        print(f"\n{'Region':<35s} {'Operating':>12s} {'Pipeline':>12s} {'Total':>12s} {'Pipeline %':>12s}")
        print("-" * 85)

        for _, row in pipeline_df.iterrows():
            print(f"{row['region']:<35s} {row['operating_generation_twh']:>12,.1f} "
                  f"{row['pipeline_generation_twh']:>12,.1f} {row['total_generation_twh']:>12,.1f} "
                  f"{row['pipeline_percentage']:>11,.1f}%")

        return pipeline_df

    def save_projections(self, output_path: str = None):
        """
        Save projections to CSV

        Args:
            output_path: Path to save projections
        """
        if self.projections is None:
            print("WARNING -  No projections created.")
            return

        if output_path is None:
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / "nuclear_projections_2050.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.projections.to_csv(output_path, index=False)
        print(f"\nOK - Projections saved to: {output_path}")
        print(f"   {len(self.projections):,} rows (region × year combinations)")


def main():
    """Main execution function for projection modeling"""
    print("="*80)
    print("NUCLEAR GENERATION PROJECTION MODEL (2025-2050)")
    print("="*80)

    # Initialize model
    model = NuclearProjectionModel()

    # Load data
    print("\n[1] Loading cleaned data...")
    model.load_data()

    # Prepare projection data
    print("\n[2] Preparing projection data...")
    model.prepare_projection_data()

    # Project regional generation
    print("\n[3] Projecting regional generation...")
    regional_proj = model.project_regional_generation(start_year=2025, end_year=2050)

    # Project global generation
    print("\n[4] Calculating global projections...")
    global_proj = model.project_global_generation(end_year=2050)

    # Analyze pipeline contribution
    print("\n[5] Analyzing pipeline contribution...")
    pipeline = model.analyze_pipeline_contribution(target_year=2050)

    # Save projections
    print("\n[6] Saving projections...")
    model.save_projections()

    # Save additional outputs
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"

    global_proj.to_csv(output_dir / "global_projections_2050.csv", index=False)
    pipeline.to_csv(output_dir / "pipeline_contribution_2050.csv", index=False)

    print(f"\nOK - Additional files saved to: {output_dir}")
    print("  * global_projections_2050.csv")
    print("  * pipeline_contribution_2050.csv")

    print("\n" + "="*80)
    print("OK - PROJECTION MODELING COMPLETE!")
    print("="*80)
    print(f"\nKey findings for 2050:")

    final_year = global_proj[global_proj['year'] == 2050].iloc[0]
    print(f"  * Global Capacity: {final_year['capacity_gw']:,.1f} GW")
    print(f"  * Global Generation: {final_year['generation_twh']:,.1f} TWh/year")
    print(f"  * Total Units: {final_year['unit_count']:,.0f}")

    total_pipeline_gen = pipeline['pipeline_generation_twh'].sum()
    total_gen = pipeline['total_generation_twh'].sum()
    print(f"  * Pipeline contribution: {(total_pipeline_gen/total_gen)*100:.1f}% of 2050 generation")


if __name__ == "__main__":
    main()
