"""
Feature Engineering Module for Nuclear Power Analysis

This module calculates capacity factors, regional aggregations, and
prepares features for projection modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class NuclearFeatureEngineer:
    """Calculate capacity factors and regional aggregations for nuclear plants"""

    # Standard capacity factors by reactor status and type
    CAPACITY_FACTORS = {
        'operating': 0.90,      # Operating plants typically achieve 90% capacity factor
        'construction': 0.85,   # New reactors might be slightly lower initially
        'pre-construction': 0.85,
        'announced': 0.85,
        'default': 0.85
    }

    def __init__(self, data_path: str = None):
        """
        Initialize feature engineer

        Args:
            data_path: Path to cleaned nuclear tracker CSV
        """
        if data_path is None:
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "processed" / "nuclear_tracker_cleaned.csv"
        else:
            self.data_path = Path(data_path)

        self.df = None
        self.features_df = None

    def load_data(self):
        """Load cleaned nuclear tracker data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"OK - Loaded data: {len(self.df):,} rows × {len(self.df.columns)} columns")
            print(f"   Source: {self.data_path.name}")
            return self.df
        except FileNotFoundError:
            print(f"ERROR - Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"ERROR - Error loading data: {e}")
            return None

    def calculate_capacity_factors(self):
        """
        Calculate capacity factor for each plant based on status

        Returns:
            DataFrame with capacity_factor column added
        """
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return None

        print("\n" + "="*80)
        print("CALCULATING CAPACITY FACTORS")
        print("="*80)

        # Map status to capacity factor
        def get_capacity_factor(status):
            status_lower = str(status).lower()
            for key in self.CAPACITY_FACTORS:
                if key in status_lower:
                    return self.CAPACITY_FACTORS[key]
            return self.CAPACITY_FACTORS['default']

        self.df['capacity_factor'] = self.df['Status'].apply(get_capacity_factor)

        # Display capacity factor distribution
        print("\nCapacity Factor Distribution by Status:")
        cf_by_status = self.df.groupby('Status').agg({
            'capacity_factor': 'first',
            'Unit Name': 'count'
        }).rename(columns={'Unit Name': 'count'})

        for status, row in cf_by_status.iterrows():
            print(f"  {status:30s}: CF={row['capacity_factor']:.2f} ({row['count']:,} units)")

        return self.df

    def calculate_annual_generation(self):
        """
        Calculate potential annual generation (TWh/year) for each plant

        Formula: Generation (TWh) = Capacity (MW) × 8760 hours × Capacity Factor / 1,000,000
        """
        if self.df is None or 'capacity_factor' not in self.df.columns:
            print("WARNING -  Calculate capacity factors first.")
            return None

        print("\n" + "="*80)
        print("CALCULATING ANNUAL GENERATION")
        print("="*80)

        # Handle missing capacity values
        self.df['capacity_numeric'] = pd.to_numeric(self.df['capacity_numeric'], errors='coerce')

        # Calculate annual generation in TWh
        # MW × 8760 hours × CF / 1,000,000 = TWh
        self.df['annual_generation_twh'] = (
            self.df['capacity_numeric'] * 8760 * self.df['capacity_factor'] / 1_000_000
        )

        # Summary statistics
        total_capacity = self.df['capacity_numeric'].sum() / 1000  # GW
        total_generation = self.df['annual_generation_twh'].sum()

        print(f"\n📊 Overall Statistics:")
        print(f"  Total Capacity:    {total_capacity:,.1f} GW")
        print(f"  Total Generation:  {total_generation:,.1f} TWh/year")
        print(f"  Average CF:        {self.df['capacity_factor'].mean():.2%}")

        return self.df

    def aggregate_by_region(self) -> pd.DataFrame:
        """
        Aggregate capacity and generation by region

        Returns:
            DataFrame with regional aggregations
        """
        if self.df is None or 'annual_generation_twh' not in self.df.columns:
            print("WARNING -  Calculate annual generation first.")
            return None

        print("\n" + "="*80)
        print("REGIONAL AGGREGATION")
        print("="*80)

        regional_agg = self.df.groupby('Region').agg({
            'capacity_numeric': 'sum',
            'annual_generation_twh': 'sum',
            'Unit Name': 'count',
            'capacity_factor': 'mean'
        }).rename(columns={
            'capacity_numeric': 'total_capacity_mw',
            'Unit Name': 'unit_count'
        })

        # Convert capacity to GW
        regional_agg['total_capacity_gw'] = regional_agg['total_capacity_mw'] / 1000

        # Sort by generation
        regional_agg = regional_agg.sort_values('annual_generation_twh', ascending=False)

        print("\nRegional Statistics (sorted by generation):")
        print(f"{'Region':<35s} {'Capacity (GW)':>15s} {'Generation (TWh)':>18s} {'Units':>8s} {'Avg CF':>8s}")
        print("-" * 90)

        for region, row in regional_agg.iterrows():
            print(f"{region:<35s} {row['total_capacity_gw']:>15,.1f} {row['annual_generation_twh']:>18,.1f} "
                  f"{row['unit_count']:>8,} {row['capacity_factor']:>8.1%}")

        return regional_agg

    def aggregate_by_subregion(self) -> pd.DataFrame:
        """
        Aggregate capacity and generation by subregion

        Returns:
            DataFrame with subregional aggregations
        """
        if self.df is None or 'annual_generation_twh' not in self.df.columns:
            print("WARNING -  Calculate annual generation first.")
            return None

        print("\n" + "="*80)
        print("SUBREGIONAL AGGREGATION")
        print("="*80)

        subregional_agg = self.df.groupby(['Region', 'Subregion']).agg({
            'capacity_numeric': 'sum',
            'annual_generation_twh': 'sum',
            'Unit Name': 'count',
            'capacity_factor': 'mean'
        }).rename(columns={
            'capacity_numeric': 'total_capacity_mw',
            'Unit Name': 'unit_count'
        })

        subregional_agg['total_capacity_gw'] = subregional_agg['total_capacity_mw'] / 1000
        subregional_agg = subregional_agg.sort_values('annual_generation_twh', ascending=False)

        print(f"\nTop 15 Subregions by Generation:")
        print(f"{'Region':<25s} {'Subregion':<30s} {'Capacity (GW)':>15s} {'Gen (TWh)':>12s}")
        print("-" * 85)

        for (region, subregion), row in subregional_agg.head(15).iterrows():
            print(f"{region:<25s} {subregion:<30s} {row['total_capacity_gw']:>15,.1f} {row['annual_generation_twh']:>12,.1f}")

        return subregional_agg

    def aggregate_by_country(self) -> pd.DataFrame:
        """
        Aggregate capacity and generation by country

        Returns:
            DataFrame with country-level aggregations
        """
        if self.df is None or 'annual_generation_twh' not in self.df.columns:
            print("WARNING -  Calculate annual generation first.")
            return None

        print("\n" + "="*80)
        print("COUNTRY AGGREGATION")
        print("="*80)

        country_agg = self.df.groupby('Country/Area').agg({
            'capacity_numeric': 'sum',
            'annual_generation_twh': 'sum',
            'Unit Name': 'count',
            'capacity_factor': 'mean',
            'Region': 'first'
        }).rename(columns={
            'capacity_numeric': 'total_capacity_mw',
            'Unit Name': 'unit_count'
        })

        country_agg['total_capacity_gw'] = country_agg['total_capacity_mw'] / 1000
        country_agg = country_agg.sort_values('annual_generation_twh', ascending=False)

        print(f"\nTop 20 Countries by Generation:")
        print(f"{'Country':<30s} {'Region':<25s} {'Capacity (GW)':>15s} {'Gen (TWh)':>12s} {'Units':>7s}")
        print("-" * 95)

        for country, row in country_agg.head(20).iterrows():
            print(f"{country:<30s} {row['Region']:<25s} {row['total_capacity_gw']:>15,.1f} "
                  f"{row['annual_generation_twh']:>12,.1f} {row['unit_count']:>7,}")

        return country_agg

    def aggregate_by_status(self) -> pd.DataFrame:
        """
        Aggregate capacity and generation by status

        Returns:
            DataFrame with status-level aggregations
        """
        if self.df is None or 'annual_generation_twh' not in self.df.columns:
            print("WARNING -  Calculate annual generation first.")
            return None

        print("\n" + "="*80)
        print("STATUS AGGREGATION")
        print("="*80)

        status_agg = self.df.groupby('Status').agg({
            'capacity_numeric': 'sum',
            'annual_generation_twh': 'sum',
            'Unit Name': 'count',
            'capacity_factor': 'mean'
        }).rename(columns={
            'capacity_numeric': 'total_capacity_mw',
            'Unit Name': 'unit_count'
        })

        status_agg['total_capacity_gw'] = status_agg['total_capacity_mw'] / 1000
        status_agg = status_agg.sort_values('annual_generation_twh', ascending=False)

        print(f"\nGeneration by Status:")
        print(f"{'Status':<30s} {'Capacity (GW)':>15s} {'Gen (TWh)':>12s} {'Units':>7s} {'CF':>7s}")
        print("-" * 75)

        for status, row in status_agg.iterrows():
            print(f"{status:<30s} {row['total_capacity_gw']:>15,.1f} "
                  f"{row['annual_generation_twh']:>12,.1f} {row['unit_count']:>7,} {row['capacity_factor']:>7.1%}")

        return status_agg

    def create_feature_dataset(self) -> pd.DataFrame:
        """
        Create complete feature dataset with all calculated fields

        Returns:
            DataFrame with all features
        """
        if self.df is None:
            print("WARNING -  No data loaded.")
            return None

        print("\n" + "="*80)
        print("CREATING FEATURE DATASET")
        print("="*80)

        # Select and organize key columns
        feature_columns = [
            'Country/Area', 'Region', 'Subregion', 'State/Province',
            'Project Name', 'Unit Name', 'Status', 'Reactor Type', 'Model',
            'capacity_numeric', 'capacity_factor', 'annual_generation_twh',
            'Start Year', 'Retirement Year', 'Commercial Operation Date',
            'Construction Start Date', 'Latitude', 'Longitude'
        ]

        # Keep only columns that exist
        existing_cols = [col for col in feature_columns if col in self.df.columns]
        self.features_df = self.df[existing_cols].copy()

        print(f"\nOK - Feature dataset created:")
        print(f"   Rows: {len(self.features_df):,}")
        print(f"   Columns: {len(self.features_df.columns)}")
        print(f"\nFeature columns:")
        for col in self.features_df.columns:
            print(f"  • {col}")

        return self.features_df

    def save_features(self, output_path: str = None):
        """
        Save feature dataset to CSV

        Args:
            output_path: Path to save features CSV
        """
        if self.features_df is None:
            print("WARNING -  No features created. Call create_feature_dataset() first.")
            return

        if output_path is None:
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / "nuclear_features.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.features_df.to_csv(output_path, index=False)
        print(f"\nOK - Features saved to: {output_path}")
        print(f"   {len(self.features_df):,} rows × {len(self.features_df.columns)} columns")


def main():
    """Main execution function for feature engineering"""
    print("="*80)
    print("NUCLEAR POWER TRACKER - FEATURE ENGINEERING")
    print("="*80)

    # Initialize feature engineer
    engineer = NuclearFeatureEngineer()

    # Load data
    print("\n[1] Loading cleaned data...")
    engineer.load_data()

    # Calculate capacity factors
    print("\n[2] Calculating capacity factors...")
    engineer.calculate_capacity_factors()

    # Calculate annual generation
    print("\n[3] Calculating annual generation...")
    engineer.calculate_annual_generation()

    # Regional aggregations
    print("\n[4] Performing regional aggregations...")
    regional = engineer.aggregate_by_region()

    print("\n[5] Performing subregional aggregations...")
    subregional = engineer.aggregate_by_subregion()

    print("\n[6] Performing country aggregations...")
    country = engineer.aggregate_by_country()

    print("\n[7] Performing status aggregations...")
    status = engineer.aggregate_by_status()

    # Create and save feature dataset
    print("\n[8] Creating feature dataset...")
    features = engineer.create_feature_dataset()

    print("\n[9] Saving features...")
    engineer.save_features()

    # Save aggregations
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"

    regional.to_csv(output_dir / "regional_aggregations.csv")
    subregional.to_csv(output_dir / "subregional_aggregations.csv")
    country.to_csv(output_dir / "country_aggregations.csv")
    status.to_csv(output_dir / "status_aggregations.csv")

    print(f"\nOK - Aggregations saved to: {output_dir}")

    print("\n" + "="*80)
    print("OK - FEATURE ENGINEERING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • nuclear_features.csv - Full feature dataset")
    print("  • regional_aggregations.csv - Regional statistics")
    print("  • subregional_aggregations.csv - Subregional statistics")
    print("  • country_aggregations.csv - Country statistics")
    print("  • status_aggregations.csv - Status statistics")


if __name__ == "__main__":
    main()
