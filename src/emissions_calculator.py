"""
Emissions Calculator Module

This module calculates avoided CO2 emissions from nuclear energy deployment
by comparing nuclear generation against fossil fuel baseline scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class EmissionsCalculator:
    """Calculate avoided CO2 emissions from nuclear energy"""

    # Emission factors (gCO2/kWh)
    # Source: IPCC lifecycle emissions assessments
    EMISSION_FACTORS = {
        'coal': 820,          # gCO2/kWh (lifecycle)
        'natural_gas': 490,   # gCO2/kWh (lifecycle)
        'oil': 650,           # gCO2/kWh (lifecycle)
        'nuclear': 12,        # gCO2/kWh (lifecycle)
        'mixed_fossil': 655   # Weighted average of coal + gas (50/50 mix)
    }

    # Regional fossil fuel mix assumptions (% coal vs gas)
    # Used to calculate region-specific baseline emissions
    REGIONAL_FOSSIL_MIX = {
        'Asia': {'coal': 0.70, 'gas': 0.30},
        'Europe': {'coal': 0.30, 'gas': 0.70},
        'Northern America': {'coal': 0.35, 'gas': 0.65},
        'Latin America and the Caribbean': {'coal': 0.40, 'gas': 0.60},
        'Africa': {'coal': 0.60, 'gas': 0.40},
        'Oceania': {'coal': 0.50, 'gas': 0.50},
        'Western Asia': {'coal': 0.20, 'gas': 0.80}
    }

    def __init__(self, projections_path: str = None):
        """
        Initialize emissions calculator

        Args:
            projections_path: Path to nuclear projections CSV
        """
        if projections_path is None:
            project_root = Path(__file__).parent.parent
            self.projections_path = project_root / "data" / "processed" / "nuclear_projections_2050.csv"
        else:
            self.projections_path = Path(projections_path)

        self.projections = None
        self.emissions_avoided = None

    def load_projections(self):
        """Load nuclear generation projections"""
        try:
            self.projections = pd.read_csv(self.projections_path)
            print(f"OK - Loaded projections: {len(self.projections):,} rows")
            return self.projections
        except FileNotFoundError:
            print(f"ERROR - Error: Projections file not found at {self.projections_path}")
            print(f"   Please run projection_model.py first.")
            return None
        except Exception as e:
            print(f"ERROR - Error loading projections: {e}")
            return None

    def calculate_regional_emission_factor(self, region: str) -> float:
        """
        Calculate region-specific baseline emission factor based on fossil mix

        Args:
            region: Region name

        Returns:
            Emission factor in gCO2/kWh
        """
        if region not in self.REGIONAL_FOSSIL_MIX:
            # Use global average if region not specified
            return self.EMISSION_FACTORS['mixed_fossil']

        mix = self.REGIONAL_FOSSIL_MIX[region]
        coal_share = mix.get('coal', 0.5)
        gas_share = mix.get('gas', 0.5)

        # Weighted average emission factor
        ef = (coal_share * self.EMISSION_FACTORS['coal'] +
              gas_share * self.EMISSION_FACTORS['natural_gas'])

        return ef

    def calculate_avoided_emissions(self, baseline: str = 'mixed_fossil'):
        """
        Calculate avoided emissions from nuclear generation

        Args:
            baseline: Baseline scenario ('coal', 'natural_gas', 'mixed_fossil', or 'regional')

        Returns:
            DataFrame with avoided emissions by region and year
        """
        if self.projections is None:
            print("WARNING -  No projections loaded.")
            return None

        print("\n" + "="*80)
        print(f"CALCULATING AVOIDED CO2 EMISSIONS (Baseline: {baseline})")
        print("="*80)

        emissions_data = []

        for _, row in self.projections.iterrows():
            year = row['year']
            region = row['region']
            generation_twh = row['generation_twh']

            # Determine baseline emission factor
            if baseline == 'regional':
                baseline_ef = self.calculate_regional_emission_factor(region)
            else:
                baseline_ef = self.EMISSION_FACTORS.get(baseline, self.EMISSION_FACTORS['mixed_fossil'])

            nuclear_ef = self.EMISSION_FACTORS['nuclear']

            # Calculate avoided emissions
            # Generation (TWh) × (Baseline EF - Nuclear EF) (gCO2/kWh) / 1e9 = MtCO2
            # 1 TWh = 1e9 kWh
            avoided_emissions_mt = (
                generation_twh * 1e9 * (baseline_ef - nuclear_ef) / 1e9
            )

            emissions_data.append({
                'year': year,
                'region': region,
                'generation_twh': generation_twh,
                'baseline_emission_factor_gco2_kwh': baseline_ef,
                'nuclear_emission_factor_gco2_kwh': nuclear_ef,
                'avoided_emissions_mtco2': avoided_emissions_mt,
                'baseline_scenario': baseline
            })

        self.emissions_avoided = pd.DataFrame(emissions_data)

        return self.emissions_avoided

    def analyze_global_avoided_emissions(self):
        """
        Analyze global avoided emissions over time

        Returns:
            DataFrame with global emissions by year
        """
        if self.emissions_avoided is None:
            print("WARNING -  Calculate avoided emissions first.")
            return None

        print("\n" + "="*80)
        print("GLOBAL AVOIDED EMISSIONS ANALYSIS")
        print("="*80)

        # Aggregate by year
        global_emissions = self.emissions_avoided.groupby('year').agg({
            'generation_twh': 'sum',
            'avoided_emissions_mtco2': 'sum'
        }).reset_index()

        # Calculate cumulative emissions
        global_emissions['cumulative_avoided_mtco2'] = global_emissions['avoided_emissions_mtco2'].cumsum()
        global_emissions['cumulative_avoided_gtco2'] = global_emissions['cumulative_avoided_mtco2'] / 1000

        # Display milestones
        milestone_years = [2025, 2030, 2035, 2040, 2045, 2050]
        milestones = global_emissions[global_emissions['year'].isin(milestone_years)]

        print(f"\n{'Year':>6s} {'Generation (TWh)':>18s} {'Avoided (MtCO2)':>18s} {'Cumulative (GtCO2)':>20s}")
        print("-" * 70)

        for _, row in milestones.iterrows():
            print(f"{row['year']:>6.0f} {row['generation_twh']:>18,.1f} "
                  f"{row['avoided_emissions_mtco2']:>18,.1f} {row['cumulative_avoided_gtco2']:>20,.2f}")

        # Summary
        total_2050 = milestones[milestones['year'] == 2050].iloc[0]
        cumulative_2025_2050 = global_emissions[
            (global_emissions['year'] >= 2025) & (global_emissions['year'] <= 2050)
        ]['avoided_emissions_mtco2'].sum()

        print(f"\n📊 Summary (2025-2050):")
        print(f"  * Annual avoided emissions (2050): {total_2050['avoided_emissions_mtco2']:,.1f} MtCO2")
        print(f"  * Cumulative avoided emissions: {cumulative_2025_2050/1000:,.2f} GtCO2")
        print(f"  * Equivalent to global CO2 emissions of ~{cumulative_2025_2050/36000:.1f} years (at 36 GtCO2/year)")

        return global_emissions

    def analyze_regional_avoided_emissions(self, target_year: int = 2050):
        """
        Analyze avoided emissions by region for a specific year

        Args:
            target_year: Year to analyze

        Returns:
            DataFrame with regional avoided emissions
        """
        if self.emissions_avoided is None:
            print("WARNING -  Calculate avoided emissions first.")
            return None

        print("\n" + "="*80)
        print(f"REGIONAL AVOIDED EMISSIONS ANALYSIS ({target_year})")
        print("="*80)

        regional = self.emissions_avoided[self.emissions_avoided['year'] == target_year].copy()
        regional = regional.sort_values('avoided_emissions_mtco2', ascending=False)

        print(f"\n{'Region':<35s} {'Generation (TWh)':>18s} {'Avoided Emissions (MtCO2)':>28s}")
        print("-" * 85)

        total_gen = 0
        total_avoided = 0

        for _, row in regional.iterrows():
            print(f"{row['region']:<35s} {row['generation_twh']:>18,.1f} {row['avoided_emissions_mtco2']:>28,.1f}")
            total_gen += row['generation_twh']
            total_avoided += row['avoided_emissions_mtco2']

        print("-" * 85)
        print(f"{'TOTAL':<35s} {total_gen:>18,.1f} {total_avoided:>28,.1f}")

        return regional

    def calculate_cumulative_regional_emissions(self, start_year: int = 2025, end_year: int = 2050):
        """
        Calculate cumulative avoided emissions by region over time period

        Args:
            start_year: Starting year
            end_year: Ending year

        Returns:
            DataFrame with cumulative emissions by region
        """
        if self.emissions_avoided is None:
            print("WARNING -  Calculate avoided emissions first.")
            return None

        print("\n" + "="*80)
        print(f"CUMULATIVE AVOIDED EMISSIONS BY REGION ({start_year}-{end_year})")
        print("="*80)

        # Filter time period
        period_data = self.emissions_avoided[
            (self.emissions_avoided['year'] >= start_year) &
            (self.emissions_avoided['year'] <= end_year)
        ]

        # Sum by region
        cumulative = period_data.groupby('region').agg({
            'generation_twh': 'sum',
            'avoided_emissions_mtco2': 'sum'
        }).reset_index()

        cumulative = cumulative.sort_values('avoided_emissions_mtco2', ascending=False)
        cumulative['avoided_emissions_gtco2'] = cumulative['avoided_emissions_mtco2'] / 1000

        print(f"\n{'Region':<35s} {'Total Gen (TWh)':>18s} {'Avoided (GtCO2)':>18s} {'Share':>10s}")
        print("-" * 85)

        total = cumulative['avoided_emissions_mtco2'].sum()

        for _, row in cumulative.iterrows():
            share = (row['avoided_emissions_mtco2'] / total * 100) if total > 0 else 0
            print(f"{row['region']:<35s} {row['generation_twh']:>18,.1f} "
                  f"{row['avoided_emissions_gtco2']:>18,.2f} {share:>9.1f}%")

        print("-" * 85)
        print(f"{'TOTAL':<35s} {cumulative['generation_twh'].sum():>18,.1f} {cumulative['avoided_emissions_gtco2'].sum():>18,.2f}")

        return cumulative

    def compare_baseline_scenarios(self, year: int = 2050):
        """
        Compare avoided emissions under different baseline scenarios

        Args:
            year: Year to compare

        Returns:
            DataFrame with scenario comparison
        """
        print("\n" + "="*80)
        print(f"BASELINE SCENARIO COMPARISON ({year})")
        print("="*80)

        scenarios = ['coal', 'natural_gas', 'mixed_fossil', 'regional']
        comparison_results = []

        for scenario in scenarios:
            # Calculate for this scenario
            self.calculate_avoided_emissions(baseline=scenario)

            # Get total for target year
            year_data = self.emissions_avoided[self.emissions_avoided['year'] == year]
            total_avoided = year_data['avoided_emissions_mtco2'].sum()

            comparison_results.append({
                'baseline_scenario': scenario,
                'avoided_emissions_mtco2': total_avoided,
                'avoided_emissions_gtco2': total_avoided / 1000
            })

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('avoided_emissions_mtco2', ascending=False)

        print(f"\n{'Baseline Scenario':<25s} {'Avoided Emissions (MtCO2)':>28s} {'Avoided (GtCO2)':>18s}")
        print("-" * 75)

        for _, row in comparison_df.iterrows():
            print(f"{row['baseline_scenario']:<25s} {row['avoided_emissions_mtco2']:>28,.1f} {row['avoided_emissions_gtco2']:>18,.2f}")

        print(f"\n📊 Interpretation:")
        print(f"  * Coal baseline: Maximum potential emissions reduction (replacing coal)")
        print(f"  * Gas baseline: Minimum potential emissions reduction (replacing gas)")
        print(f"  * Mixed/Regional: Realistic scenario (replacing typical fossil mix)")

        return comparison_df

    def save_emissions_data(self, output_path: str = None):
        """
        Save emissions calculations to CSV

        Args:
            output_path: Path to save emissions data
        """
        if self.emissions_avoided is None:
            print("WARNING -  No emissions data to save.")
            return

        if output_path is None:
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / "avoided_emissions.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.emissions_avoided.to_csv(output_path, index=False)
        print(f"\nOK - Emissions data saved to: {output_path}")


def main():
    """Main execution function for emissions calculation"""
    print("="*80)
    print("NUCLEAR AVOIDED EMISSIONS CALCULATOR")
    print("="*80)

    # Initialize calculator
    calculator = EmissionsCalculator()

    # Load projections
    print("\n[1] Loading nuclear projections...")
    calculator.load_projections()

    # Calculate avoided emissions (regional baseline)
    print("\n[2] Calculating avoided emissions...")
    emissions = calculator.calculate_avoided_emissions(baseline='regional')

    # Global analysis
    print("\n[3] Analyzing global avoided emissions...")
    global_emissions = calculator.analyze_global_avoided_emissions()

    # Regional analysis
    print("\n[4] Analyzing regional avoided emissions (2050)...")
    regional_2050 = calculator.analyze_regional_avoided_emissions(target_year=2050)

    # Cumulative regional
    print("\n[5] Calculating cumulative regional emissions...")
    cumulative_regional = calculator.calculate_cumulative_regional_emissions(
        start_year=2025,
        end_year=2050
    )

    # Baseline comparison
    print("\n[6] Comparing baseline scenarios (2050)...")
    baseline_comparison = calculator.compare_baseline_scenarios(year=2050)

    # Save results
    print("\n[7] Saving emissions data...")
    calculator.save_emissions_data()

    # Save additional outputs
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"

    global_emissions.to_csv(output_dir / "global_avoided_emissions.csv", index=False)
    cumulative_regional.to_csv(output_dir / "cumulative_regional_emissions.csv", index=False)
    baseline_comparison.to_csv(output_dir / "baseline_scenario_comparison.csv", index=False)

    print(f"\nOK - Additional files saved to: {output_dir}")
    print("  * global_avoided_emissions.csv")
    print("  * cumulative_regional_emissions.csv")
    print("  * baseline_scenario_comparison.csv")

    print("\n" + "="*80)
    print("OK - EMISSIONS CALCULATION COMPLETE!")
    print("="*80)

    # Final summary
    final_year = global_emissions[global_emissions['year'] == 2050].iloc[0]
    cumulative_total = cumulative_regional['avoided_emissions_gtco2'].sum()

    print(f"\n🌍 Key Findings (2025-2050):")
    print(f"  * Annual avoided emissions (2050): {final_year['avoided_emissions_mtco2']:,.1f} MtCO2/year")
    print(f"  * Cumulative avoided emissions: {cumulative_total:,.2f} GtCO2")
    print(f"  * Climate impact: Equivalent to removing ~{cumulative_total*1000/7.5:.0f} million cars for 25 years")
    print(f"    (Assuming 7.5 tCO2/car/year)")


if __name__ == "__main__":
    main()
