"""
Scenario Comparison Module

This module compares nuclear generation projections against IEA Net Zero scenarios
to identify generation gaps and assess alignment with climate targets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ScenarioComparison:
    """Compare nuclear projections against IEA Net Zero scenarios"""

    # IEA Net Zero 2050 Scenario - Nuclear Generation Targets (TWh)
    # Source: IEA Net Zero by 2050 Roadmap - NZE2021 Annex A
    # These values are loaded from the actual IEA dataset
    IEA_NZE_TARGETS = {
        2019: 2792,
        2020: 2698,
        2030: 3777,
        2040: 4855,
        2050: 5497
    }

    # Regional distribution assumptions (% of global)
    # Replace with actual IEA regional targets when available
    REGIONAL_DISTRIBUTION = {
        'Asia': 0.45,
        'Europe': 0.20,
        'Northern America': 0.20,
        'Latin America and the Caribbean': 0.05,
        'Africa': 0.05,
        'Oceania': 0.03,
        'Western Asia': 0.02
    }

    def __init__(self, projections_path: str = None, iea_data_path: str = None):
        """
        Initialize scenario comparison

        Args:
            projections_path: Path to nuclear projections CSV
            iea_data_path: Path to IEA scenario data (if available)
        """
        if projections_path is None:
            project_root = Path(__file__).parent.parent
            self.projections_path = project_root / "data" / "processed" / "nuclear_projections_2050.csv"
        else:
            self.projections_path = Path(projections_path)

        self.iea_data_path = iea_data_path
        self.projections = None
        self.iea_scenarios = None
        self.comparison = None

    def load_projections(self):
        """Load nuclear generation projections"""
        try:
            self.projections = pd.read_csv(self.projections_path)
            print(f"OK - Loaded projections: {len(self.projections):,} rows")
            return self.projections
        except FileNotFoundError:
            print(f"ERROR - Error: Projections file not found at {self.projections_path}")
            print(f"   Please run projection_model.py first to generate projections.")
            return None
        except Exception as e:
            print(f"ERROR - Error loading projections: {e}")
            return None

    def load_iea_scenarios(self):
        """
        Load IEA Net Zero scenario data

        Loads from NZE2021_AnnexA.csv or uses built-in targets.
        """
        print("\n" + "="*80)
        print("LOADING IEA NET ZERO SCENARIO DATA")
        print("="*80)

        # Try to load from NZE2021_AnnexA.csv first
        project_root = Path(__file__).parent.parent
        nze_annex_path = project_root / "data" / "raw" / "NZE2021_AnnexA.csv"

        if nze_annex_path.exists():
            try:
                print(f"DATA - Loading from IEA NZE2021 Annex A dataset...")
                iea_raw = pd.read_csv(nze_annex_path)

                # Filter for nuclear electricity generation
                nuclear_data = iea_raw[
                    (iea_raw['Product'] == 'Nuclear') &
                    (iea_raw['Flow'] == 'Electricity generation') &
                    (iea_raw['Unit'] == 'TWh')
                ].copy()

                if len(nuclear_data) > 0:
                    # Update IEA_NZE_TARGETS with actual data
                    for _, row in nuclear_data.iterrows():
                        self.IEA_NZE_TARGETS[int(row['Year'])] = float(row['Value'])

                    print(f"OK - Loaded {len(nuclear_data)} data points from IEA NZE2021")
                    print(f"   Source: {nze_annex_path.name}")
                else:
                    print(f"WARNING -  No nuclear generation data found in {nze_annex_path.name}")
                    print("   Using built-in IEA NZE targets")

            except Exception as e:
                print(f"WARNING -  Error loading IEA Annex A data: {e}")
                print("   Using built-in IEA NZE targets")

        elif self.iea_data_path and Path(self.iea_data_path).exists():
            try:
                # Load external IEA data if available
                self.iea_scenarios = pd.read_csv(self.iea_data_path)
                print(f"OK - Loaded IEA data from: {self.iea_data_path}")
                return self.iea_scenarios
            except Exception as e:
                print(f"WARNING -  Error loading IEA data: {e}")
                print("   Using built-in IEA NZE targets")

        # Use IEA_NZE_TARGETS (either loaded or default)
        print(f"INFO -  Using IEA Net Zero 2050 targets from NZE2021 Annex A")

        # Interpolate missing years (2025, 2035, 2045)
        years_sorted = sorted(self.IEA_NZE_TARGETS.keys())
        interpolated_targets = self.IEA_NZE_TARGETS.copy()

        # Linear interpolation for missing years
        for year in [2025, 2035, 2045]:
            if year not in interpolated_targets:
                # Find surrounding years
                year_before = max([y for y in years_sorted if y < year], default=None)
                year_after = min([y for y in years_sorted if y > year], default=None)

                if year_before and year_after:
                    # Linear interpolation
                    val_before = self.IEA_NZE_TARGETS[year_before]
                    val_after = self.IEA_NZE_TARGETS[year_after]
                    interpolated_value = val_before + (val_after - val_before) * (year - year_before) / (year_after - year_before)
                    interpolated_targets[year] = interpolated_value

        self.IEA_NZE_TARGETS = interpolated_targets

        # Create default scenario DataFrame
        iea_data = []
        for year, global_target in self.IEA_NZE_TARGETS.items():
            # Global total
            iea_data.append({
                'year': year,
                'region': 'Global',
                'scenario': 'IEA_NZE_2050',
                'target_generation_twh': global_target
            })

            # Regional breakdown
            for region, share in self.REGIONAL_DISTRIBUTION.items():
                iea_data.append({
                    'year': year,
                    'region': region,
                    'scenario': 'IEA_NZE_2050',
                    'target_generation_twh': global_target * share
                })

        self.iea_scenarios = pd.DataFrame(iea_data)

        print(f"\nDATA - IEA NZE Global Targets:")
        global_targets = self.iea_scenarios[self.iea_scenarios['region'] == 'Global']
        for _, row in global_targets.iterrows():
            print(f"  {row['year']}: {row['target_generation_twh']:,.0f} TWh")

        return self.iea_scenarios

    def compare_scenarios(self):
        """
        Compare nuclear projections against IEA scenarios

        Returns:
            DataFrame with comparison results
        """
        if self.projections is None or self.iea_scenarios is None:
            print("WARNING -  Load projections and IEA scenarios first.")
            return None

        print("\n" + "="*80)
        print("COMPARING PROJECTIONS VS IEA NET ZERO SCENARIO")
        print("="*80)

        # Aggregate projections by year and region
        proj_agg = self.projections.groupby(['year', 'region']).agg({
            'generation_twh': 'sum',
            'capacity_gw': 'sum',
            'unit_count': 'sum'
        }).reset_index()

        # Merge with IEA targets
        comparison = proj_agg.merge(
            self.iea_scenarios[self.iea_scenarios['region'] != 'Global'],
            on=['year', 'region'],
            how='outer'
        )

        # Fill NaN values
        comparison['generation_twh'] = comparison['generation_twh'].fillna(0)
        comparison['target_generation_twh'] = comparison['target_generation_twh'].fillna(0)

        # Calculate gap
        comparison['gap_twh'] = comparison['target_generation_twh'] - comparison['generation_twh']
        comparison['gap_percentage'] = (
            (comparison['gap_twh'] / comparison['target_generation_twh'] * 100)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        # Categorize alignment
        def categorize_alignment(gap_pct):
            if gap_pct < -10:
                return 'Exceeds target'
            elif -10 <= gap_pct < 10:
                return 'On track'
            elif 10 <= gap_pct < 30:
                return 'Shortfall'
            else:
                return 'Major shortfall'

        comparison['alignment'] = comparison['gap_percentage'].apply(categorize_alignment)

        self.comparison = comparison

        return comparison

    def analyze_global_gap(self):
        """
        Analyze global generation gap vs IEA NZE scenario

        Returns:
            DataFrame with global gap analysis
        """
        if self.comparison is None:
            print("WARNING -  Run compare_scenarios() first.")
            return None

        print("\n" + "="*80)
        print("GLOBAL GENERATION GAP ANALYSIS")
        print("="*80)

        # Calculate global totals by year
        global_comparison = self.comparison.groupby('year').agg({
            'generation_twh': 'sum',
            'target_generation_twh': 'sum',
            'gap_twh': 'sum'
        }).reset_index()

        global_comparison['gap_percentage'] = (
            global_comparison['gap_twh'] / global_comparison['target_generation_twh'] * 100
        )

        # Get milestone years
        milestone_years = [2025, 2030, 2035, 2040, 2045, 2050]
        milestones = global_comparison[global_comparison['year'].isin(milestone_years)]

        print(f"\n{'Year':>6s} {'Projected (TWh)':>18s} {'IEA Target (TWh)':>18s} {'Gap (TWh)':>15s} {'Gap (%)':>10s}")
        print("-" * 72)

        for _, row in milestones.iterrows():
            gap_sign = "+" if row['gap_twh'] > 0 else ""
            print(f"{row['year']:>6.0f} {row['generation_twh']:>18,.1f} {row['target_generation_twh']:>18,.1f} "
                  f"{gap_sign}{row['gap_twh']:>14,.1f} {row['gap_percentage']:>9,.1f}%")

        # Summary statistics
        gap_2050 = milestones[milestones['year'] == 2050].iloc[0]
        print(f"\nDATA - 2050 Summary:")
        print(f"  * Projected Generation: {gap_2050['generation_twh']:,.1f} TWh")
        print(f"  * IEA NZE Target: {gap_2050['target_generation_twh']:,.1f} TWh")
        print(f"  * Gap: {gap_2050['gap_twh']:+,.1f} TWh ({gap_2050['gap_percentage']:+.1f}%)")

        if gap_2050['gap_twh'] > 0:
            print(f"  * Status: WARNING -  SHORTFALL - Additional capacity needed")
            additional_capacity = gap_2050['gap_twh'] / (8760 * 0.85 / 1000)  # Convert TWh to GW
            print(f"  * Additional capacity needed: ~{additional_capacity:,.0f} GW")
        else:
            print(f"  * Status: OK - ON TRACK or exceeding target")

        return global_comparison

    def analyze_regional_gaps(self, target_year: int = 2050):
        """
        Analyze regional generation gaps for a specific year

        Args:
            target_year: Year to analyze (default: 2050)

        Returns:
            DataFrame with regional gap analysis
        """
        if self.comparison is None:
            print("WARNING -  Run compare_scenarios() first.")
            return None

        print("\n" + "="*80)
        print(f"REGIONAL GENERATION GAP ANALYSIS ({target_year})")
        print("="*80)

        regional_gaps = self.comparison[self.comparison['year'] == target_year].copy()
        regional_gaps = regional_gaps.sort_values('gap_twh', ascending=False)

        print(f"\n{'Region':<35s} {'Projected':>12s} {'Target':>12s} {'Gap':>12s} {'Gap %':>10s} {'Status':>15s}")
        print("-" * 100)

        for _, row in regional_gaps.iterrows():
            gap_sign = "+" if row['gap_twh'] > 0 else ""
            print(f"{row['region']:<35s} {row['generation_twh']:>12,.1f} "
                  f"{row['target_generation_twh']:>12,.1f} {gap_sign}{row['gap_twh']:>11,.1f} "
                  f"{row['gap_percentage']:>9,.1f}% {row['alignment']:>15s}")

        # Summary by alignment category
        print(f"\nDATA - Regional Alignment Summary:")
        alignment_summary = regional_gaps['alignment'].value_counts()
        for category, count in alignment_summary.items():
            print(f"  * {category}: {count} regions")

        return regional_gaps

    def calculate_capacity_requirements(self, target_year: int = 2050):
        """
        Calculate additional capacity requirements to meet IEA targets

        Args:
            target_year: Year to analyze

        Returns:
            DataFrame with capacity requirements by region
        """
        if self.comparison is None:
            print("WARNING -  Run compare_scenarios() first.")
            return None

        print("\n" + "="*80)
        print(f"ADDITIONAL CAPACITY REQUIREMENTS ({target_year})")
        print("="*80)

        requirements = self.comparison[self.comparison['year'] == target_year].copy()

        # Calculate additional capacity needed (assuming 85% capacity factor)
        # Gap (TWh) → Additional Capacity (GW)
        # Capacity (GW) = Gap (TWh) / (8760 hours × CF / 1000)
        requirements['additional_capacity_needed_gw'] = (
            requirements['gap_twh'] / (8760 * 0.85 / 1000)
        )

        # Only show regions with shortfall
        shortfall = requirements[requirements['gap_twh'] > 0].copy()
        shortfall = shortfall.sort_values('additional_capacity_needed_gw', ascending=False)

        if len(shortfall) == 0:
            print("\nOK - All regions on track to meet IEA targets!")
            return requirements

        print(f"\nRegions with capacity shortfall:")
        print(f"{'Region':<35s} {'Gap (TWh)':>15s} {'Additional Capacity (GW)':>25s}")
        print("-" * 80)

        total_gap = 0
        total_capacity = 0

        for _, row in shortfall.iterrows():
            print(f"{row['region']:<35s} {row['gap_twh']:>15,.1f} {row['additional_capacity_needed_gw']:>25,.1f}")
            total_gap += row['gap_twh']
            total_capacity += row['additional_capacity_needed_gw']

        print("-" * 80)
        print(f"{'TOTAL SHORTFALL':<35s} {total_gap:>15,.1f} {total_capacity:>25,.1f}")

        # Estimate number of reactors needed (assuming 1 GW average reactor size)
        avg_reactor_size = 1.0  # GW
        reactors_needed = total_capacity / avg_reactor_size

        print(f"\nDATA - Summary:")
        print(f"  * Total capacity shortfall: {total_capacity:,.1f} GW")
        print(f"  * Estimated reactors needed: ~{reactors_needed:,.0f} (assuming {avg_reactor_size} GW average)")
        print(f"  * Total generation gap: {total_gap:,.1f} TWh/year")

        return requirements

    def save_comparison(self, output_path: str = None):
        """
        Save comparison results to CSV

        Args:
            output_path: Path to save comparison results
        """
        if self.comparison is None:
            print("WARNING -  No comparison data to save.")
            return

        if output_path is None:
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / "scenario_comparison.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.comparison.to_csv(output_path, index=False)
        print(f"\nOK - Comparison saved to: {output_path}")


def main():
    """Main execution function for scenario comparison"""
    print("="*80)
    print("NUCLEAR SCENARIO COMPARISON - IEA NET ZERO 2050")
    print("="*80)

    # Initialize comparison
    comparator = ScenarioComparison()

    # Load projections
    print("\n[1] Loading nuclear projections...")
    comparator.load_projections()

    # Load IEA scenarios
    print("\n[2] Loading IEA Net Zero scenarios...")
    comparator.load_iea_scenarios()

    # Compare scenarios
    print("\n[3] Comparing scenarios...")
    comparison = comparator.compare_scenarios()

    # Analyze global gap
    print("\n[4] Analyzing global generation gap...")
    global_gap = comparator.analyze_global_gap()

    # Analyze regional gaps
    print("\n[5] Analyzing regional gaps (2050)...")
    regional_gaps = comparator.analyze_regional_gaps(target_year=2050)

    # Calculate capacity requirements
    print("\n[6] Calculating additional capacity requirements...")
    requirements = comparator.calculate_capacity_requirements(target_year=2050)

    # Save results
    print("\n[7] Saving comparison results...")
    comparator.save_comparison()

    # Save additional outputs
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"

    global_gap.to_csv(output_dir / "global_gap_analysis.csv", index=False)
    regional_gaps.to_csv(output_dir / "regional_gaps_2050.csv", index=False)

    print(f"\nOK - Additional files saved to: {output_dir}")
    print("  * global_gap_analysis.csv")
    print("  * regional_gaps_2050.csv")

    print("\n" + "="*80)
    print("OK - SCENARIO COMPARISON COMPLETE!")
    print("="*80)
    print("\nINFO -  Analysis uses IEA Net Zero Emissions by 2050 Scenario (NZE2021)")
    print("   Source: data/raw/NZE2021_AnnexA.csv")
    print("\nNOTE: IEA data provides global targets only.")
    print("      Regional distribution is estimated using standard assumptions.")


if __name__ == "__main__":
    main()
