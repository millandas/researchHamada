"""
Enhanced Projection Model for Nuclear Energy Generation to 2050
Version 2.0 - With Probabilistic Modeling and Uncertainty Quantification

This module improves upon the base projection model by adding:
- Monte Carlo simulations for uncertainty quantification
- Construction delay modeling
- Probabilistic plant completion rates
- Multiple scenarios (Base, Conservative, Aggressive)
- Sensitivity analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EnhancedNuclearProjectionModel:
    """Project nuclear energy generation to 2050 with uncertainty quantification"""

    # Capacity factors by status
    CAPACITY_FACTORS = {
        'operating': {'mean': 0.90, 'std': 0.05},  # Operating plants: 90% ± 5%
        'construction': {'mean': 0.85, 'std': 0.08},  # New plants: 85% ± 8%
        'announced': {'mean': 0.85, 'std': 0.10}  # Announced: 85% ± 10%
    }

    # Construction delay distributions (years)
    # Based on historical nuclear construction data
    CONSTRUCTION_DELAYS = {
        'base': {'mean': 0, 'std': 2},  # Base case: slight delays
        'conservative': {'mean': 3, 'std': 4},  # Conservative: significant delays
        'aggressive': {'mean': -1, 'std': 1}  # Aggressive: ahead of schedule
    }

    # Plant completion probability by status
    COMPLETION_PROBABILITIES = {
        'operating': 1.0,  # Already operating
        'construction': {'base': 0.95, 'conservative': 0.75, 'aggressive': 0.98},
        'pre-construction': {'base': 0.80, 'conservative': 0.50, 'aggressive': 0.90},
        'announced': {'base': 0.60, 'conservative': 0.30, 'aggressive': 0.75}
    }

    # Lifetime extension probability for operating plants
    LIFETIME_EXTENSION = {
        'probability': 0.7,  # 70% of plants get 20-year extension
        'extension_years': 20
    }

    DEFAULT_LIFETIME = 60

    def __init__(self, data_path: str = None, scenario: str = 'base', n_simulations: int = 1000):
        """
        Initialize enhanced projection model

        Args:
            data_path: Path to cleaned nuclear tracker CSV
            scenario: Scenario type ('base', 'conservative', 'aggressive')
            n_simulations: Number of Monte Carlo simulations
        """
        if data_path is None:
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "processed" / "nuclear_tracker_cleaned.csv"
        else:
            self.data_path = Path(data_path)

        self.scenario = scenario
        self.n_simulations = n_simulations
        self.df = None
        self.projections = None
        self.simulations = []
        self.current_year = datetime.now().year

        print(f"\nInitialized Enhanced Projection Model")
        print(f"  Scenario: {scenario.upper()}")
        print(f"  Simulations: {n_simulations}")

    def load_data(self):
        """Load cleaned nuclear tracker data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"\n[OK] Loaded data: {len(self.df):,} rows")
            return self.df
        except FileNotFoundError:
            print(f"\n[ERROR] File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"\n[ERROR] Error loading data: {e}")
            return None

    def prepare_projection_data(self):
        """Prepare data for probabilistic projection"""
        if self.df is None:
            print("[ERROR] No data loaded.")
            return None

        print("\n" + "="*80)
        print("PREPARING DATA FOR PROBABILISTIC PROJECTION")
        print("="*80)

        # Clean capacity
        self.df['capacity_numeric'] = pd.to_numeric(self.df['capacity_numeric'], errors='coerce')

        # Extract start year
        self.df['start_year_extracted'] = pd.to_numeric(self.df['Start Year'], errors='coerce')

        # Fill missing start years from operation date
        def extract_year_from_date(date_str):
            try:
                if pd.isna(date_str):
                    return np.nan
                date_str = str(date_str)
                if '-' in date_str:
                    year = int(date_str.split('-')[0])
                    return year if 1950 <= year <= 2100 else np.nan
                return np.nan
            except:
                return np.nan

        mask = self.df['start_year_extracted'].isna()
        self.df.loc[mask, 'start_year_extracted'] = self.df.loc[mask, 'Commercial Operation Date'].apply(extract_year_from_date)

        # Extract retirement year
        self.df['retirement_year_extracted'] = pd.to_numeric(self.df['Retirement Year'], errors='coerce')

        # For operating plants without retirement year, estimate with lifetime + potential extension
        operating_mask = self.df['Status'].str.lower() == 'operating'
        no_retirement = self.df['retirement_year_extracted'].isna()

        self.df.loc[operating_mask & no_retirement, 'retirement_year_extracted'] = (
            self.df.loc[operating_mask & no_retirement, 'start_year_extracted'] + self.DEFAULT_LIFETIME
        )

        # Categorize plant status
        self.df['status_category'] = self.df['Status'].apply(self._categorize_status)

        print(f"\n[OK] Data preparation complete")
        print(f"  Plants with start year: {self.df['start_year_extracted'].notna().sum():,}")
        print(f"  Plants with retirement year: {self.df['retirement_year_extracted'].notna().sum():,}")

        # Status breakdown
        print(f"\nPlants by status category:")
        status_counts = self.df['status_category'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status:20s}: {count:,}")

        return self.df

    def _categorize_status(self, status):
        """Categorize plant status into simplified categories"""
        status_lower = str(status).lower()
        if 'operating' in status_lower:
            return 'operating'
        elif 'construction' in status_lower or 'under construction' in status_lower:
            return 'construction'
        elif 'announced' in status_lower or 'planned' in status_lower or 'proposed' in status_lower:
            return 'announced'
        else:
            return 'other'

    def run_monte_carlo_simulation(self, start_year: int = None, end_year: int = 2050) -> List[pd.DataFrame]:
        """
        Run Monte Carlo simulations for projections

        Args:
            start_year: Starting year for projection
            end_year: Ending year for projection

        Returns:
            List of DataFrames, one per simulation
        """
        if self.df is None:
            print("[ERROR] No data loaded.")
            return None

        if start_year is None:
            start_year = self.current_year

        print("\n" + "="*80)
        print(f"RUNNING MONTE CARLO SIMULATIONS ({self.n_simulations} iterations)")
        print("="*80)
        print(f"Scenario: {self.scenario.upper()}")
        print(f"Year range: {start_year} - {end_year}")

        self.simulations = []

        for sim_idx in range(self.n_simulations):
            if (sim_idx + 1) % 100 == 0 or sim_idx == 0:
                print(f"  Running simulation {sim_idx + 1}/{self.n_simulations}...")

            # Create simulated dataset for this iteration
            sim_df = self._simulate_plant_parameters(self.df.copy())

            # Project generation for this simulation
            sim_projection = self._project_single_simulation(sim_df, start_year, end_year)

            self.simulations.append(sim_projection)

        print(f"\n[OK] Completed {self.n_simulations} simulations")
        return self.simulations

    def _simulate_plant_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate plant parameters with uncertainty for a single iteration

        Args:
            df: DataFrame with plant data

        Returns:
            DataFrame with simulated parameters
        """
        sim_df = df.copy()

        for idx, row in sim_df.iterrows():
            status = row['status_category']

            # 1. Simulate completion (for pipeline plants)
            if status in ['construction', 'announced']:
                completion_prob = self.COMPLETION_PROBABILITIES[status][self.scenario]
                completed = np.random.random() < completion_prob

                if not completed:
                    # Plant doesn't get built in this simulation
                    sim_df.at[idx, 'completed'] = False
                    continue
                else:
                    sim_df.at[idx, 'completed'] = True

            else:
                sim_df.at[idx, 'completed'] = True

            # 2. Simulate construction delays (for pipeline plants)
            if status in ['construction', 'announced'] and not pd.isna(row['start_year_extracted']):
                delay_params = self.CONSTRUCTION_DELAYS[self.scenario]
                delay = np.random.normal(delay_params['mean'], delay_params['std'])
                delay = max(-2, delay)  # Don't allow more than 2 years early

                sim_df.at[idx, 'simulated_start_year'] = row['start_year_extracted'] + delay
            else:
                sim_df.at[idx, 'simulated_start_year'] = row['start_year_extracted']

            # 3. Simulate lifetime extension (for operating plants)
            if status == 'operating' and not pd.isna(row['retirement_year_extracted']):
                if np.random.random() < self.LIFETIME_EXTENSION['probability']:
                    extension = self.LIFETIME_EXTENSION['extension_years']
                    sim_df.at[idx, 'simulated_retirement_year'] = row['retirement_year_extracted'] + extension
                else:
                    sim_df.at[idx, 'simulated_retirement_year'] = row['retirement_year_extracted']
            else:
                sim_df.at[idx, 'simulated_retirement_year'] = row['retirement_year_extracted']

            # 4. Simulate capacity factor
            if status in self.CAPACITY_FACTORS:
                cf_params = self.CAPACITY_FACTORS[status]
                cf = np.random.normal(cf_params['mean'], cf_params['std'])
                cf = np.clip(cf, 0.5, 0.95)  # Reasonable bounds
                sim_df.at[idx, 'simulated_capacity_factor'] = cf
            else:
                sim_df.at[idx, 'simulated_capacity_factor'] = 0.85

        return sim_df

    def _project_single_simulation(self, sim_df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Project generation for a single simulation

        Args:
            sim_df: Simulated plant parameters
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with projections by region and year
        """
        years = range(start_year, end_year + 1)
        regions = sim_df['Region'].dropna().unique()

        projections_data = []

        for region in regions:
            region_plants = sim_df[(sim_df['Region'] == region) & (sim_df['completed'] == True)].copy()

            for year in years:
                # Filter plants operating in this year
                operating_plants = region_plants[
                    (region_plants['simulated_start_year'] <= year) &
                    (
                        (region_plants['simulated_retirement_year'] > year) |
                        (region_plants['simulated_retirement_year'].isna())
                    )
                ].copy()

                # Calculate generation
                operating_plants['generation_twh'] = (
                    operating_plants['capacity_numeric'] *
                    8760 *
                    operating_plants['simulated_capacity_factor'] /
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

        return pd.DataFrame(projections_data)

    def aggregate_simulations(self) -> pd.DataFrame:
        """
        Aggregate Monte Carlo simulations to get statistics

        Returns:
            DataFrame with mean, P10, P50, P90 projections
        """
        if not self.simulations:
            print("[ERROR] No simulations run yet.")
            return None

        print("\n" + "="*80)
        print("AGGREGATING SIMULATION RESULTS")
        print("="*80)

        # Combine all simulations
        all_sims = pd.concat([
            df.assign(simulation_id=i) for i, df in enumerate(self.simulations)
        ])

        # Calculate statistics for each year-region combination
        stats = all_sims.groupby(['year', 'region']).agg({
            'generation_twh': ['mean', lambda x: np.percentile(x, 10),
                              lambda x: np.percentile(x, 50),
                              lambda x: np.percentile(x, 90), 'std'],
            'capacity_gw': 'mean',
            'unit_count': 'mean'
        }).reset_index()

        # Flatten column names
        stats.columns = ['year', 'region', 'generation_mean', 'generation_p10',
                        'generation_p50', 'generation_p90', 'generation_std',
                        'capacity_mean', 'unit_count_mean']

        self.projections = stats

        print(f"\n[OK] Aggregated {self.n_simulations} simulations")
        print(f"  Result includes mean, P10, P50, P90, and std for each year-region")

        return stats

    def analyze_2050_projections(self):
        """Analyze 2050 projections with uncertainty"""
        if self.projections is None:
            print("[ERROR] No aggregated projections available.")
            return None

        print("\n" + "="*80)
        print(f"2050 PROJECTIONS ({self.scenario.upper()} SCENARIO)")
        print("="*80)

        final_year = self.projections[self.projections['year'] == 2050].copy()
        final_year = final_year.sort_values('generation_mean', ascending=False)

        print(f"\n{'Region':<35s} {'Mean (TWh)':>15s} {'P10 (TWh)':>15s} {'P90 (TWh)':>15s} {'Uncertainty':>15s}")
        print("-" * 100)

        for _, row in final_year.iterrows():
            uncertainty = row['generation_p90'] - row['generation_p10']
            print(f"{row['region']:<35s} {row['generation_mean']:>15,.1f} "
                  f"{row['generation_p10']:>15,.1f} {row['generation_p90']:>15,.1f} "
                  f"{uncertainty:>15,.1f}")

        # Global totals
        global_mean = final_year['generation_mean'].sum()
        global_p10 = final_year['generation_p10'].sum()
        global_p90 = final_year['generation_p90'].sum()

        print("-" * 100)
        print(f"{'GLOBAL TOTAL':<35s} {global_mean:>15,.1f} {global_p10:>15,.1f} {global_p90:>15,.1f}")

        print(f"\n[SUMMARY] 2050 Global Generation:")
        print(f"  Mean projection: {global_mean:,.1f} TWh")
        print(f"  10th percentile: {global_p10:,.1f} TWh (pessimistic)")
        print(f"  90th percentile: {global_p90:,.1f} TWh (optimistic)")
        print(f"  Uncertainty range: {global_p90 - global_p10:,.1f} TWh")

        return final_year

    def compare_scenarios(self, other_models: List['EnhancedNuclearProjectionModel']) -> pd.DataFrame:
        """
        Compare this scenario against other scenarios

        Args:
            other_models: List of other projection models (different scenarios)

        Returns:
            DataFrame with scenario comparison
        """
        print("\n" + "="*80)
        print("SCENARIO COMPARISON (2050)")
        print("="*80)

        scenarios = [self] + other_models
        comparison_data = []

        for model in scenarios:
            if model.projections is None:
                print(f"[WARNING] Model {model.scenario} has no projections")
                continue

            final_year = model.projections[model.projections['year'] == 2050]
            global_mean = final_year['generation_mean'].sum()
            global_p10 = final_year['generation_p10'].sum()
            global_p90 = final_year['generation_p90'].sum()

            comparison_data.append({
                'scenario': model.scenario,
                'generation_mean': global_mean,
                'generation_p10': global_p10,
                'generation_p50': final_year['generation_p50'].sum(),
                'generation_p90': global_p90,
                'uncertainty_range': global_p90 - global_p10
            })

        comparison_df = pd.DataFrame(comparison_data)

        print(f"\n{'Scenario':<20s} {'Mean (TWh)':>15s} {'P10 (TWh)':>15s} {'P90 (TWh)':>15s} {'Range (TWh)':>15s}")
        print("-" * 90)

        for _, row in comparison_df.iterrows():
            print(f"{row['scenario'].upper():<20s} {row['generation_mean']:>15,.1f} "
                  f"{row['generation_p10']:>15,.1f} {row['generation_p90']:>15,.1f} "
                  f"{row['uncertainty_range']:>15,.1f}")

        return comparison_df

    def sensitivity_analysis(self) -> pd.DataFrame:
        """
        Perform sensitivity analysis on key parameters

        Returns:
            DataFrame with sensitivity results
        """
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS")
        print("="*80)

        # Parameters to test
        sensitivity_params = {
            'completion_rate': [0.5, 0.7, 0.9],
            'construction_delay': [0, 2, 5],
            'capacity_factor': [0.80, 0.85, 0.90],
            'lifetime_extension_prob': [0.5, 0.7, 0.9]
        }

        # TODO: Implement full sensitivity analysis
        # This would involve running simulations with varying parameters
        # and measuring the impact on 2050 projections

        print("\n[NOTE] Full sensitivity analysis not yet implemented")
        print("  Use compare_scenarios() to compare Base, Conservative, Aggressive scenarios")

        return None

    def save_projections(self, output_path: str = None):
        """
        Save probabilistic projections to CSV

        Args:
            output_path: Path to save projections
        """
        if self.projections is None:
            print("[ERROR] No projections to save.")
            return

        if output_path is None:
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / f"nuclear_projections_2050_{self.scenario}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.projections.to_csv(output_path, index=False)
        print(f"\n[OK] Projections saved to: {output_path.name}")
        print(f"  {len(self.projections):,} rows (region × year combinations)")


def run_all_scenarios():
    """Run projections for all three scenarios"""
    print("="*80)
    print("ENHANCED NUCLEAR PROJECTION MODEL - ALL SCENARIOS")
    print("="*80)

    scenarios = ['base', 'conservative', 'aggressive']
    models = []

    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"RUNNING {scenario.upper()} SCENARIO")
        print(f"{'='*80}")

        model = EnhancedNuclearProjectionModel(scenario=scenario, n_simulations=1000)
        model.load_data()
        model.prepare_projection_data()
        model.run_monte_carlo_simulation(start_year=2025, end_year=2050)
        model.aggregate_simulations()
        model.analyze_2050_projections()
        model.save_projections()

        models.append(model)

    # Compare scenarios
    print(f"\n{'='*80}")
    print("COMPARING ALL SCENARIOS")
    print(f"{'='*80}")

    base_model = models[0]
    other_models = models[1:]
    comparison = base_model.compare_scenarios(other_models)

    # Save comparison
    project_root = Path(__file__).parent.parent
    comparison_path = project_root / "data" / "processed" / "scenario_comparison_enhanced.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n[OK] Scenario comparison saved to: {comparison_path.name}")

    return models, comparison


def main():
    """Main execution function"""
    models, comparison = run_all_scenarios()

    print("\n" + "="*80)
    print("ENHANCED PROJECTION MODEL COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - nuclear_projections_2050_base.csv")
    print("  - nuclear_projections_2050_conservative.csv")
    print("  - nuclear_projections_2050_aggressive.csv")
    print("  - scenario_comparison_enhanced.csv")

    print("\nKey features:")
    print("  - Monte Carlo simulations (1000 iterations per scenario)")
    print("  - Probabilistic construction delays")
    print("  - Plant completion uncertainty")
    print("  - Lifetime extension modeling")
    print("  - Uncertainty quantification (P10, P50, P90)")


if __name__ == "__main__":
    main()
