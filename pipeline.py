"""
Nuclear Energy Projection Pipeline - Main Orchestrator

This script runs the complete end-to-end pipeline for nuclear energy projection
and avoided emissions analysis.

Usage:
    python pipeline.py --all                    # Run all steps
    python pipeline.py --steps ingestion,features    # Run specific steps
    python pipeline.py --all --clean            # Clean outputs and rerun
    python pipeline.py --all --skip-if-exists   # Skip steps with existing outputs
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import time

# Import pipeline modules
from src.data_ingestion import NuclearDataLoader
from src.feature_engineering import NuclearFeatureEngineer
from src.projection_model import NuclearProjectionModel
from src.scenario_comparison import ScenarioComparison
from src.emissions_calculator import EmissionsCalculator


class PipelineOrchestrator:
    """Orchestrates the complete nuclear energy projection pipeline"""

    PIPELINE_STEPS = {
        'ingestion': {
            'name': 'Data Ingestion',
            'module': 'data_ingestion',
            'output': 'data/processed/nuclear_tracker_cleaned.csv',
            'description': 'Load and clean Global Nuclear Power Tracker data'
        },
        'features': {
            'name': 'Feature Engineering',
            'module': 'feature_engineering',
            'output': 'data/processed/nuclear_features.csv',
            'description': 'Calculate capacity factors and aggregations'
        },
        'projection': {
            'name': 'Projection Model',
            'module': 'projection_model',
            'output': 'data/processed/nuclear_projections_2050.csv',
            'description': 'Project nuclear generation 2025-2050'
        },
        'scenario': {
            'name': 'Scenario Comparison',
            'module': 'scenario_comparison',
            'output': 'data/processed/scenario_comparison.csv',
            'description': 'Compare against IEA Net Zero scenarios'
        },
        'emissions': {
            'name': 'Emissions Calculator',
            'module': 'emissions_calculator',
            'output': 'data/processed/avoided_emissions.csv',
            'description': 'Calculate avoided CO2 emissions'
        }
    }

    def __init__(self, log_level=logging.INFO):
        """
        Initialize pipeline orchestrator

        Args:
            log_level: Logging level (default: INFO)
        """
        self.project_root = Path(__file__).parent
        self.setup_logging(log_level)
        self.results = {}

    def setup_logging(self, log_level):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline log file: {log_file}")

    def check_dependencies(self):
        """Check if required data files exist"""
        self.logger.info("\n" + "="*80)
        self.logger.info("CHECKING PIPELINE DEPENDENCIES")
        self.logger.info("="*80)

        required_files = [
            self.project_root / "data" / "raw" / "Global-Nuclear-Power-Tracker-September-2025.xlsx",
            self.project_root / "data" / "raw" / "NZE2021_AnnexA.csv"
        ]

        missing_files = []
        for file_path in required_files:
            if file_path.exists():
                self.logger.info(f"[OK] Found: {file_path.name}")
            else:
                self.logger.error(f"[ERROR] Missing: {file_path}")
                missing_files.append(file_path)

        if missing_files:
            self.logger.error("\nERROR: Missing required data files!")
            self.logger.error("Please ensure all raw data files are in data/raw/")
            return False

        self.logger.info("\n[OK] All dependencies satisfied")
        return True

    def clean_outputs(self, steps=None):
        """
        Clean output files from previous runs

        Args:
            steps: List of steps to clean (None = clean all)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("CLEANING OUTPUT FILES")
        self.logger.info("="*80)

        steps_to_clean = steps if steps else self.PIPELINE_STEPS.keys()

        for step_name in steps_to_clean:
            if step_name not in self.PIPELINE_STEPS:
                continue

            output_file = self.project_root / self.PIPELINE_STEPS[step_name]['output']
            if output_file.exists():
                output_file.unlink()
                self.logger.info(f"[OK] Deleted: {output_file.name}")

        # Also clean aggregation files
        processed_dir = self.project_root / "data" / "processed"
        if processed_dir.exists():
            for csv_file in processed_dir.glob("*.csv"):
                if csv_file.exists():
                    csv_file.unlink()
                    self.logger.info(f"[OK] Deleted: {csv_file.name}")

        self.logger.info("\n[OK] Cleanup complete")

    def check_step_completed(self, step_name):
        """
        Check if a step has already been completed

        Args:
            step_name: Name of the step

        Returns:
            bool: True if output file exists
        """
        output_file = self.project_root / self.PIPELINE_STEPS[step_name]['output']
        return output_file.exists()

    def run_step(self, step_name, skip_if_exists=False):
        """
        Run a single pipeline step

        Args:
            step_name: Name of the step to run
            skip_if_exists: Skip if output already exists

        Returns:
            bool: True if successful
        """
        if step_name not in self.PIPELINE_STEPS:
            self.logger.error(f"Unknown step: {step_name}")
            return False

        step_info = self.PIPELINE_STEPS[step_name]

        self.logger.info("\n" + "="*80)
        self.logger.info(f"STEP: {step_info['name'].upper()}")
        self.logger.info("="*80)
        self.logger.info(f"Description: {step_info['description']}")

        # Check if already completed
        if skip_if_exists and self.check_step_completed(step_name):
            self.logger.info(f"[SKIP] Output already exists: {step_info['output']}")
            self.logger.info("  Skipping step (use --clean to force rerun)")
            self.results[step_name] = {'status': 'skipped', 'time': 0}
            return True

        start_time = time.time()

        try:
            # Run the appropriate step
            if step_name == 'ingestion':
                success = self._run_ingestion()
            elif step_name == 'features':
                success = self._run_features()
            elif step_name == 'projection':
                success = self._run_projection()
            elif step_name == 'scenario':
                success = self._run_scenario()
            elif step_name == 'emissions':
                success = self._run_emissions()
            else:
                success = False

            elapsed_time = time.time() - start_time

            if success:
                self.logger.info(f"\n[SUCCESS] {step_info['name']} completed in {elapsed_time:.2f}s")
                self.results[step_name] = {'status': 'success', 'time': elapsed_time}
                return True
            else:
                self.logger.error(f"\n[FAILED] {step_info['name']} failed")
                self.results[step_name] = {'status': 'failed', 'time': elapsed_time}
                return False

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"\n[ERROR] {step_info['name']} failed with exception: {e}")
            self.logger.exception("Full traceback:")
            self.results[step_name] = {'status': 'error', 'time': elapsed_time, 'error': str(e)}
            return False

    def _run_ingestion(self):
        """Run data ingestion step"""
        loader = NuclearDataLoader()
        sheets = loader.load_excel_sheets()
        if sheets is None:
            return False

        loader.load_data(sheet_name=0)
        loader.explore_structure()
        loader.identify_key_columns()
        loader.analyze_status_distribution()
        loader.analyze_capacity()
        loader.analyze_regional_distribution()
        loader.save_processed_data()
        return True

    def _run_features(self):
        """Run feature engineering step"""
        engineer = NuclearFeatureEngineer()
        engineer.load_data()
        engineer.calculate_capacity_factors()
        engineer.calculate_annual_generation()

        # Aggregations
        regional = engineer.aggregate_by_region()
        subregional = engineer.aggregate_by_subregion()
        country = engineer.aggregate_by_country()
        status = engineer.aggregate_by_status()

        # Create and save features
        engineer.create_feature_dataset()
        engineer.save_features()

        # Save aggregations
        output_dir = self.project_root / "data" / "processed"
        regional.to_csv(output_dir / "regional_aggregations.csv")
        subregional.to_csv(output_dir / "subregional_aggregations.csv")
        country.to_csv(output_dir / "country_aggregations.csv")
        status.to_csv(output_dir / "status_aggregations.csv")

        return True

    def _run_projection(self):
        """Run projection model step"""
        model = NuclearProjectionModel()
        model.load_data()
        model.prepare_projection_data()

        # Project generation
        regional_proj = model.project_regional_generation(start_year=2025, end_year=2050)
        global_proj = model.project_global_generation(end_year=2050)
        pipeline = model.analyze_pipeline_contribution(target_year=2050)

        # Save results
        model.save_projections()
        output_dir = self.project_root / "data" / "processed"
        global_proj.to_csv(output_dir / "global_projections_2050.csv", index=False)
        pipeline.to_csv(output_dir / "pipeline_contribution_2050.csv", index=False)

        return True

    def _run_scenario(self):
        """Run scenario comparison step"""
        comparator = ScenarioComparison()
        comparator.load_projections()
        comparator.load_iea_scenarios()

        # Compare scenarios
        comparison = comparator.compare_scenarios()
        global_gap = comparator.analyze_global_gap()
        regional_gaps = comparator.analyze_regional_gaps(target_year=2050)
        requirements = comparator.calculate_capacity_requirements(target_year=2050)

        # Save results
        comparator.save_comparison()
        output_dir = self.project_root / "data" / "processed"
        global_gap.to_csv(output_dir / "global_gap_analysis.csv", index=False)
        regional_gaps.to_csv(output_dir / "regional_gaps_2050.csv", index=False)

        return True

    def _run_emissions(self):
        """Run emissions calculator step"""
        calculator = EmissionsCalculator()
        calculator.load_projections()

        # Calculate emissions
        emissions = calculator.calculate_avoided_emissions(baseline='regional')
        global_emissions = calculator.analyze_global_avoided_emissions()
        regional_2050 = calculator.analyze_regional_avoided_emissions(target_year=2050)
        cumulative_regional = calculator.calculate_cumulative_regional_emissions(
            start_year=2025,
            end_year=2050
        )
        baseline_comparison = calculator.compare_baseline_scenarios(year=2050)

        # Save results
        calculator.save_emissions_data()
        output_dir = self.project_root / "data" / "processed"
        global_emissions.to_csv(output_dir / "global_avoided_emissions.csv", index=False)
        cumulative_regional.to_csv(output_dir / "cumulative_regional_emissions.csv", index=False)
        baseline_comparison.to_csv(output_dir / "baseline_scenario_comparison.csv", index=False)

        return True

    def run_pipeline(self, steps=None, skip_if_exists=False):
        """
        Run the complete pipeline or specific steps

        Args:
            steps: List of steps to run (None = run all)
            skip_if_exists: Skip steps with existing outputs

        Returns:
            bool: True if all steps successful
        """
        steps_to_run = steps if steps else list(self.PIPELINE_STEPS.keys())

        # Validate steps
        invalid_steps = [s for s in steps_to_run if s not in self.PIPELINE_STEPS]
        if invalid_steps:
            self.logger.error(f"Invalid steps: {', '.join(invalid_steps)}")
            self.logger.error(f"Valid steps: {', '.join(self.PIPELINE_STEPS.keys())}")
            return False

        self.logger.info("\n" + "="*80)
        self.logger.info("NUCLEAR ENERGY PROJECTION PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Steps to run: {', '.join(steps_to_run)}")
        self.logger.info(f"Skip if exists: {skip_if_exists}")

        pipeline_start = time.time()

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Run each step
        all_success = True
        for step_name in steps_to_run:
            success = self.run_step(step_name, skip_if_exists)
            if not success:
                all_success = False
                self.logger.error(f"\nPipeline failed at step: {step_name}")
                break

        pipeline_time = time.time() - pipeline_start

        # Print summary
        self.print_summary(pipeline_time)

        return all_success

    def print_summary(self, total_time):
        """Print pipeline execution summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*80)

        for step_name, result in self.results.items():
            status = result['status']
            elapsed = result['time']

            if status == 'success':
                icon = "[OK]"
                status_text = "SUCCESS"
            elif status == 'skipped':
                icon = "[--]"
                status_text = "SKIPPED"
            elif status == 'failed':
                icon = "[!!]"
                status_text = "FAILED"
            else:
                icon = "[!!]"
                status_text = "ERROR"

            step_info = self.PIPELINE_STEPS[step_name]
            self.logger.info(f"{icon} {step_info['name']:<30s} {status_text:<10s} ({elapsed:.2f}s)")

        self.logger.info("-" * 80)
        self.logger.info(f"Total execution time: {total_time:.2f}s")

        # Overall status
        success_count = sum(1 for r in self.results.values() if r['status'] == 'success')
        total_count = len(self.results)

        if success_count == total_count:
            self.logger.info("\n[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            self.logger.error(f"\n[FAILED] PIPELINE FAILED ({success_count}/{total_count} steps successful)")

        # Print key output files
        self.logger.info("\n" + "="*80)
        self.logger.info("OUTPUT FILES")
        self.logger.info("="*80)

        processed_dir = self.project_root / "data" / "processed"
        if processed_dir.exists():
            csv_files = sorted(processed_dir.glob("*.csv"))
            for csv_file in csv_files:
                size_kb = csv_file.stat().st_size / 1024
                self.logger.info(f"  - {csv_file.name:<45s} ({size_kb:>8,.1f} KB)")


def main():
    """Main entry point for pipeline execution"""
    parser = argparse.ArgumentParser(
        description="Nuclear Energy Projection Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --all                    # Run all steps
  python pipeline.py --steps ingestion        # Run only data ingestion
  python pipeline.py --steps projection,emissions  # Run specific steps
  python pipeline.py --all --clean            # Clean and rerun everything
  python pipeline.py --all --skip-if-exists   # Skip completed steps

Available steps:
  ingestion   - Load and clean nuclear tracker data
  features    - Calculate capacity factors and aggregations
  projection  - Project nuclear generation to 2050
  scenario    - Compare against IEA Net Zero scenarios
  emissions   - Calculate avoided CO2 emissions
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all pipeline steps'
    )

    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated list of steps to run (e.g., ingestion,features)'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean output files before running'
    )

    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip steps that have already been completed'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Determine log level
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(log_level=log_level)

    # Determine which steps to run
    if args.all:
        steps = None  # Run all steps
    elif args.steps:
        steps = [s.strip() for s in args.steps.split(',')]
    else:
        # No steps specified, show help
        parser.print_help()
        return 1

    # Clean outputs if requested
    if args.clean:
        orchestrator.clean_outputs(steps)

    # Run pipeline
    success = orchestrator.run_pipeline(steps=steps, skip_if_exists=args.skip_if_exists)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
