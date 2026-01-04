"""
Data Ingestion Module for Global Nuclear Power Tracker

This module loads and performs initial exploration of the nuclear power tracker dataset.
It identifies key columns, data structure, and prepares data for further analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

# Configure stdout to handle Unicode characters on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')


class NuclearDataLoader:
    """Load and explore Global Nuclear Power Tracker data"""

    def __init__(self, data_path: str = None):
        """
        Initialize the data loader

        Args:
            data_path: Path to the Excel file. If None, uses default location.
        """
        if data_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / "data" / "raw" / "Global-Nuclear-Power-Tracker-September-2025.xlsx"
        else:
            self.data_path = Path(data_path)

        self.df = None
        self.sheet_names = None

    def load_excel_sheets(self):
        """Load and display all available sheets in the Excel file"""
        try:
            excel_file = pd.ExcelFile(self.data_path)
            self.sheet_names = excel_file.sheet_names
            print(f"[INFO] Found {len(self.sheet_names)} sheets in the Excel file:")
            for i, sheet in enumerate(self.sheet_names, 1):
                print(f"  {i}. {sheet}")
            return self.sheet_names
        except FileNotFoundError:
            print(f"ERROR - Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"ERROR - Error loading Excel file: {e}")
            return None

    def load_data(self, sheet_name: str = 0):
        """
        Load data from specified sheet

        Args:
            sheet_name: Name or index of sheet to load (default: 0 for first sheet)
        """
        try:
            self.df = pd.read_excel(self.data_path, sheet_name=sheet_name)
            print(f"OK - Loaded data from sheet: {sheet_name if isinstance(sheet_name, str) else self.sheet_names[sheet_name]}")
            print(f"   Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"ERROR - Error loading sheet: {e}")
            return None

    def explore_structure(self):
        """Explore the structure of the loaded data"""
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return

        print("\n" + "="*80)
        print("DATA STRUCTURE EXPLORATION")
        print("="*80)

        # Basic info
        print(f"\n[INFO] Dataset Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")

        # Column names and types
        print(f"\n[INFO] Column Names and Data Types:")
        print("-" * 80)
        for i, (col, dtype) in enumerate(zip(self.df.columns, self.df.dtypes), 1):
            null_count = self.df[col].isna().sum()
            null_pct = (null_count / len(self.df)) * 100
            print(f"{i:3}. {col:50s} | {str(dtype):15s} | {null_pct:5.1f}% null")

        # Preview data
        print(f"\n[PREVIEW] First 3 rows:")
        print("-" * 80)
        print(self.df.head(3).to_string())

        return self.df.dtypes

    def identify_key_columns(self):
        """Identify key columns for nuclear projection analysis"""
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return

        print("\n" + "="*80)
        print("KEY COLUMNS IDENTIFICATION")
        print("="*80)

        # Look for common column patterns
        key_patterns = {
            'Status': ['status', 'project status', 'plant status'],
            'Capacity': ['capacity', 'mw', 'megawatt', 'power'],
            'Location': ['country', 'region', 'location', 'province', 'state'],
            'Timeline': ['year', 'date', 'operational', 'construction', 'announced'],
            'Plant Info': ['plant', 'reactor', 'unit', 'name', 'project'],
            'Owner': ['owner', 'developer', 'operator', 'company'],
        }

        columns_lower = [col.lower() for col in self.df.columns]

        for category, patterns in key_patterns.items():
            print(f"\n{category} columns:")
            found = []
            for col, col_lower in zip(self.df.columns, columns_lower):
                if any(pattern in col_lower for pattern in patterns):
                    found.append(col)
            if found:
                for col in found:
                    print(f"  [x] {col}")
            else:
                print(f"  WARNING -  No columns found matching: {', '.join(patterns)}")

    def analyze_status_distribution(self, status_col: str = None):
        """
        Analyze the distribution of plant statuses

        Args:
            status_col: Name of the status column. If None, will try to auto-detect.
        """
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return

        # Auto-detect status column
        if status_col is None:
            status_keywords = ['status', 'project status', 'plant status']
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in status_keywords):
                    status_col = col
                    break

        if status_col is None or status_col not in self.df.columns:
            print(f"WARNING -  Could not find status column. Available columns:")
            print(f"   {', '.join(self.df.columns[:10])}...")
            return

        print(f"\n" + "="*80)
        print(f"STATUS DISTRIBUTION (Column: '{status_col}')")
        print("="*80)

        status_counts = self.df[status_col].value_counts()
        total = len(self.df)

        for status, count in status_counts.items():
            pct = (count / total) * 100
            print(f"  {status:40s}: {count:6,} ({pct:5.1f}%)")

        # Identify pipeline projects (announced, under construction)
        pipeline_keywords = ['announced', 'construction', 'planned', 'proposed', 'under construction']
        pipeline_count = 0

        print(f"\n[ANALYSIS] Pipeline Analysis (Future Projects):")
        for status, count in status_counts.items():
            if any(keyword in str(status).lower() for keyword in pipeline_keywords):
                pipeline_count += count
                print(f"  - {status}: {count:,}")

        print(f"\n[SUMMARY] Total Pipeline Projects: {pipeline_count:,} ({(pipeline_count/total)*100:.1f}%)")

        return status_counts

    def analyze_capacity(self, capacity_col: str = None):
        """
        Analyze capacity statistics

        Args:
            capacity_col: Name of the capacity column. If None, will try to auto-detect.
        """
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return

        # Auto-detect capacity column
        if capacity_col is None:
            capacity_keywords = ['capacity', 'mw', 'megawatt']
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in capacity_keywords):
                    capacity_col = col
                    break

        if capacity_col is None or capacity_col not in self.df.columns:
            print(f"WARNING -  Could not find capacity column.")
            return

        print(f"\n" + "="*80)
        print(f"CAPACITY ANALYSIS (Column: '{capacity_col}')")
        print("="*80)

        capacity_data = pd.to_numeric(self.df[capacity_col], errors='coerce')

        print(f"\n  Total Capacity: {capacity_data.sum():,.0f} MW ({capacity_data.sum()/1000:.1f} GW)")
        print(f"  Mean Capacity:  {capacity_data.mean():,.0f} MW")
        print(f"  Median Capacity: {capacity_data.median():,.0f} MW")
        print(f"  Min Capacity:   {capacity_data.min():,.0f} MW")
        print(f"  Max Capacity:   {capacity_data.max():,.0f} MW")
        print(f"  Missing Values: {capacity_data.isna().sum():,} ({(capacity_data.isna().sum()/len(capacity_data))*100:.1f}%)")

        return capacity_data.describe()

    def analyze_regional_distribution(self, country_col: str = None):
        """
        Analyze regional/country distribution

        Args:
            country_col: Name of the country column. If None, will try to auto-detect.
        """
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return

        # Auto-detect country column
        if country_col is None:
            location_keywords = ['country', 'nation', 'region']
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in location_keywords):
                    country_col = col
                    break

        if country_col is None or country_col not in self.df.columns:
            print(f"WARNING -  Could not find country/region column.")
            return

        print(f"\n" + "="*80)
        print(f"REGIONAL DISTRIBUTION (Column: '{country_col}')")
        print("="*80)

        country_counts = self.df[country_col].value_counts()
        total = len(self.df)

        print(f"\nTop 15 Countries by Number of Projects:")
        for i, (country, count) in enumerate(country_counts.head(15).items(), 1):
            pct = (count / total) * 100
            print(f"  {i:2}. {country:30s}: {count:4,} ({pct:5.1f}%)")

        return country_counts

    def save_processed_data(self, output_path: str = None):
        """
        Save processed data to CSV

        Args:
            output_path: Path to save the CSV. If None, uses default location.
        """
        if self.df is None:
            print("WARNING -  No data loaded. Call load_data() first.")
            return

        if output_path is None:
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / "nuclear_tracker_cleaned.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        print(f"\nOK - Data saved to: {output_path}")
        print(f"   {len(self.df):,} rows × {len(self.df.columns)} columns")


def main():
    """Main execution function for data exploration"""
    print("="*80)
    print("GLOBAL NUCLEAR POWER TRACKER - DATA EXPLORATION")
    print("="*80)

    # Initialize loader
    loader = NuclearDataLoader()

    # Load sheets
    print("\n[1] Loading Excel file...")
    sheets = loader.load_excel_sheets()

    if sheets is None:
        return

    # Load first sheet (typically contains main data)
    print("\n[2] Loading data from first sheet...")
    loader.load_data(sheet_name=0)

    # Explore structure
    print("\n[3] Exploring data structure...")
    loader.explore_structure()

    # Identify key columns
    print("\n[4] Identifying key columns...")
    loader.identify_key_columns()

    # Analyze status distribution
    print("\n[5] Analyzing status distribution...")
    loader.analyze_status_distribution()

    # Analyze capacity
    print("\n[6] Analyzing capacity statistics...")
    loader.analyze_capacity()

    # Analyze regional distribution
    print("\n[7] Analyzing regional distribution...")
    loader.analyze_regional_distribution()

    # Save processed data
    print("\n[8] Saving processed data...")
    loader.save_processed_data()

    print("\n" + "="*80)
    print("OK - DATA EXPLORATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the output above to understand the data structure")
    print("  2. Identify the exact column names for status, capacity, timeline, etc.")
    print("  3. Create feature engineering script to prepare data for modeling")
    print("  4. Source IEA Net Zero scenario data for comparison")


if __name__ == "__main__":
    main()
