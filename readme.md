# Nuclear Energy Generation Projection & Avoided Emissions Analysis

## Project Overview

This project builds an end-to-end automated modeling pipeline that uses the Global Nuclear Power Tracker to project regional nuclear energy generation until 2050. By comparing the current pipeline (announced, under-construction) against IEA Net Zero scenarios, we identify "Generation Gaps" and quantify "Avoided Emissions" if the nuclear pipeline is successfully deployed.

**Key Features**:
- Automated pipeline orchestration with error handling
- Probabilistic projection modeling with uncertainty quantification
- Comprehensive statistical analysis
- Interactive Streamlit dashboard for data exploration
- Reproducible workflow with version control

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
# Run all steps
python pipeline.py --all

# Or run specific steps
python pipeline.py --steps ingestion features projection
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## Project Structure

```
research-final-proj/
├── data/
│   ├── raw/                              # Original data sources
│   │   ├── Global-Nuclear-Power-Tracker-September-2025.xlsx
│   │   └── NZE2021_AnnexA.csv
│   ├── processed/                        # Pipeline outputs
│   │   ├── nuclear_tracker_cleaned.csv
│   │   ├── nuclear_features.csv
│   │   ├── nuclear_projections_2050.csv
│   │   ├── scenario_comparison.csv
│   │   ├── avoided_emissions.csv
│   │   └── [17 other CSV files...]
│   └── external/
│       └── IEA_DATA_REQUIREMENTS.md
├── src/
│   ├── data_ingestion.py                 # Load & clean nuclear tracker
│   ├── feature_engineering.py            # Calculate capacity factors, aggregations
│   ├── projection_model.py               # Deterministic projection model (v1)
│   ├── projection_model_v2.py            # Enhanced probabilistic model with Monte Carlo
│   ├── scenario_comparison.py            # Compare vs IEA Net Zero scenarios
│   └── emissions_calculator.py           # Calculate avoided CO2 emissions
├── notebooks/
│   ├── 01_exploratory_analysis_executed.ipynb
│   └── 02_statistical_analysis.ipynb     # Statistical tests & analysis
├── outputs/
│   ├── figures/                          # Generated visualizations
│   └── results/                          # Statistical summaries
├── logs/                                 # Pipeline execution logs
├── pipeline.py                           # Main pipeline orchestrator
├── app.py                                # Streamlit interactive dashboard
├── main.py                               # Legacy runner (use pipeline.py instead)
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # Project configuration
├── guidelines.txt                        # Assignment guidelines
└── readme.md                             # This file
```

---

## Pipeline Components

### 1. Data Ingestion (`src/data_ingestion.py`)
- Loads Global Nuclear Power Tracker Excel file
- Cleans and standardizes data
- Handles missing values and outliers
- **Output**: `nuclear_tracker_cleaned.csv`

### 2. Feature Engineering (`src/feature_engineering.py`)
- Calculates capacity factors (Operating: 0.90, Pipeline: 0.85)
- Computes annual generation: `Generation = Capacity × 8760 × CF / 1,000,000`
- Creates regional, country, and status aggregations
- **Outputs**: `nuclear_features.csv`, aggregation files

### 3. Projection Modeling
#### Original Model (`src/projection_model.py`)
- Deterministic projections based on announced start dates
- Fixed capacity factors
- **Output**: `nuclear_projections_2050.csv`

#### Enhanced Model v2 (`src/projection_model_v2.py`)
- **Monte Carlo simulation** (1000 iterations)
- **Probabilistic parameters**:
  - Construction delays with scenario-specific distributions
  - Plant completion probabilities (Operating: 100%, Construction: 95%, Announced: 60%)
  - Variable capacity factors (mean ± std deviation)
  - Lifetime extensions (70% probability)
- **Three scenarios**: Base, Conservative, Aggressive
- **Uncertainty quantification**: P10, P50, P90 percentiles
- **Output**: `nuclear_projections_2050_base.csv` (and conservative/aggressive variants)

### 4. Scenario Comparison (`src/scenario_comparison.py`)
- Compares projections against IEA Net Zero 2050 targets
- Calculates generation gaps (TWh) by region
- Classifies alignment status:
  - On track (gap < 5%)
  - Shortfall (5-20% gap)
  - Major shortfall (>20% gap)
  - Exceeds target (negative gap)
- **Outputs**: `scenario_comparison.csv`, `global_gap_analysis.csv`, `regional_gaps_2050.csv`

### 5. Emissions Calculator (`src/emissions_calculator.py`)
- Calculates avoided CO2 emissions vs fossil fuel baseline
- **Emission factors**: Coal: 820, Gas: 490, Nuclear: 12 gCO2/kWh
- Computes annual and cumulative emissions 2025-2050
- **Outputs**: `avoided_emissions.csv`, `global_avoided_emissions.csv`

---

## Running the Pipeline

### Command-Line Interface

```bash
# Run all pipeline steps
python pipeline.py --all

# Run specific steps
python pipeline.py --steps ingestion features projection scenario emissions

# Clean previous outputs and rerun
python pipeline.py --all --clean

# Skip already completed steps
python pipeline.py --all --skip-if-exists
```

### Pipeline Steps

| Step | Description | Output File |
|------|-------------|-------------|
| `ingestion` | Load and clean raw data | `nuclear_tracker_cleaned.csv` |
| `features` | Calculate capacity factors | `nuclear_features.csv` |
| `projection` | Generate 2050 projections | `nuclear_projections_2050.csv` |
| `scenario` | Compare against IEA NZE | `scenario_comparison.csv` |
| `emissions` | Calculate avoided emissions | `avoided_emissions.csv` |

### Logs

All pipeline runs are logged to `logs/pipeline_YYYYMMDD_HHMMSS.log`

---

## Statistical Analysis

The project includes a comprehensive statistical analysis notebook (`notebooks/02_statistical_analysis.ipynb`) covering:

1. **Descriptive Statistics**: Distribution analysis, normality tests (Shapiro-Wilk, K-S)
2. **Hypothesis Testing**: t-tests, ANOVA, Chi-square for group comparisons
3. **Correlation Analysis**: Pearson correlation matrix, partial correlations
4. **Time Series Analysis**: Trend analysis, autocorrelation (ACF/PACF)
5. **Regression Modeling**: Multiple linear regression with diagnostics
6. **Gap Analysis Statistics**: Confidence intervals for IEA scenario gaps
7. **Emissions Uncertainty**: Propagating uncertainty through calculations
8. **PCA**: Dimensionality reduction for regional pattern analysis

### Running the Notebook

```bash
# Option 1: Jupyter Notebook
jupyter notebook notebooks/02_statistical_analysis.ipynb

# Option 2: Jupyter Lab
jupyter lab notebooks/02_statistical_analysis.ipynb

# Option 3: VS Code
# Open the .ipynb file directly in VS Code
```

---

## Interactive Dashboard

The Streamlit dashboard (`app.py`) provides 7 interactive analysis pages:

### Pages

1. ** Global Overview**: High-level metrics, plant status distribution, global timeline
2. ** Regional Analysis**: Regional projections, capacity distribution, generation share
3. ** Projections 2025-2050**: Multiple visualization types (line, area, bar, heatmap)
4. ** IEA Scenario Comparison**: Gap analysis, regional alignment status
5. ** Avoided Emissions**: Annual and cumulative CO2 reductions by region
6. ** Pipeline Analysis**: Operating vs pipeline contribution breakdown
7. ** Statistical Analysis**: Capacity distributions, correlations, summary stats

### Features

- **Interactive Filters**: Year range, regions, scenarios
- **Multiple Visualizations**: 21 charts with consistent formatting
- **Responsive Design**: Works on desktop, tablet, mobile
- **Export Capabilities**: Download charts as PNG
- **Hover Tooltips**: Detailed values on mouse hover

### Running the Dashboard

```bash
# Default (auto-opens browser)
streamlit run app.py

# Specific port
streamlit run app.py --server.port 8080

# Headless mode (no browser)
streamlit run app.py --server.headless true
```

---

## Data Sources

- **Global Nuclear Power Tracker** (September 2025): Primary dataset containing worldwide nuclear plant data
  - 1,400+ plants across all statuses
  - Data fields: Capacity, Country, Region, Status, Start Year, Technology

- **IEA Net Zero by 2050 Scenario** (2021): Reference targets for nuclear generation
  - Regional nuclear generation targets 2025-2050
  - Annex A data from IEA NZE 2021 report

---

## Methodology

### 1. Data Cleaning
- Process nuclear tracker Excel data
- Standardize units (MW → GW, GWh → TWh)
- Handle missing values and outliers
- Normalize region/country names

### 2. Feature Engineering
- Calculate capacity factors based on operational status
- Compute annual generation estimates
- Aggregate by region, country, status, subregion

### 3. Projection Modeling
**Deterministic Approach (v1)**:
- Project based on announced start dates
- Fixed capacity factors per status

**Probabilistic Approach (v2)**:
- Monte Carlo simulation with randomized parameters
- Construction delays modeled as normal distributions
- Completion probabilities vary by status and scenario
- Capacity factors sampled from distributions
- Uncertainty bounds (P10-P90) quantified

### 4. Scenario Alignment
- Map nuclear projections to IEA regional classifications
- Calculate year-by-year generation gaps
- Classify regions by alignment status

### 5. Gap Analysis
- Identify shortfalls/surpluses vs IEA Net Zero targets
- Calculate absolute gaps (TWh) and percentage gaps
- Regional and global aggregations

### 6. Emissions Calculation
- Use emission factors for coal, gas, nuclear
- Calculate avoided emissions assuming nuclear displaces fossil fuels
- Compute annual and cumulative emissions 2025-2050
- Quantify uncertainty in emissions estimates

---

## Key Results (Expected)

### Projection Findings
- **Global 2050 Generation**: ~5,000-7,000 TWh (Base scenario)
- **IEA Gap**: Significant shortfall (~500-1,500 TWh) in most scenarios
- **Regional Leaders**: Asia, North America dominate capacity additions
- **Pipeline Risk**: 30-60% of 2050 generation depends on unbuilt plants

### Statistical Insights
- **Capacity Distribution**: Right-skewed, most plants 600-1200 MW
- **Operating vs Pipeline**: Pipeline plants trend larger (modern designs) or smaller (SMRs)
- **Time Trend**: Strong positive trend (R² > 0.95), significant growth 2025-2050
- **Correlation**: Capacity-Generation r > 0.99 (expected from formula)

### Emissions Impact
- **Annual Avoided (2050)**: ~2,000-3,000 MtCO2/year
- **Cumulative (2025-2050)**: ~30-50 GtCO2
- **Equivalent**: Removing 400-600 million cars for 25 years

---

## Outputs

### Processed Data Files (`data/processed/`)
- `nuclear_tracker_cleaned.csv`: Cleaned raw data
- `nuclear_features.csv`: Engineered features
- `nuclear_projections_2050.csv`: Year-by-year projections
- `scenario_comparison.csv`: IEA comparison results
- `avoided_emissions.csv`: Emissions calculations
- Plus 12 additional aggregation and analysis files

### Figures (`outputs/figures/`)
Generated from statistical analysis notebook:
- Distribution plots (histograms, Q-Q plots, box plots)
- Correlation heatmaps
- Time series analysis (trend, ACF/PACF)
- Regression diagnostics
- PCA biplots
- Gap analysis charts

### Results (`outputs/results/`)
- `statistical_summary.csv`: Summary statistics table

### Logs (`logs/`)
- Timestamped pipeline execution logs
- Error traces for debugging

---

## Development

### Requirements
- Python 3.11+
- See `requirements.txt` for package dependencies

### Key Dependencies
- **Data**: pandas, numpy, openpyxl
- **Visualization**: matplotlib, seaborn, plotly
- **Stats/ML**: scipy, statsmodels, scikit-learn
- **Dashboard**: streamlit
- **Notebooks**: jupyter, ipykernel
- **Automation**: papermill (for notebook execution)

## References

- **Global Nuclear Power Tracker**: Global Energy Monitor (Sept 2025)
- **IEA Net Zero by 2050**: International Energy Agency (2021)
- **Emission Factors**: IPCC Guidelines for National GHG Inventories (2006)
- **Statistical Methods**:
  - Field, A. (2013). *Discovering Statistics*
  - Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*

---

## License

Academic research project for Climate Data Science course (AIDAMS S5 - IDSI 51003).

---

## Contributors

Research team project for emission projection and transition risk analysis.

---

## Contact

For questions or issues:
1. Check the troubleshooting section above
2. Review pipeline logs in `logs/`
3. Consult inline code documentation
4. Refer to statistical analysis guide in notebook

**Last Updated**: January 2026
