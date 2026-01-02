# Nuclear Energy Generation Projection & Avoided Emissions Analysis

## Project Overview

This project builds an end-to-end modeling pipeline that uses the Global Nuclear Power Tracker to project regional nuclear energy generation until 2050. By comparing the current pipeline (announced, under-construction) against IEA Net Zero scenarios, we identify "Generation Gaps" and quantify "Avoided Emissions" if the nuclear pipeline is successfully deployed.

## Objectives

1. **Data Ingestion**: Load and clean Global Nuclear Power Tracker data
2. **Regional Projection**: Model nuclear energy generation by region until 2050
3. **Scenario Comparison**: Compare projections against IEA Net Zero scenarios
4. **Gap Analysis**: Identify generation gaps between current pipeline and Net Zero requirements
5. **Emission Quantification**: Calculate avoided emissions from nuclear deployment

## Project Structure

```
researchHamada/
├── data/
│   ├── raw/                    # Original data sources
│   ├── processed/              # Cleaned, structured datasets
│   └── external/               # IEA Net Zero scenario data
├── src/
│   ├── data_ingestion.py       # Load & clean nuclear tracker
│   ├── feature_engineering.py  # Calculate capacity factors, regional aggregations
│   ├── projection_model.py     # Project 2050 generation
│   ├── scenario_comparison.py  # Compare vs IEA scenarios
│   └── emissions_calculator.py # Calculate avoided emissions
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_modeling_pipeline.ipynb
│   └── 03_results_visualization.ipynb
├── outputs/
│   ├── figures/               # Visualizations and charts
│   └── results/               # Model outputs and reports
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run exploratory analysis:
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## Data Sources

- **Global Nuclear Power Tracker** (September 2025): Primary dataset containing nuclear plant pipeline data
- **IEA Net Zero Scenarios**: Reference scenarios for comparison (to be sourced)

## Methodology

1. **Data Cleaning**: Process nuclear tracker data, standardize units, handle missing values
2. **Feature Engineering**: Calculate capacity factors, regional aggregations, operational timelines
3. **Projection Modeling**: Forecast generation based on announced/under-construction plants
4. **Scenario Alignment**: Map projections to IEA regional classifications
5. **Gap Analysis**: Calculate differences between projections and Net Zero requirements
6. **Emissions Calculation**: Quantify avoided CO2 emissions vs fossil fuel baseline

## Deliverables

- Reproducible data pipeline (version-controlled code)
- Technical documentation with architecture diagrams
- Visualization of regional generation gaps
- Avoided emissions quantification report
- Presentation materials

## License

Research project for climate analytics.
