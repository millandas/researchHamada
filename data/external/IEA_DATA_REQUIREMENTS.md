# IEA Net Zero Scenario Data Requirements

## Overview

To compare your nuclear pipeline projections against global climate targets, you'll need data from the **IEA Net Zero Emissions by 2050 Scenario (NZE)**. This document outlines the specific data requirements and sources.

---

## 1. IEA Scenario Background

### What is the Net Zero Scenario?

The **IEA Net Zero Emissions by 2050 Scenario** provides a pathway for the global energy sector to achieve net-zero CO2 emissions by 2050, limiting global warming to 1.5°C.

### Key Publications to Source:

1. **World Energy Outlook (WEO)** - Annual flagship report
   - Latest: World Energy Outlook 2024/2025
   - Contains nuclear generation projections by region to 2050

2. **Net Zero Roadmap** - Detailed pathway analysis
   - Updates to the Net Zero by 2050 pathway

3. **Energy Technology Perspectives (ETP)** - Technology-specific analysis

---

## 2. Required Data Fields

### A. Nuclear Generation Capacity Projections

You need **annual or milestone year projections** for nuclear capacity (GW) or generation (TWh):

| Data Field | Description | Unit | Timeline |
|------------|-------------|------|----------|
| Nuclear Capacity | Installed nuclear capacity | GW | 2025, 2030, 2035, 2040, 2050 |
| Nuclear Generation | Annual electricity generation | TWh/year | 2025, 2030, 2035, 2040, 2050 |
| Regional Breakdown | By IEA region | GW or TWh | Same years |

### B. Regional Classifications

IEA typically uses these regional groupings (align your nuclear tracker data accordingly):

**Major Regions:**
- **Advanced Economies**: US, EU, Japan, South Korea, Canada, Australia, etc.
- **Emerging & Developing Economies**: China, India, Southeast Asia, Latin America, Africa, Middle East

**Sub-regions** (if available):
- North America (US, Canada)
- Europe (EU27, UK, others)
- Asia Pacific (China, India, Japan, South Korea, ASEAN)
- Middle East
- Africa
- Latin America
- Eurasia (Russia, Central Asia)

### C. Additional Context Data (Optional but Recommended)

| Data Type | Use Case |
|-----------|----------|
| Total electricity generation | Calculate nuclear share |
| Fossil fuel generation baseline | Calculate avoided emissions |
| CO2 emission factors | Emission calculations |
| Capacity factors assumed | Reconcile capacity ↔ generation |

---

## 3. Data Sources & Access

### Option 1: IEA Official Data Products (RECOMMENDED)

**IEA Data & Statistics Platform**
- URL: https://www.iea.org/data-and-statistics
- Coverage: Comprehensive scenario data
- Access: Some data is free; detailed datasets require subscription
- Format: Excel, CSV downloads

**Specific Datasets to Look For:**
1. **World Energy Model (WEM) outputs** - Scenario projections
2. **Extended Energy Balances** - Historical + projection data
3. **Net Zero Scenario databases**

### Option 2: Published Reports (Free Access)

**IEA World Energy Outlook Reports**
- URL: https://www.iea.org/reports/world-energy-outlook
- Look for: Executive Summary, Nuclear chapter, Scenario appendices
- Data availability: Summary tables and charts (may need data extraction)

**IEA Net Zero by 2050 Roadmap**
- URL: https://www.iea.org/reports/net-zero-by-2050
- Contains: Milestone data for 2030, 2050

**Data Extraction Approach:**
- Download PDF reports
- Extract tables from Annexes/Appendices
- Manually transcribe scenario data for key years

### Option 3: Academic & Third-Party Sources

If direct IEA data is inaccessible:

1. **IIASA Database** (Integrated Assessment Model database)
   - URL: https://data.ece.iiasa.ac.at/
   - Contains: Multiple climate scenarios including IEA-aligned pathways

2. **IPCC AR6 Scenario Database**
   - URL: https://data.ece.iiasa.ac.at/ar6/
   - Contains: 1.5°C pathways similar to IEA NZE

3. **Energy research papers** citing IEA scenarios
   - Search Google Scholar for: "IEA Net Zero nuclear capacity 2050"

### Option 4: Proxy Data (If Official Data Unavailable)

Use alternative Net Zero scenarios as benchmarks:
- **IRENA 1.5°C Scenario** (REmap case)
- **Bloomberg NEF Net Zero Scenario**
- **Shell Sky Scenario**
- Academic IAM scenarios (IMAGE, MESSAGE, REMIND)

---

## 4. Data Structure Template

Create a CSV file in `data/external/` with this structure:

### File: `iea_nze_nuclear_projections.csv`

```csv
region,year,nuclear_capacity_gw,nuclear_generation_twh,total_generation_twh,source,notes
Advanced Economies,2025,XXX,XXX,XXX,IEA WEO 2024,Baseline
Advanced Economies,2030,XXX,XXX,XXX,IEA WEO 2024,NZE Scenario
Advanced Economies,2040,XXX,XXX,XXX,IEA WEO 2024,NZE Scenario
Advanced Economies,2050,XXX,XXX,XXX,IEA WEO 2024,NZE Scenario
Emerging Economies,2025,XXX,XXX,XXX,IEA WEO 2024,Baseline
...
China,2030,XXX,XXX,XXX,IEA WEO 2024,Country-specific
United States,2030,XXX,XXX,XXX,IEA WEO 2024,Country-specific
...
```

**Fields:**
- `region`: Geographic region (align with IEA classifications)
- `year`: Projection year (2025, 2030, 2035, 2040, 2050)
- `nuclear_capacity_gw`: Nuclear capacity in GW
- `nuclear_generation_twh`: Annual generation in TWh
- `total_generation_twh`: Total electricity generation for context
- `source`: Citation (e.g., "IEA WEO 2024 NZE Scenario")
- `notes`: Any assumptions or caveats

---

## 5. Data Acquisition Workflow

### Step 1: Access IEA Reports
1. Visit IEA website
2. Download latest World Energy Outlook (WEO 2024 or 2025)
3. Look for Net Zero Scenario chapter and Annexes

### Step 2: Extract Nuclear Data
1. Find tables showing nuclear capacity/generation by region
2. Extract data for milestone years: 2025, 2030, 2040, 2050
3. Note the regional definitions used

### Step 3: Structure Data
1. Create CSV file following the template above
2. Map IEA regions to match your nuclear tracker regional classification
3. Document assumptions and data sources

### Step 4: Validate Data
- Check if capacity and generation are consistent (using capacity factors)
- Typical nuclear capacity factor: 80-90%
- Formula: `Generation (TWh) = Capacity (GW) × 8760 hours × Capacity Factor / 1000`

### Step 5: Document
Create a data dictionary explaining:
- Source citations
- Regional mappings
- Assumptions
- Data quality notes

---

## 6. Region Mapping Strategy

You'll need to map your **Global Nuclear Power Tracker** country-level data to **IEA regions**.

### Recommended Approach:

1. **Create a mapping file**: `data/external/country_to_iea_region_mapping.csv`

```csv
country,iea_region,iea_subregion
United States,Advanced Economies,North America
Canada,Advanced Economies,North America
China,Emerging Economies,China
India,Emerging Economies,India
France,Advanced Economies,Europe
Germany,Advanced Economies,Europe
Japan,Advanced Economies,Asia Pacific
South Korea,Advanced Economies,Asia Pacific
...
```

2. **Use this mapping** to aggregate your nuclear tracker projects to IEA regions

3. **Enable comparison** between:
   - Your projection (bottom-up from nuclear tracker)
   - IEA NZE target (top-down scenario)

---

## 7. Key Metrics for Comparison

Once you have both datasets (nuclear tracker + IEA scenarios), calculate:

### Generation Gap
```
Gap = IEA NZE Target (TWh) - Your Projection (TWh)
```

### Capacity Gap
```
Gap = IEA NZE Capacity (GW) - Your Pipeline Capacity (GW)
```

### Avoided Emissions
```
Avoided CO2 = Generation Gap × Emission Factor (tCO2/MWh)
```

Where emission factor = displaced fossil generation (typically coal/gas)
- Coal: ~0.9-1.0 tCO2/MWh
- Natural Gas: ~0.4-0.5 tCO2/MWh
- Mixed baseline: ~0.5-0.6 tCO2/MWh (use IEA's baseline emission intensity)

---

## 8. Fallback Strategy

If you cannot access official IEA data:

### Option A: Use Published Summary Data
- Extract key numbers from free IEA report summaries
- Use for major regions only (Advanced vs Emerging economies)

### Option B: Use Academic Proxies
- Search for papers that cite IEA NZE nuclear projections
- Use IPCC 1.5°C scenario database as benchmark

### Option C: Create Simplified Benchmark
Use global-level targets:
- IEA NZE estimates **~850-900 GW** global nuclear capacity by 2050
- Current global capacity: ~370 GW (2023)
- Derive regional shares based on current distribution and announced plans

---

## 9. Expected Data Output

After data acquisition, you should have:

1. **`iea_nze_nuclear_projections.csv`** - IEA scenario data
2. **`country_to_iea_region_mapping.csv`** - Region mapping
3. **`emission_factors.csv`** - CO2 emission factors for avoided emissions calculation
4. **`DATA_SOURCES.md`** - Full documentation of sources and methodology

---

## 10. Next Steps After Data Acquisition

1. Load IEA scenario data in your pipeline
2. Map nuclear tracker data to IEA regions
3. Project nuclear tracker capacity/generation to 2050
4. Compare projections vs IEA NZE targets
5. Visualize gaps by region
6. Calculate avoided emissions if gaps are closed

---

## Resources & References

### IEA Official
- Main site: https://www.iea.org
- Data portal: https://www.iea.org/data-and-statistics
- WEO reports: https://www.iea.org/reports/world-energy-outlook
- Net Zero roadmap: https://www.iea.org/reports/net-zero-by-2050

### Alternative Scenario Sources
- IRENA: https://www.irena.org/Energy-Transition/Outlook
- IPCC Database: https://data.ece.iiasa.ac.at/ar6/
- IIASA: https://data.ece.iiasa.ac.at/

### Academic Resources
- Google Scholar: "IEA Net Zero nuclear"
- Research papers on 1.5°C pathways and nuclear role

---

## Contact & Support

If you need help accessing IEA data:
- Check if your institution has IEA data subscription
- Request specific data tables from IEA directly
- Use library resources for report access
- Collaborate with researchers who have data access

---

**Document Version:** 1.0
**Last Updated:** 2026-01-02
**Author:** Research Project Team
