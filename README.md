# Cropland expansion and greening in global protected areas
This code reproduces the analysis in the manuscript: Cropland expansion constrains future greening in global protected areas

The folders follow the main steps described in the Methods:

1. protected area filtering (WDPA, dissolve, rasterize)

2. distance predictors

3. LUH2-ESA-CCI harmonization and land demand calibration

4. CMIP6 ensemble processing

5. kNDVI calculation + grid-specific XGBoost model

6. counterfactual attribution

7. trend stats (Sen's slope, Mann-Kendall, GLM, Kruskal-Wallis/Dunn)
   
The analysis uses both Python and R.
Python dependencies are listed in `requirements.txt` and can be installed using: `pip install -r requirements.txt`

The required R packages are listed in `requirements_R.txt` and should be installed in R before running the statistical analyses.

Folders 01–07 contain the code for the main analyses reported in the manuscript. Before running each script, update the input and output paths to match the directory structure. 

The input datasets are not included in this repository because of their large file sizes. Their original sources are listed in the manuscript and Supplementary Information.


Email:qsai@nus.edu.sg 
