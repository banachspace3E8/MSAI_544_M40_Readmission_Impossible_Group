Readmission Impossible
Predicting 30-Day Hospital Readmission Risk in Patients with Diabetes
MSAI-544 — Application of AI in Healthcare | Residency Group Project

Team members: Shishir Khanal, Shiva Sai Reddy Gayam, Ruthvik Reddy Rakashi, Rexford Essilfie, Rahul Motha

Project overview

This project develops and evaluates machine-learning approaches for identifying patients with diabetes who may be readmitted to the hospital within 30 days of discharge. The work covers data quality, exploratory analysis, feature engineering, predictive modeling, explainability, fairness, and a proposed clinical deployment workflow.

The goal is to help a hospital prioritize post-discharge follow-up when its transitional-care team can contact approximately 15% of discharged patients. These are retrospective research results, not a validated tool for clinical use.
Dataset
Source: UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999–2008 (dataset 296; Strack et al., 2014).
Item	Description
Raw data	diabetic_data.csv — 101,766 inpatient encounters, 71,518 unique patients, 50 original columns
Code descriptions	ids_mapping.csv — admission type, discharge disposition, and admission source mappings
Target	readmit_30 = 1 if readmitted is <30; otherwise 0
Cohort definition	First encounter per patient, then exclude death, hospice, and unmapped/invalid discharge dispositions as implemented in the repository
Final cohort in the supplied results	69,195 records; 9.002% 30-day readmission
Model features	65 numeric and encoded predictors


The code treats ? as missing and represents absent HbA1c/glucose testing as a not-tested category. It groups diagnosis and administrative codes rather than feeding high-cardinality raw codes directly into the models. Patient identifiers and the outcome label are excluded from model predictors.
Technologies
- Language: Python
- Data processing: pandas, NumPy
- Machine learning: scikit-learn
- Interactive visualization: Plotly
- Execution: Jupyter Notebook in VS Code / GitHub Codespaces, or Python script
- Environment: requirements.txt contains the environment's pinned packages
Setup and execution
Run commands from the repository root, the folder containing diabetic_data.csv, ids_mapping.csv, and group_python_starter.py.
Option A — GitHub Codespaces / Jupyter Notebook (recommended)
1. Open this GitHub repository in a Codespace.
2. Open group_python_starter.ipynb in VS Code.
3. Select an installed Python kernel. If prompted, install the Jupyter/Python extensions or ipykernel.
4. Install dependencies in the Codespaces terminal:
   python -m pip install -r requirements.txt
5. Use Restart Kernel and Run All, or execute cells in order from setup and data loading through Analyses 1–10. The notebooks depend on variables created by earlier cells; running Analysis 3 or Analysis 6 on its own will not work in a fresh kernel.
6. Review the displayed charts and the generated .html, .csv, and model_card.md files.
If the pinned requirements fail on your operating system, install the core dependencies with:
python -m pip install pandas numpy plotly scikit-learn ipykernel nbformat
Option B — Python script
python group_python_starter.py
Remote-terminal note: The script contains Plotly .show() calls. In a headless Codespace terminal, these can wait for a browser connection and appear to hang. Prefer running the notebook. If running the script in a remote terminal, comment out Plotly .show() calls; the corresponding write_html() statements save the interactive reports separately.

The supplied group_r_starter.R is an R starter/reference file; the completed analysis in this repository is implemented in Python.
Analyses and visualizations
Analysis	Purpose	Interactive HTML output
1. Data quality & cohort	Repeated patients, exclusions, missing values, cohort funnel	analysis1_cohort_funnel.html
2. Demographics	Readmission by age, race, and gender; additional demographic view	analysis2_demographics.html, analysis2_demographic_heatmaps.html
3. Utilization & clinical factors	Prior visits, length of stay, HbA1c results	analysis3_utilization.html
4. Diagnoses & medications	Grouped ICD-9 diagnoses, insulin, medication changes and counts	analysis4_diagnoses_medications.html
5. Feature engineering	Numeric/binary features, grouped/one-hot categoricals, correlations	analysis5_feature_correlations.html
6. Baseline model	Class-weighted logistic regression; ROC and precision-recall curves	analysis6_baseline_model.html
7. Advanced models	Random forest and histogram gradient boosting compared with baseline	analysis7_model_comparison.html
8. Explainability	Feature importance and partial dependence	analysis8_explainability.html
9. Fairness audit	Performance and flag rates across race, gender, and age	analysis9_fairness_audit.html
10. Deployment analysis	Outreach capacity, cumulative gains, risk tiers	analysis10_deployment.html


Open the generated HTML files in a web browser to explore the interactive charts. They are report artifacts; they are not input data files.
Repository files and outputs
File	Use
group_python_starter.ipynb	Main interactive Python notebook; analyses in order
group_python_starter.py	Python script version of the analysis
diabetic_data.csv, ids_mapping.csv	Original dataset and mapping inputs
cohort_definition.csv	Cohort funnel step counts; not the patient-level cleaned data
cohort_features.csv	Encoded features, target, and demographic columns for downstream analyses
feature_dictionary.csv	Names, types, and descriptions of engineered model features
analysis6_X_test.csv	Held-out test predictors with retained row indices
analysis6_y_test.csv	Held-out test outcomes with retained row indices
analysis6_g_test.csv	Race, gender, and age for the same test rows
model_results.csv	Cross-validation and held-out model comparison metrics
analysis7_p_best.csv	Final-model test predicted probabilities
feature_importance.csv	Explainability feature-importance estimates
fairness_audit.csv	Group-level fairness metrics at the selected threshold
capacity_analysis.csv	Outreach capacity, capture, precision, and workload calculations
risk_tiers.csv	Low/medium/high risk tier summaries
model_card.md	Model purpose, evaluation, subgroup findings, intended use and limitations
project_summary.csv	Consolidated headline project metrics
GROUP_PROJECT_Instructions.pdf / .docx	Assignment specifications
Data_Dictionary_Diabetes_130_Hospitals.pdf / .docx	Data descriptions and known limitations
requirements.txt	Package versions for the supplied environment


Model handoff: Analyses 7–9 should use the same stratified train/test split and preserve test row alignment. The shared split is 80% training / 20% test, random_state=42. analysis6_X_test.csv, analysis6_y_test.csv, analysis6_g_test.csv, and analysis7_p_best.csv must refer to the same held-out rows before model predictions and fairness statistics are compared.
Results from the included outputs
The following numbers are read from the repository's included output CSV files and model card; rerunning the analysis after changes may produce different results.
Model	Test ROC AUC	Test PR AUC	Test Brier score
Logistic regression (balanced)	0.644	0.158	0.2316
Random forest	0.651	0.162	0.1729
Histogram gradient boosting (selected)	0.648	0.162	0.0797


The final model was selected for its calibration and runtime trade-offs, despite random forest's slightly higher test ROC AUC. At 15% outreach capacity, the included capacity analysis reports:
- 2,076 test patients flagged out of 13,839.
- 371 of 1,246 observed readmissions identified (29.8% sensitivity/capture).
- 17.9% positive predictive value (PPV), approximately 5.6 flags per readmission identified.
These are retrospective prioritization metrics, not the number of readmissions prevented. Intervention effectiveness needs prospective evaluation.
Reproducibility and quality checks
- Keep all required inputs in the repository root when running the supplied script/notebook.
- Execute all notebook cells sequentially after restarting the kernel; downstream variables depend on earlier analyses.
- Verify cohort counts against cohort_definition.csv (101,766 raw → 71,518 first encounters → 69,195 final rows in this snapshot).
- Verify that cohort_features.csv and the test-set CSVs use the same feature definitions and row indices before running later analyses.
- Check that all .html files open and that exported CSVs agree with the figures, slides and report.
- Before final submission: review project_summary.csv. In the supplied ZIP, several fields are still blank even though detailed results exist in model_results.csv, capacity_analysis.csv, and model_card.md; regenerate the summary if appropriate.
Limitations and responsible use
The data were collected during 1999–2008 and may not represent modern hospital operations or diabetes treatment. Some social, behavioral, and post-discharge risk factors are unavailable. The target includes any recorded readmission within 30 days and does not reliably distinguish preventable or unplanned readmissions. Subgroup results show meaningful disparities, particularly by age, and require fairness review and local validation before any operational use.

The model is intended to support discussions about a controlled clinical pilot, not to deny services or automate treatment decisions. See model_card.md for details on proposed care-team workflow, subgroup auditing, monitoring, oversight, and privacy/regulatory considerations.
Sources
1. Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999–2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
2. Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. BioMed Research International, 2014, 781670. https://doi.org/10.1155/2014/781670
Academic submission
This repository contains the code and analytical outputs. The assignment separately requires an interactive dashboard, an executive slide deck (10–15 slides), and a technical report (4–6 pages plus model-card and AI-tool-use disclosure appendices). Include the required AI-use disclosure and evidence in the report as directed by the course instructions. The README does not replace those deliverables.

License: See the repository's LICENSE file for software licensing terms. The UCI dataset has its own attribution requirements.