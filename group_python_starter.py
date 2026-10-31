"""
GROUP PROJECT: Hospital Readmission Risk - An End-to-End Clinical AI Solution
Group Project - Starter Code (Python)
Saturday: 10 hours work time

MSAI-544 Application of AI in Healthcare - Residency Group Project

TEAM INFORMATION:
Team Name: _________Readmission Impossible______________
Members:   ________Shishir Khanal, Shiva Sai Reddy Gayam, Ruthvik Reddy Rakashi, Rexford Essilfie, Rahul Motha_______________

DELIVERABLES CHECKLIST:
[ ] Interactive dashboard (HTML / notebook)
[ ] Slide deck (10-15 slides, .pptx)
[ ] Technical report (4-6 pages) with model card
[ ] Clean code with README
[ ] All 10 analyses complete

AI TOOL POLICY: AI tools may be used for debugging, ideating and report/slide clean-up only,
never to write code. Disclose all use (with screenshots) in the technical report.
"""

# ==============================================================================
# SETUP & IMPORTS
# ==============================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             brier_score_loss, confusion_matrix, recall_score, precision_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance, partial_dependence
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
# Professional color palette: use consistently across every chart
BLUE, RED, GREEN, GRAY, PURPLE = '#3498db', '#e74c3c', '#2ecc71', '#95a5a6', '#9b59b6'

TEAM_NAME = "Readmission Impossible"

print("=" * 70)
print(" " * 12 + "HOSPITAL READMISSION RISK: END-TO-END CLINICAL AI")
print("=" * 70)
print(f"Team: {TEAM_NAME}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==============================================================================
# DATA LOADING
# ==============================================================================

print("\nLOADING DATA...")

# NOTE: both files are provided in the project folder. Update paths if needed.
# '?' is the missing-value marker in diabetic_data.csv
df = pd.read_csv("diabetic_data.csv", na_values=['?', 'None'], low_memory=False)  # 'None' = test not performed
ids = pd.read_csv("ids_mapping.csv", keep_default_na=False)   # admission/discharge/source code descriptions

print(f"Encounters: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Unique patients: {df['patient_nbr'].nunique():,}")
print("\nColumn names:")
print(df.columns.tolist())

# ==============================================================================
# DATA EXPLORATION
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 25 + "DATA EXPLORATION")
print("=" * 70)

print("\nReadmitted (raw, three classes):")
print(df['readmitted'].value_counts())

print("\nMissing fraction (columns with any missing):")
miss = df.isnull().mean().sort_values(ascending=False)
print(miss[miss > 0].round(3))

print("\nEncounters per patient:")
print(df['patient_nbr'].value_counts().describe().round(2))

print("\nNumeric summary:")
print(df[['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
          'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']].describe().round(2).T)

print("\nID code descriptions (first rows):")
print(ids.head(10).to_string(index=False))

# ==============================================================================
# ANALYSIS 1: DATA QUALITY & COHORT DEFINITION
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 1: DATA QUALITY & COHORT DEFINITION")
print("=" * 70)

n0 = len(df)

# 1a. Drop unusable columns: weight is 97% missing; examide and citoglipton are never prescribed
df = df.drop(columns=['weight', 'examide', 'citoglipton'])
print("Dropped weight, examide, citoglipton")

# 1b. TODO: Handle repeated patients
# Many patients have several encounters. Rows from the same patient are NOT independent:
# if one encounter lands in the training set and another in the test set, the model has
# effectively seen the test patient. Decision options:
#   Option A: keep only the FIRST encounter per patient (Strack et al., 2014)
#   Option B: keep all encounters but split train/test BY PATIENT (GroupShuffleSplit)
#   Option C: keep all encounters and ignore the issue (NOT acceptable; explain why in report)
# Hint for Option A: df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')

# YOUR DECISION HERE:



n1 = len(df)

# 1c. TODO: Exclude encounters where readmission is impossible or undefined
# Look up discharge_disposition_id descriptions in `ids`. Patients who died or went to
# hospice cannot be readmitted. Decide what else, if anything, to exclude and justify it.
disch = ids[ids['id_type'] == 'discharge_disposition_id'].set_index('id')['description']
print("\nDischarge dispositions:\n", disch.to_string())

# YOUR DECISION HERE:



n2 = len(df)

# 1d. TODO: Handle remaining missing values (race, payer_code, medical_specialty, diag_1/2/3,
#     A1Cresult, max_glu_serum). Missing is often informative in clinical data ("not tested").
#     Decide: drop rows, fill with "Unknown"/"Missing", or treat as its own category.

# YOUR DECISION HERE:



n3 = len(df)

# 1e. Define the target: readmitted within 30 days
df['readmit_30'] = (df['readmitted'] == '<30').astype(int)
print(f"\nCohort: {n3:,} rows; 30-day readmission rate = {df['readmit_30'].mean():.3%}")

# TODO: Build a cohort table (step, rows remaining) and save it as 'cohort_definition.csv'.
#       Plot it as a funnel chart (go.Funnel) and save as 'analysis1_cohort_funnel.html'.
# Expected (Option A with expired/hospice excluded): roughly 101,766 -> 71,500 -> 70,000 rows, ~9% positive



print("Analysis 1 complete")

# ==============================================================================
# ANALYSIS 2: DEMOGRAPHICS & READMISSION RATES
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 2: DEMOGRAPHICS & READMISSION RATES")
print("=" * 70)

df['race'] = df['race'].fillna('Unknown')


def rate_table(col):
    """30-day readmission rate and count by a grouping column."""
    t = df.groupby(col)['readmit_30'].agg(['mean', 'count']).rename(columns={'mean': 'rate', 'count': 'n'})
    return t.sort_index()


age_t, race_t, gender_t = rate_table('age'), rate_table('race'), rate_table('gender')
print("\nBy age:\n", age_t.round(3))
print("\nBy race:\n", race_t.round(3))
print("\nBy gender:\n", gender_t.round(3))

fig2 = make_subplots(rows=1, cols=3, subplot_titles=('By Age Group', 'By Race', 'By Gender'))
for i, (t, color) in enumerate([(age_t, BLUE), (race_t, PURPLE), (gender_t, GREEN)], start=1):
    fig2.add_trace(go.Bar(x=t.index.astype(str), y=t['rate'], text=t['n'], textposition='outside',
                          marker_color=color, showlegend=False), row=1, col=i)
    fig2.add_hline(y=df['readmit_30'].mean(), line_dash='dash', line_color=GRAY, row=1, col=i)
fig2.update_yaxes(title_text='30-day readmission rate', tickformat='.0%', row=1, col=1)
fig2.update_layout(height=500,
                   title_text='Analysis 2: 30-Day Readmission Rate by Demographics (labels = n; dashed = overall)')
fig2.show()
fig2.write_html('analysis2_demographics.html')

# TODO: Add at least one more demographic view (e.g., age x gender heatmap, or payer type)



print("Analysis 2 complete")

# ==============================================================================
# ANALYSIS 3: UTILIZATION & CLINICAL FACTORS
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 3: UTILIZATION & CLINICAL FACTORS")
print("=" * 70)

util_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
             'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
print("\nMean utilization by outcome:")
print(df.groupby('readmit_30')[util_cols].mean().round(2).T)

# Readmission rate by number of inpatient visits in the prior year (capped at 4+)
inp = df.copy()
inp['prior_inpatient'] = inp['number_inpatient'].clip(upper=4).astype(str).replace('4', '4+')
inp_t = inp.groupby('prior_inpatient')['readmit_30'].agg(['mean', 'count'])
print("\nBy prior inpatient visits:\n", inp_t.round(3))

# TODO: Build a 1x3 figure:
#   Panel 1: bar of readmission rate by prior inpatient visits (inp_t, provided above)
#   Panel 2: line of readmission rate by length of stay (time_in_hospital, 1-14 days)
#   Panel 3: bar of readmission rate by HbA1c result (treat missing as "Not tested")
# Save as 'analysis3_utilization.html'
# Discussion point for the report: does testing HbA1c relate to readmission? (Strack et al. said yes)



print("Analysis 3 complete")

# ==============================================================================
# ANALYSIS 4: DIAGNOSES & MEDICATIONS
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 4: DIAGNOSES & MEDICATIONS")
print("=" * 70)


def icd9_group(code):
    """Group ICD-9 diagnosis codes into clinical categories following Strack et al. (2014)."""
    if pd.isna(code):
        return 'Missing'
    s = str(code)
    if s.startswith(('V', 'E')):
        return 'Other'
    try:
        c = float(s)
    except ValueError:
        return 'Other'
    if 390 <= c <= 459 or c == 785: return 'Circulatory'
    if 460 <= c <= 519 or c == 786: return 'Respiratory'
    if 520 <= c <= 579 or c == 787: return 'Digestive'
    if int(c) == 250: return 'Diabetes'
    if 800 <= c <= 999: return 'Injury'
    if 710 <= c <= 739: return 'Musculoskeletal'
    if 580 <= c <= 629 or c == 788: return 'Genitourinary'
    if 140 <= c <= 239: return 'Neoplasms'
    return 'Other'


df['diag_group'] = df['diag_1'].apply(icd9_group)
print("\nPrimary diagnosis groups:\n", df['diag_group'].value_counts())

MED_COLS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
            'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
# Each medication column is No / Steady / Up / Down
df['n_meds_changed'] = (df[MED_COLS].isin(['Up', 'Down'])).sum(axis=1)
df['n_diabetes_meds'] = (df[MED_COLS] != 'No').sum(axis=1)

# TODO: Build a 1x3 figure:
#   Panel 1: readmission rate by primary diagnosis group (horizontal bars, sorted)
#   Panel 2: readmission rate by insulin order (No / Steady / Up / Down)
#   Panel 3: readmission rate by number of diabetes medications (0, 1, 2, 3+)
# Save as 'analysis4_diagnoses_medications.html'
# Optional: which individual medications are most associated with readmission?



print("Analysis 4 complete")

# ==============================================================================
# ANALYSIS 5: FEATURE ENGINEERING
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 5: FEATURE ENGINEERING")
print("=" * 70)

# Provided helpers: look-ups for the three coded ID columns
adm_type = ids[ids['id_type'] == 'admission_type_id'].set_index('id')['description']
adm_src = ids[ids['id_type'] == 'admission_source_id'].set_index('id')['description']


def group_discharge(i):
    """Collapse 30 discharge codes into 4 clinically meaningful groups."""
    d = disch.get(i, 'Unknown')
    if d.startswith('Discharged to home') and 'health' not in d: return 'Home'
    if 'home health' in d.lower() or 'Home IV' in d: return 'Home with services'
    if any(k in d for k in ['SNF', 'ICF', 'inpatient care', 'hospital']) or 'rehab' in d.lower() or 'nursing' in d.lower():
        return 'Facility transfer'
    return 'Other/Unknown'


# Age band "[70-80)" -> 75 (band midpoint)
feat = pd.DataFrame(index=df.index)
feat['age_num'] = df['age'].str.extract(r'\[(\d+)').astype(int)[0] + 5

# TODO: Build the feature matrix X and target y. Requirements:
#   - Numeric: all util_cols, n_meds_changed, n_diabetes_meds, total prior visits (sum of the three)
#   - Binary flags: A1C tested, A1C high (>7 or >8), glucose tested, insulin ordered, medication
#     changed, any diabetes medication, female
#   - Categorical (one-hot with pd.get_dummies): race, admission type (map codes with adm_type,
#     collapse Not Available/NULL/Not Mapped into Unknown), discharge group (group_discharge),
#     admission source (group into Emergency room / Referral / Transfer / Other), diag_group,
#     payer (top codes + Other + Missing), medical_specialty (top 8 + Other + Missing)
#   - Cast everything to float: X = X.astype(float)
#   - Do NOT include: encounter_id, patient_nbr, readmitted, diag_2/diag_3 raw codes (900+ values)
# Save a feature dictionary (feature, type, description) as 'feature_dictionary.csv'
# Save the modeling table as 'cohort_features.csv' (X + y + race, gender, age for the audit)
# Chart: top 10 positive and top 10 negative correlations with readmit_30 -> 'analysis5_feature_correlations.html'
# Expected: roughly 55-70 features



print("Analysis 5 complete")

# ==============================================================================
# ANALYSIS 6: BASELINE MODEL
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 6: BASELINE MODEL (LOGISTIC REGRESSION)")
print("=" * 70)

# This block assumes you created X (DataFrame) and y (Series) in Analysis 5.
try:
    X, y
except NameError:
    raise SystemExit("\nSTOP: Analysis 6 needs X and y from Analysis 5. Complete Analysis 5, then re-run.")

# Stratified 80/20 split; keep the demographic columns of the test rows for the fairness audit.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
g_test = df.loc[X_test.index, ['race', 'gender', 'age']]
print(f"Train {len(X_train):,} / Test {len(X_test):,}; positive rate {y_train.mean():.3f} / {y_test.mean():.3f}")

# class_weight='balanced' because only ~9% of patients are readmitted within 30 days
lr = Pipeline([('scale', StandardScaler()),
               ('lr', LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced', random_state=RANDOM_STATE))])
lr.fit(X_train, y_train)
p_lr = lr.predict_proba(X_test)[:, 1]


def summarize(name, p, thr=0.5):
    """Standard metrics for an imbalanced clinical classifier."""
    pred = (p >= thr).astype(int)
    return {'Model': name,
            'ROC AUC': round(roc_auc_score(y_test, p), 3),
            'PR AUC': round(average_precision_score(y_test, p), 3),
            'Brier': round(brier_score_loss(y_test, p), 4),
            f'Sensitivity@{thr}': round(recall_score(y_test, pred), 3),
            f'Precision@{thr}': round(precision_score(y_test, pred, zero_division=0), 3)}


res_lr = summarize('Logistic Regression (balanced)', p_lr)
print(res_lr)
# Expected: ROC AUC about 0.62-0.66. Readmission is a HARD prediction problem; published
# models on this dataset land in the same range. Do not "fix" this by leaking information.

fpr, tpr, _ = roc_curve(y_test, p_lr)
prec, rec, _ = precision_recall_curve(y_test, p_lr)
fig6 = make_subplots(rows=1, cols=2, subplot_titles=('ROC Curve', 'Precision-Recall Curve'))
fig6.add_trace(go.Scatter(x=fpr, y=tpr, name=f"LR (AUC {res_lr['ROC AUC']})", line_color=BLUE), row=1, col=1)
fig6.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color=GRAY), showlegend=False), row=1, col=1)
fig6.add_trace(go.Scatter(x=rec, y=prec, name=f"LR (PR AUC {res_lr['PR AUC']})", line_color=RED), row=1, col=2)
fig6.add_hline(y=y_test.mean(), line_dash='dash', line_color=GRAY, row=1, col=2)
fig6.update_layout(height=500, title_text='Analysis 6: Baseline Logistic Regression (dashed = no-skill)')
fig6.show()
fig6.write_html('analysis6_baseline_model.html')

# TODO: Print the 10 largest positive and 10 largest negative coefficients with feature names



print("Analysis 6 complete")

# ==============================================================================
# ANALYSIS 7: ADVANCED MODELS & COMPARISON
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 7: ADVANCED MODELS & COMPARISON")
print("=" * 70)

# TODO: Train at least TWO additional models and compare all of them:
#   RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight='balanced_subsample', n_jobs=-1)
#   HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)
#   (optional) tune one hyperparameter with a small grid; document what you tried
# Requirements:
#   - 5-fold stratified CV ROC AUC on the TRAINING set for each model (StratifiedKFold, shuffle=True)
#   - Test-set ROC AUC, PR AUC, Brier score via summarize()
#   - Table -> 'model_results.csv'
#   - 1x2 figure: ROC curves for all models | calibration curves (sklearn.calibration.calibration_curve)
#     -> 'analysis7_model_comparison.html'
#   - Choose a final model and justify it (discrimination, calibration, interpretability, speed)
#   - Keep its test probabilities in a variable called p_best and the model in best_model



print("Analysis 7 complete")

# ==============================================================================
# ANALYSIS 8: EXPLAINABILITY
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 8: EXPLAINABILITY")
print("=" * 70)

# TODO: Explain the final model
#   - Permutation importance on the TEST set (sklearn.inspection.permutation_importance, scoring='roc_auc',
#     n_repeats=5). Table -> 'feature_importance.csv'
#   - Partial dependence for at least one key feature (e.g., number_inpatient) with partial_dependence()
#   - 1x2 figure: top 15 features with error bars | partial dependence curve -> 'analysis8_explainability.html'
#   - Compare with the logistic regression coefficients from Analysis 6: do the models agree?
#   - Clinical sense check: is each top driver plausible, or is it a proxy for something else
#     (e.g., discharge to a facility = sicker patient)?
# Optional: SHAP values (pip install shap) for individual-patient explanations



print("Analysis 8 complete")

# ==============================================================================
# ANALYSIS 9: FAIRNESS & BIAS AUDIT
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 9: FAIRNESS & BIAS AUDIT")
print("=" * 70)

# TODO: Audit the final model across race, gender and age groups (use g_test)
#   - Pick an operating point: e.g., flag the top 15% highest-risk patients
#     thr = np.quantile(p_best, 0.85)
#   - For each group with >= 100 test patients compute: n, prevalence, ROC AUC, flag rate,
#     sensitivity, false positive rate, PPV. Table -> 'fairness_audit.csv'
#   - 1x3 figure: AUC by group | sensitivity by group | flag rate vs prevalence scatter
#     -> 'analysis9_fairness_audit.html'
#   - Discuss: which groups are under-flagged relative to their risk? Is that a model problem,
#     a data problem, or both? What would you do about it before deployment?
# Hint: pd.Series(p_best, index=X_test.index) lets you align probabilities with g_test



print("Analysis 9 complete")

# ==============================================================================
# ANALYSIS 10: DEPLOYMENT & CLINICAL INTEGRATION
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 10: DEPLOYMENT & CLINICAL INTEGRATION")
print("=" * 70)

# TODO: Translate the model into an operational plan
#   - Capacity analysis: if the transitional-care team can follow up 5 / 10 / 15 / 20 / 30 / 50% of
#     discharges, how many 30-day readmissions would be captured, what is the PPV, and how many
#     patients must be flagged per readmission captured? Table -> 'capacity_analysis.csv'
#   - Cumulative gains chart (share flagged vs share of readmissions captured) with a random baseline
#   - Risk tiers (Low / Medium / High) with observed readmission rate per tier
#   - Figure -> 'analysis10_deployment.html'
#   - Write a MODEL CARD (model_card.md): intended use, training data, performance, fairness,
#     limitations, monitoring plan, who is accountable. A template is in the instructions.
#   - In the report: where in the workflow does the score appear, who acts on it, what is the
#     intervention, how will you know it is working, and what regulatory/privacy considerations apply



print("Analysis 10 complete")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 25 + "PROJECT SUMMARY")
print("=" * 70)

# TODO: Fill in from your analyses and export 'project_summary.csv'
summary = pd.DataFrame({
    'Metric': ['Cohort size', '30-day readmission rate', 'Features', 'Best model', 'Test ROC AUC',
               'Test PR AUC', 'Readmissions captured at chosen capacity', 'Top driver'],
    'Value': [len(df), f"{df['readmit_30'].mean():.3%}", None, None, None, None, None, None]
})
summary.to_csv('project_summary.csv', index=False)
print(summary.to_string(index=False))

# ==============================================================================
# NEXT STEPS
# ==============================================================================

print("""
REMAINING TASKS:

1. COMPLETE ALL ANALYSES
   [ ] Analyses 1-6 are started: finish the TODOs and refine
   [ ] Complete Analyses 7-10

2. CREATE THE DASHBOARD
   [ ] Decide on format (single HTML with subplots, Jupyter notebook, Dash/Streamlit)
   [ ] Executive summary with key metrics at the top
   [ ] 10-12 interconnected visualizations
   [ ] Clinical insights highlighted

3. CREATE THE SLIDE DECK (10-15 slides, executive-ready, no live presentation)
   [ ] Title slide with team info
   [ ] Clinical problem and why readmission matters (1-2)
   [ ] Data and cohort (1)
   [ ] Methods (1)
   [ ] Key findings (5-7)
   [ ] Fairness and deployment plan (2-3)
   [ ] Limitations and next steps (1)

4. WRITE THE TECHNICAL REPORT (4-6 pages)
   [ ] Executive summary, data and cohort, methods, results, fairness, deployment plan,
       limitations, model card (appendix), AI Tool Use Disclosure

5. FINALIZE CODE
   [ ] Comments throughout; README.md; requirements.txt; test a clean run

6. QUALITY CHECK
   [ ] All files open; properly named; zip ready; team review complete
""")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
