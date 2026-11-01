# %%
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



# %%
# ==============================================================================
# SETUP & IMPORTS
# ==============================================================================

import numpy as np
import pandas as pd
import math
from datetime import date
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

from time import perf_counter
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
# Professional color palette: use consistently across every chart
BLUE, RED, GREEN, GRAY, PURPLE = '#3498db', '#e74c3c', '#2ecc71', '#95a5a6', '#9b59b6'

TEAM_NAME = 'Readmission Impossible'

print("=" * 70)
print(" " * 12 + "HOSPITAL READMISSION RISK: END-TO-END CLINICAL AI")
print("=" * 70)
print(f"Team: {TEAM_NAME}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)



# %%
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



# %%
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



# %%
# ==============================================================================
# ANALYSIS 1: DATA QUALITY & COHORT DEFINITION
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 1: DATA QUALITY & COHORT DEFINITION")
print("=" * 70)

steps = []

n0 = len(df)
steps.append(("initial_dataset", n0))

# 1a. Drop unusable columns: weight is 97% missing; examide and citoglipton are never prescribed
df = df.drop(columns=['weight', 'examide', 'citoglipton'])
print("Dropped weight, examide, citoglipton")

# 1b. Handle repeated patients
# Many patients have several encounters. Rows from the same patient are NOT independent:
# if one encounter lands in the training set and another in the test set, the model has
# effectively seen the test patient. Decision options:
#   Option A: keep only the FIRST encounter per patient (Strack et al., 2014)
#   Option B: keep all encounters but split train/test BY PATIENT (GroupShuffleSplit)
#   Option C: keep all encounters and ignore the issue (NOT acceptable; explain why in report)
# Hint for Option A: df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')

# YOUR DECISION HERE: Option B, since we can have more accurate result by considering more of the picture. Picking only the first encounter may miss later readmission result.
df = df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first')


n1 = len(df)
steps.append(("dropped_duplicates", n1))




# %%
# 1c. Exclude encounters where readmission is impossible or undefined
# Look up discharge_disposition_id descriptions in `ids`. Patients who died or went to
# hospice cannot be readmitted. Decide what else, if anything, to exclude and justify it.
disch = ids[ids['id_type'] == 'discharge_disposition_id'].set_index('id')['description']
# print("\nDischarge dispositions:\n", disch.to_string())

# YOUR DECISION HERE:
# Encounters where readmission is impossible or undefined will be ones where the patient is either expired, or went to hospice 
# 11: Expired,
# 13: Hospice / home,
# 14: Hospice / medical facility,
# 19: Expired at home. Medicaid only, hospice., 
# 20: Expired in a medical facility. Medicaid only, hospice., 
# 21: Expired, place unknown. Medicaid only, hospice.
# 25: Not Mapped, 
# 26: Unknown/Invalid

df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21, 25, 26])]

n2 = len(df)
steps.append(('exclude_impossible_readmissions', n2))



# %%
# 1d. TODO: Handle remaining missing values (race, payer_code, medical_specialty, diag_1/2/3,
#     A1Cresult, max_glu_serum). Missing is often informative in clinical data ("not tested").
#     Decide: drop rows, fill with "Unknown"/"Missing", or treat as its own category.

# YOUR DECISION HERE:
 
# Making missing values into own category
df['A1Cresult'] = df['A1Cresult'].fillna("not tested")
df['max_glu_serum'] = df['max_glu_serum'].fillna("not tested")

df['payer_code'] = df['payer_code'].fillna("Missing")
df['medical_specialty'] = df['medical_specialty'].fillna("Unknown")

df['diag_1'] = df['diag_1'].fillna("Unknown")
df['diag_2'] = df['diag_2'].fillna("Unknown")
df['diag_3'] = df['diag_3'].fillna("Unknown")


n3 = len(df)
steps.append(("missing_values", n3))



# %%
# 1e. Define the target: readmitted within 30 days
df['readmit_30'] = (df['readmitted'] == '<30').astype(int)
print(f"\nCohort: {n3:,} rows; 30-day readmission rate = {df['readmit_30'].mean():.3%}")

# TODO: Build a cohort table (step, rows remaining) and save it as 'cohort_definition.csv'.
#       Plot it as a funnel chart (go.Funnel) and save as 'analysis1_cohort_funnel.html'.
# Expected (Option A with expired/hospice excluded): roughly 101,766 -> 71,500 -> 70,000 rows, ~9% positive

print(steps)


cohort = pd.DataFrame(steps, columns=["step", "rows_remaining"])
cohort.to_csv("cohort_definition.csv", index=False)

fig = go.Figure(go.Funnel(y=cohort["step"], x=cohort["rows_remaining"],
                          textinfo="value+percent initial"))
fig.write_html("analysis1_cohort_funnel.html")


# Plot funnel chart
fig = go.Figure(go.Funnel(
    y=cohort['step'],
    x=cohort['rows_remaining'],
    textinfo="value+percent initial",
    marker={"color": ["deepskyblue", "lightsalmon", "tan"]},
    textposition="inside"
))
fig.update_layout(title_text="Cohort Funnel", title_x=0.5)
fig.write_html("analysis1_cohort_funnel.html")


print("Analysis 1 complete")



# %%
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


# Analysis 2b: 30-day readmission rate by demographic combinations (heatmaps)

df_hm = df[df['gender'].isin(['Male', 'Female'])].copy()

def rate_grid(col, row, sort=False):
    """30-day readmission rate and patient count for each (row, col) combination."""
    rate = df_hm.pivot_table(index=row, columns=col, values='readmit_30', aggfunc='mean')
    pt = df_hm.pivot_table(index=row, columns=col, values='readmit_30',
                           aggfunc='size', fill_value=0)
    if sort:
        order = df_hm.groupby(col)['readmit_30'].mean().sort_values(ascending=True).index
        rate, pt = rate[order], pt[order]
    return rate, pt

age_grid = rate_grid('age', 'gender', sort=False)
payer_grid = rate_grid('payer_code', 'gender', sort=True)

for name, (rate, pt) in [('Age group', age_grid), ('Payer code', payer_grid)]:
    table = rate.T.map('{:.1%}'.format)
    table.index.name = name
    table.columns.name = None
    print(f"\n30-day readmission rate by {name.lower()} and gender:\n")
    print(table.to_string())

fig3 = make_subplots(rows=1, cols=2,
                     subplot_titles=('Age Group x Gender', 'Payer Type x Gender'),
                     shared_yaxes=True,
                     horizontal_spacing=0.04,
                     column_widths=[0.45, 0.55])

for i, (rate, pt) in enumerate([age_grid, payer_grid], start=1):
    pt_labels = pt.map('{:,}'.format).values
    hover = [[f'{x} · {y}<br>Rate: {r:.1%}<br>Patients: {c}' if pd.notna(r) else f'{x} · {y}<br>No patients'
              for x, r, c in zip(rate.columns, rate_row, pt_row)]
             for y, rate_row, pt_row in zip(rate.index, rate.values, pt_labels)]

    fig3.add_trace(go.Heatmap(
        z=rate.values,
        x=rate.columns.astype(str),
        y=rate.index.astype(str),
        text=pt_labels,
        texttemplate='%{z:.1%}<br>%{text} patients',
        hovertext=hover,
        hoverinfo='text',
        coloraxis='coloraxis',
    ), row=1, col=i)

fig3.update_yaxes(title_text='Gender', row=1, col=1)
fig3.update_layout(
    height=450,
    coloraxis=dict(colorscale='Reds',
                   colorbar=dict(title='Rate', tickformat='.0%')),
    title_text='Analysis 2b: 30-Day Readmission Rate by Age Group and Payer Type',
)
fig3.show()
fig3.write_html('analysis2_demographic_heatmaps.html')

print("Analysis 2 complete")



# %%
# ==============================================================================
# ANALYSIS 3: UTILIZATION & CLINICAL FACTORS
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 3: UTILIZATION & CLINICAL FACTORS")
print("=" * 70)


util_cols = [
    'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

print('Mean utilization by outcome:')
print(df.groupby('readmit_30')[util_cols].mean().round(2).T)

prior = df.assign(
    prior_inpatient=df['number_inpatient']
    .clip(upper=4).astype(str).replace({'4': '4+'})
)

prior_rates = (
    prior.groupby('prior_inpatient')['readmit_30']
    .agg(['mean', 'count'])
    .reindex(['0', '1', '2', '3', '4+'])
)

stay_rates = (
    df.groupby('time_in_hospital')['readmit_30']
    .agg(['mean', 'count']).sort_index()
)

a1c_values = (
    df['A1Cresult']
    .fillna('Not tested')
    .replace({'not tested': 'Not tested', 'None': 'Not tested'})
)

a1c_rates = (
    df.assign(a1c_group=a1c_values)
    .groupby('a1c_group')['readmit_30']
    .agg(['mean', 'count'])
    .reindex(['Not tested', 'Norm', '>7', '>8'])
)

print('\nPrior inpatient visits:')
print(prior_rates.round(3))
print('\nHbA1c results:')
print(a1c_rates.round(3))

fig3 = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        'Prior inpatient visits',
        'Length of stay',
        'HbA1c result'
    )
)

fig3.add_trace(
    go.Bar(
        x=prior_rates.index,
        y=prior_rates['mean'],
        text=prior_rates['count'],
        marker_color=BLUE
    ),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(
        x=stay_rates.index,
        y=stay_rates['mean'],
        mode='lines+markers',
        marker_color=PURPLE,
        customdata=stay_rates[['count']],
        hovertemplate='Days: %{x}<br>Rate: %{y:.1%}<br>n: %{customdata[0]}<extra></extra>'
    ),
    row=1, col=2
)

fig3.add_trace(
    go.Bar(
        x=a1c_rates.index,
        y=a1c_rates['mean'],
        text=a1c_rates['count'],
        marker_color=GREEN
    ),
    row=1, col=3
)

for col in (1, 2, 3):
    fig3.update_yaxes(tickformat='.0%', row=1, col=col)

fig3.update_layout(
    title='Analysis 3: Utilization and Clinical Factors',
    height=530,
    showlegend=False
)

fig3.write_html( 'analysis3_utilization.html')

print('Analysis 3 complete')
fig3.show()




# %%
# ==============================================================================
# ANALYSIS 4: DIAGNOSES & MEDICATIONS
# ==============================================================================
print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 4: DIAGNOSES & MEDICATIONS")
print("=" * 70)

def icd9_group(code):
    if pd.isna(code) or str(code) in ('Unknown', 'Missing'):
        return 'Missing'

    value = str(code)

    if value.startswith(('V', 'E')):
        return 'Other'

    try:
        number = float(value)
    except ValueError:
        return 'Other'

    if 390 <= number <= 459 or int(number) == 785:
        return 'Circulatory'
    if 460 <= number <= 519 or int(number) == 786:
        return 'Respiratory'
    if 520 <= number <= 579 or int(number) == 787:
        return 'Digestive'
    if int(number) == 250:
        return 'Diabetes'
    if 800 <= number <= 999:
        return 'Injury'
    if 710 <= number <= 739:
        return 'Musculoskeletal'
    if 580 <= number <= 629 or int(number) == 788:
        return 'Genitourinary'
    if 140 <= number <= 239:
        return 'Neoplasms'
    return 'Other'

df['diag_group'] = df['diag_1'].apply(icd9_group)

MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide',
    'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide',
    'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

df['n_meds_changed'] = (
    df[MED_COLS].isin(['Up', 'Down']).sum(axis=1)
)

df['n_diabetes_meds'] = (
    df[MED_COLS].ne('No').sum(axis=1)
)

diag_rates = (
    df.groupby('diag_group')['readmit_30']
    .agg(['mean', 'count'])
    .sort_values('mean')
)

insulin_rates = (
    df.groupby('insulin')['readmit_30']
    .agg(['mean', 'count'])
    .reindex(['No', 'Steady', 'Up', 'Down'])
)

med_count = (
    df['n_diabetes_meds']
    .clip(upper=3)
    .astype(str)
    .replace({'3': '3+'})
)

med_rates = (
    df.assign(med_group=med_count)
    .groupby('med_group')['readmit_30']
    .agg(['mean', 'count'])
    .reindex(['0', '1', '2', '3+'])
)

change_rates = (
    df.groupby('change')['readmit_30']
    .agg(['mean', 'count'])
)

print('Primary diagnosis groups:')
print(diag_rates.round(3))

print('\nInsulin order:')
print(insulin_rates.round(3))

print('\nMedication changes:')
print(change_rates.round(3))

print('\nNumber of diabetes medications:')
print(med_rates.round(3))

fig4 = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        'Diagnosis category',
        'Insulin order',
        'Number of diabetes medications'
    )
)

fig4.add_trace(
    go.Bar(
        y=diag_rates.index,
        x=diag_rates['mean'],
        orientation='h',
        marker_color=PURPLE,
        customdata=diag_rates[['count']],
        hovertemplate='%{y}<br>Rate: %{x:.1%}<br>n: %{customdata[0]}<extra></extra>'
    ),
    row=1, col=1
)

fig4.add_trace(
    go.Bar(
        x=insulin_rates.index,
        y=insulin_rates['mean'],
        text=insulin_rates['count'],
        marker_color=BLUE
    ),
    row=1, col=2
)

fig4.add_trace(
    go.Bar(
        x=med_rates.index,
        y=med_rates['mean'],
        text=med_rates['count'],
        marker_color=GREEN
    ),
    row=1, col=3
)

fig4.update_xaxes(tickformat='.0%', row=1, col=1)

for col in (2, 3):
    fig4.update_yaxes(tickformat='.0%', row=1, col=col)

fig4.update_layout(
    title='Analysis 4: Diagnoses and Medications',
    height=570,
    showlegend=False,
    margin=dict(l=150)
)

fig4.write_html(
    'analysis4_diagnoses_medications.html'
)

print('Analysis 4 complete')
fig4.show()




# %%
# ==============================================================================
# ANALYSIS 5: FEATURE ENGINEERING
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 5: FEATURE ENGINEERING")
print("=" * 70)

admission_codes = (
    ids.loc[ids['id_type'] == 'admission_type_id']
    .set_index('id')['description']
)

discharge_codes = (
    ids.loc[ids['id_type'] == 'discharge_disposition_id']
    .set_index('id')['description']
)

def discharge_group(code):
    desc = str(discharge_codes.get(code, 'Unknown')).lower()
    if 'home health' in desc or 'home iv' in desc:
        return 'Home with services'
    if desc.startswith('discharged to home'):
        return 'Home'
    if any(word in desc for word in [
        'hospital', 'snf', 'icf', 'rehab',
        'nursing', 'inpatient', 'facility'
    ]):
        return 'Facility transfer'
    return 'Other/Unknown'

def source_group(code):
    if code == 7:
        return 'Emergency room'
    if code in (1, 2, 3):
        return 'Referral'
    if code in (4, 5, 6, 10, 22, 25):
        return 'Transfer'
    return 'Other/Unknown'

def top_categories(values, count):
    series = (
        values.fillna('Missing')
        .replace({'Unknown': 'Missing'})
        .astype(str)
    )
    top = series[series.ne('Missing')].value_counts().head(count).index
    return series.where(
        series.isin(top) | series.eq('Missing'),
        'Other'
    )

features = pd.DataFrame(index=df.index)

features['age_num'] = (
    pd.to_numeric(
        df['age'].str.extract(r'\[(\d+)')[0],
        errors='coerce'
    ) + 5
)

for column in util_cols + [
    'n_meds_changed', 'n_diabetes_meds'
]:
    features[column] = pd.to_numeric(
        df[column], errors='coerce'
    )

features['total_prior_visits'] = (
    features[[
        'number_outpatient',
        'number_emergency',
        'number_inpatient'
    ]].sum(axis=1)
)

a1c_clean = (
    df['A1Cresult'].fillna('not tested')
    .astype(str).str.strip().str.lower()
)

glucose_clean = (
    df['max_glu_serum'].fillna('not tested')
    .astype(str).str.strip().str.lower()
)

features['a1c_tested'] = (
    ~a1c_clean.isin([
        'not tested', 'none', 'nan', 'missing', ''
    ])
).astype(int)

features['a1c_high'] = (
    df['A1Cresult'].isin(['>7', '>8']).astype(int)
)

features['glucose_tested'] = (
    ~glucose_clean.isin([
        'not tested', 'none', 'nan', 'missing', ''
    ])
).astype(int)

features['insulin_ordered'] = (
    df['insulin'].ne('No').astype(int)
)

features['med_changed'] = (
    df['change'].eq('Ch').astype(int)
)

features['any_diabetes_med'] = (
    df['diabetesMed'].eq('Yes').astype(int)
)

features['female'] = (
    df['gender'].eq('Female').astype(int)
)

admission_group = (
    df['admission_type_id']
    .map(admission_codes)
    .fillna('Unknown')
)

admission_group = admission_group.where(
    admission_group.isin([
        'Emergency', 'Urgent', 'Elective'
    ]),
    'Other/Unknown'
)

categories = pd.DataFrame(index=df.index)
categories['race'] = df['race'].fillna('Unknown')
categories['admission_type'] = admission_group

categories['discharge_group'] = (
    df['discharge_disposition_id']
    .apply(discharge_group)
)

categories['source_group'] = (
    df['admission_source_id']
    .apply(source_group)
)

categories['diag_group'] = df['diag_group']

categories['payer'] = top_categories(
    df['payer_code'], 6
)

categories['specialty'] = top_categories(
    df['medical_specialty'], 8
)

encoded = pd.get_dummies(
    categories,
    prefix_sep='__',
    dtype=float
)

X = pd.concat(
    [features, encoded],
    axis=1
).astype(float)

X = X.fillna(
    X.median(numeric_only=True)
).fillna(0)

y = df['readmit_30'].astype(int)

assert X.index.equals(y.index)
assert not X.isna().any().any()
assert not {
    'encounter_id', 'patient_nbr',
    'readmitted', 'readmit_30'
}.intersection(X.columns)

feature_dictionary = pd.DataFrame({
    'feature': X.columns,
    'type': [
        'one-hot' if c in encoded.columns
        else 'numeric/binary'
        for c in X.columns
    ],
    'description': [
        'Encoded category: ' + c
        if c in encoded.columns
        else 'Clinical predictor: ' + c
        for c in X.columns
    ]
})

feature_dictionary.to_csv(
      'feature_dictionary.csv',
    index=False
)

cohort_features = X.copy()
cohort_features['readmit_30'] = y

for column in ['race', 'gender', 'age']:
    cohort_features[column] = df[column]

cohort_features.to_csv(
    'cohort_features.csv',
    index=False
)

correlations = (
    X.corrwith(y)
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

negative = correlations.nsmallest(10)
positive = correlations.nlargest(10)

fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        '10 most negative correlations',
        '10 most positive correlations'
    )
)

fig5.add_trace(
    go.Bar(
        x=negative.values,
        y=negative.index,
        orientation='h',
        marker_color=BLUE
    ),
    row=1, col=1
)

fig5.add_trace(
    go.Bar(
        x=positive.values,
        y=positive.index,
        orientation='h',
        marker_color=RED
    ),
    row=1, col=2
)

fig5.update_layout(
    title='Analysis 5: Feature Correlations',
    height=620,
    showlegend=False,
    margin=dict(l=190)
)

fig5.write_html(
     'analysis5_feature_correlations.html'
)

print(
    'Feature matrix:', X.shape,
    'Positive rate:', round(y.mean(), 4)
)

print('Analysis 5 complete')
fig5.show()




# %%
# ==============================================================================
# ANALYSIS 6: BASELINE MODEL
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 6: BASELINE MODEL (LOGISTIC REGRESSION)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

g_test = df.loc[
    X_test.index,
    ['race', 'gender', 'age']
].copy()

lr = Pipeline([
    ('scale', StandardScaler()),
    (
        'lr',
        LogisticRegression(
            max_iter=2000,
            C=0.1,
            class_weight='balanced',
            random_state=42
        )
    )
])

lr.fit(X_train, y_train)

p_lr = lr.predict_proba(X_test)[:, 1]
pred_lr = (p_lr >= 0.5).astype(int)

res_lr = {
    'Model': 'Logistic Regression (balanced)',
    'ROC AUC': round(roc_auc_score(y_test, p_lr), 3),
    'PR AUC': round(average_precision_score(y_test, p_lr), 3),
    'Brier': round(brier_score_loss(y_test, p_lr), 4),
    'Sensitivity@0.5': round(recall_score(y_test, pred_lr), 3),
    'Precision@0.5': round(
        precision_score(y_test, pred_lr, zero_division=0), 3
    )
}

print('Train:', len(X_train), 'Test:', len(X_test))
print('Baseline metrics:')
print(pd.DataFrame([res_lr]).to_string(index=False))

fpr, tpr, _ = roc_curve(y_test, p_lr)

precision, recall, _ = precision_recall_curve(
    y_test, p_lr
)

fig6 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=('ROC Curve', 'Precision-Recall Curve')
)

fig6.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        line_color=BLUE,
        name='Logistic regression'
    ),
    row=1, col=1
)

fig6.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        line=dict(color=GRAY, dash='dash'),
        showlegend=False
    ),
    row=1, col=1
)

fig6.add_trace(
    go.Scatter(
        x=recall,
        y=precision,
        line_color=RED,
        name='Logistic regression'
    ),
    row=1, col=2
)

fig6.add_hline(
    y=y_test.mean(),
    line_dash='dash',
    line_color=GRAY,
    row=1, col=2
)

fig6.update_layout(
    title='Analysis 6: Baseline Logistic Regression',
    height=510
)

fig6.write_html(
     'analysis6_baseline_model.html'
)

coefficients = pd.Series(
    lr.named_steps['lr'].coef_[0],
    index=X.columns
)

print('\n10 strongest positive coefficients:')
print(coefficients.nlargest(10).round(3))

print('\n10 strongest negative coefficients:')
print(coefficients.nsmallest(10).round(3))

X_test.to_csv(
     'analysis6_X_test.csv',
    index=True,
    index_label='row_index'
)

y_test.to_frame('readmit_30').to_csv(
     'analysis6_y_test.csv',
    index=True,
    index_label='row_index'
)

g_test.to_csv(
     'analysis6_g_test.csv',
    index=True,
    index_label='row_index'
)

print('Analysis 6 complete')
fig6.show()




# %%
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



# Check that features and outcomes remain aligned.
assert X_train.columns.equals(X_test.columns)
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)

# Use the same folds for each model.
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

model_templates = {
    "Logistic Regression (balanced)": clone(lr),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),

    "Histogram Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        early_stopping=True,
        random_state=42
    )
}

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score"
}

# Cross-validation
cv_rows = []

for name, estimator in model_templates.items():
    print(f"Running 5-fold cross-validation: {name}", flush=True)

    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    cv_rows.append({
        "Model": name,
        "CV ROC AUC": scores["test_roc_auc"].mean(),
        "CV ROC AUC SD": scores["test_roc_auc"].std(ddof=1),
        "CV PR AUC": scores["test_pr_auc"].mean(),
        "CV Brier": -scores["test_brier"].mean(),
        "CV fit seconds": scores["fit_time"].mean()
    })

    print(
        f"  Mean CV ROC AUC: {cv_rows[-1]['CV ROC AUC']:.3f}",
        flush=True
    )

cv_results = pd.DataFrame(cv_rows)

# Choose from the training CV results.
# Within 0.005 AUC of the best score, prefer lower Brier, then faster fitting.
eligible = cv_results[
    cv_results["CV ROC AUC"]
    >= cv_results["CV ROC AUC"].max() - 0.005
]

best_model_name = eligible.sort_values(
    ["CV Brier", "CV fit seconds"]
).iloc[0]["Model"]

# Fit the models and score the test set.
trained_models = {}
test_probabilities = {}
test_rows = []

for name, estimator in model_templates.items():
    print(f"Fitting full training set: {name}", flush=True)

    started = perf_counter()
    model = clone(estimator).fit(X_train, y_train)
    fit_seconds = perf_counter() - started

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    trained_models[name] = model
    test_probabilities[name] = p

    test_rows.append({
        "Model": name,
        "ROC AUC": roc_auc_score(y_test, p),
        "PR AUC": average_precision_score(y_test, p),
        "Brier": brier_score_loss(y_test, p),
        "Sensitivity@0.5": recall_score(
            y_test, pred, zero_division=0
        ),
        "Precision@0.5": precision_score(
            y_test, pred, zero_division=0
        ),
        "Fit seconds": fit_seconds
    })

# Keep the final model and its predictions.
best_model = trained_models[best_model_name]
p_best = test_probabilities[best_model_name].copy()

model_results = cv_results.merge(
    pd.DataFrame(test_rows), on="Model"
)
model_results["Selected"] = (
    model_results["Model"] == best_model_name
)
model_results.to_csv("model_results.csv", index=False)

# ROC and calibration plots
fig7 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Test ROC curves",
        "Test calibration curves"
    )
)

colors = [BLUE, RED, GREEN]

for (name, p), color in zip(test_probabilities.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, p)

    observed, predicted = calibration_curve(
        y_test, p, n_bins=10, strategy="quantile"
    )

    fig7.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            legendgroup=name,
            name=f"{name} (AUC {roc_auc_score(y_test, p):.3f})",
            line=dict(color=color)
        ),
        row=1, col=1
    )

    fig7.add_trace(
        go.Scatter(
            x=predicted,
            y=observed,
            mode="lines+markers",
            legendgroup=name,
            name=name,
            showlegend=False,
            line=dict(color=color)
        ),
        row=1, col=2
    )

for col in (1, 2):
    fig7.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            showlegend=False,
            line=dict(color=GRAY, dash="dash")
        ),
        row=1, col=col
    )
    fig7.update_xaxes(range=[0, 1], row=1, col=col)
    fig7.update_yaxes(range=[0, 1], row=1, col=col)

fig7.update_xaxes(
    title_text="False positive rate", row=1, col=1
)
fig7.update_yaxes(
    title_text="Sensitivity", row=1, col=1
)
fig7.update_xaxes(
    title_text="Mean predicted probability", row=1, col=2
)
fig7.update_yaxes(
    title_text="Observed readmission rate", row=1, col=2
)

fig7.update_layout(
    title="Analysis 7: Advanced Models and Comparison",
    template="plotly_white",
    height=580,
    legend=dict(orientation="h", y=-0.25),
    margin=dict(b=140)
)

fig7.write_html(
    "analysis7_model_comparison.html",
    include_plotlyjs=True
)
fig7.show()

print(model_results.round(4).to_string(index=False))
print("\nSelected using training cross-validation:", best_model_name)
print("PR AUC is reported as average precision.")
print("The 0.5 threshold metrics are not the 15% outreach operating point.")
print("\nSaved: model_results.csv")
print("Saved: analysis7_model_comparison.html")

print("Analysis 7 complete")



# %%
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

from sklearn.inspection import permutation_importance, partial_dependence

print("Calculating feature importance—give it a few minutes.", flush=True)

# Feature importance
importance_results = {}

for label, model in [("selected", best_model), ("logistic", lr)]:
    print("Explaining:", label, flush=True)

    importance_results[label] = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

lr_coefficients = pd.Series(
    lr.named_steps["lr"].coef_[0],
    index=X_test.columns
)

feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": importance_results["selected"].importances_mean,
    "importance_std": importance_results["selected"].importances_std,
    "lr_importance_mean": importance_results["logistic"].importances_mean,
    "lr_importance_std": importance_results["logistic"].importances_std,
    "lr_coefficient": lr_coefficients.to_numpy()
}).sort_values(
    "importance_mean", ascending=False
).reset_index(drop=True)

feature_importance.to_csv(
    "feature_importance.csv", index=False
)

# Partial dependence
pdp_feature = "number_inpatient"
print("Calculating partial dependence:", pdp_feature, flush=True)

pdp = partial_dependence(
    best_model,
    X_test,
    features=[pdp_feature],
    method="brute",
    response_method="predict_proba",
    grid_resolution=20,
    percentiles=(0.0, 0.99)
)

pdp_x = pdp["grid_values"][0]
pdp_y = pdp["average"][0]

# Plot the results.
top15 = feature_importance.head(15).iloc[::-1]

fig8 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Top 15 features: permutation importance",
        "Partial dependence: prior inpatient visits"
    ),
    horizontal_spacing=0.18,
    column_widths=[0.58, 0.42]
)

fig8.add_trace(
    go.Bar(
        x=top15["importance_mean"],
        y=top15["feature"],
        orientation="h",
        name="Selected model",
        marker_color=PURPLE,
        error_x=dict(
            type="data",
            array=top15["importance_std"]
        )
    ),
    row=1, col=1
)

fig8.add_trace(
    go.Bar(
        x=top15["lr_importance_mean"],
        y=top15["feature"],
        orientation="h",
        name="Logistic regression",
        marker_color=BLUE,
        error_x=dict(
            type="data",
            array=top15["lr_importance_std"]
        )
    ),
    row=1, col=1
)

fig8.add_trace(
    go.Scatter(
        x=pdp_x,
        y=pdp_y,
        mode="lines+markers",
        name=pdp_feature,
        showlegend=False,
        line=dict(color=GREEN)
    ),
    row=1, col=2
)

fig8.add_vline(
    x=0,
    line_dash="dash",
    line_color=GRAY,
    row=1, col=1
)

fig8.update_xaxes(
    title_text="Decrease in ROC AUC after shuffling",
    row=1, col=1
)
fig8.update_xaxes(
    title_text="Prior inpatient visits",
    row=1, col=2
)
fig8.update_yaxes(
    title_text="Average predicted readmission risk",
    tickformat=".1%",
    row=1, col=2
)

fig8.update_layout(
    title=f"Analysis 8: Explaining {best_model_name}",
    template="plotly_white",
    barmode="group",
    height=800,
    margin=dict(l=280, r=40, t=100, b=140),
    legend=dict(orientation="h", y=-0.15)
)

fig8.add_annotation(
    x=0,
    y=-0.25,
    xref="paper",
    yref="paper",
    showarrow=False,
    xanchor="left",
    font=dict(size=11),
    text=(
        "Error bars: standard deviation across five shuffles. "
        "Partial dependence is not a causal effect."
    )
)

fig8.write_html(
    "analysis8_explainability.html",
    include_plotlyjs=True
)
fig8.show()

# Compare the main drivers with logistic regression.
selected_top10 = feature_importance.head(10)["feature"].tolist()

lr_top10 = feature_importance.nlargest(
    10, "lr_importance_mean"
)["feature"].tolist()

shared_features = [
    feature for feature in selected_top10
    if feature in lr_top10
]

coefficient_top10 = (
    lr_coefficients.abs().nlargest(10).index.tolist()
)

shared_coefficients = [
    feature for feature in selected_top10
    if feature in coefficient_top10
]

print("\nTop 15 features:")
print(
    feature_importance.head(15).round(5).to_string(index=False)
)

print(
    f"\nShared top-10 permutation features: "
    f"{len(shared_features)}/10"
)
print(", ".join(shared_features))

print("\nOverlap with the baseline's strongest coefficients:")
print(", ".join(shared_coefficients))

print("\nClinical interpretation:")
print("Prior utilization can reflect illness burden and care needs.")
print("Discharge setting can reflect care needs and available support.")
print("These are associations; correlated features can share importance.")

pd.DataFrame(
    {"p_best": p_best},
    index=X_test.index
).rename_axis("row_index").to_csv("analysis7_p_best.csv")

print("\nSaved: feature_importance.csv")
print("Saved: analysis8_explainability.html")
print("Saved: analysis7_p_best.csv")

print("Analysis 8 complete.")


# %%
# ==============================================================================
# ANALYSIS 9: FAIRNESS & BIAS AUDIT
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 9: FAIRNESS & BIAS AUDIT")
print("=" * 70)
# TODO: Audit the final model across race, gender and age groups (use g_test)
#   - Pick an operating point: e.g., flag the top 15% highest-risk patients
#     thr = np.quantile(p_best, 0.85)
p_s = pd.Series(p_best, index=X_test.index, name='p')
y_s = pd.Series(np.asarray(y_test), index=X_test.index, name='y')
audit = g_test.copy()
audit['p'], audit['y'] = p_s, y_s

FLAG_TOP_FRAC = 0.15          # operating point: flag top 15% highest-risk patients
MIN_GROUP_N = 100
GROUP_COLS = ['race', 'gender', 'age']

# --- Operating point ---------------------------------------------------------
thr = np.quantile(p_best, 1 - FLAG_TOP_FRAC)
audit['flag'] = (audit['p'] >= thr).astype(int)
print(f"Operating point: flag top {FLAG_TOP_FRAC:.0%} -> threshold = {thr:.4f}")


# --- Metrics helper ----------------------------------------------------------
def group_metrics(d: pd.DataFrame) -> dict:
    y, f, p = d['y'].astype(int), d['flag'], d['p']
    a_lo, a_hi = boot_auc_ci(d) if y.nunique() == 2 else (np.nan, np.nan)
    tp = int(((f == 1) & (y == 1)).sum())
    fn = int(((f == 0) & (y == 1)).sum())
    fp = int(((f == 1) & (y == 0)).sum())
    tn = int(((f == 0) & (y == 0)).sum())
    return {
        'n': len(d),
        'n_pos': int(y.sum()),
        'prevalence': y.mean(),
        'mean_pred': p.mean(),
        'calib_ratio': p.mean() / y.mean() if y.mean() > 0 else np.nan,
        'auc': roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
        'auc_lo': a_lo, 'auc_hi': a_hi,
        'flag_rate': f.mean(),
        'sensitivity': tp / (tp + fn) if (tp + fn) else np.nan,
        'sens_lo': wilson(tp, tp + fn)[0], 'sens_hi': wilson(tp, tp + fn)[1],
        'fpr': fp / (fp + tn) if (fp + tn) else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) else np.nan,
        'ppv_lo': wilson(tp, tp + fp)[0], 'ppv_hi': wilson(tp, tp + fp)[1]
    }


# --- Wilson CI ----------------------------------------------------------
def wilson(k, n, z=1.96):
    """
    95% Wilson interval for a proportion.
    Gives a plausible range of values for a true population proportion based on binary sample data
    """
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)

# --- Bootstrap AUC CI ----------------------------------------------------------
def boot_auc_ci(d, B=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, p = d['y'].to_numpy(), d['p'].to_numpy()
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if len(vals) > 50 else (np.nan, np.nan)


# %%
# --- Build audit table -------------------------------------------------------
rows = []
overall = group_metrics(audit)
rows.append({'attribute': 'ALL', 'group': 'Overall', **overall})

#   - For each group with >= 100 test patients compute: n, prevalence, ROC AUC, flag rate,
#     sensitivity, false positive rate, PPV. Table -> 'fairness_audit.csv'
for col in GROUP_COLS:
    levels = audit[col].astype(str)
    for lvl in sorted(levels.unique()):
        sub = audit[levels == lvl]
        if len(sub) < MIN_GROUP_N:
            continue
        rows.append({'attribute': col, 'group': lvl, **group_metrics(sub)})

fair = pd.DataFrame(rows)
fair['reliable'] = fair['n_pos'] >= 30      # fewer than 30 readmissions = noise

# Gaps relative to overall (useful for the discussion)
fair['sens_gap_vs_overall'] = fair['sensitivity'] - overall['sensitivity']
fair['flag_to_prev_ratio'] = fair['flag_rate'] / fair['prevalence']
fair['rel_flag_ratio'] = fair['flag_to_prev_ratio'] / (overall['flag_rate'] / overall['prevalence'])

fair.to_csv('fairness_audit.csv', index=False)

with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
    print(fair.to_string(index=False))

skipped = {c: sorted(audit[c].astype(str).value_counts()[lambda s: s < MIN_GROUP_N].index)
           for c in GROUP_COLS}
skipped_txt = "; ".join(f"{c}: {', '.join(v)}" for c, v in skipped.items() if v) or "none"

print("\nGroups skipped (n < %d):" % MIN_GROUP_N, {k: v for k, v in skipped.items() if v})


#   - 1x3 figure: AUC by group | sensitivity by group | flag rate vs prevalence scatter
#     -> 'analysis9_fairness_audit.html'
plot_df = fair[fair['attribute'] != 'ALL'].copy()
plot_df['label'] = plot_df['attribute'] + ': ' + plot_df['group'] + np.where(plot_df['reliable'], '', ' †')
colors = {'race': '#1f77b4', 'gender': '#ff7f0e', 'age': '#2ca02c'}

fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.12,
    subplot_titles=("ROC AUC by group", "Sensitivity (recall) by group",
                    "Flag rate vs. prevalence"))

for attr in GROUP_COLS:
    d = plot_df[plot_df['attribute'] == attr]
    fig.add_trace(go.Bar(x=d['label'], y=d['auc'], name=attr, marker_color=colors[attr],
                         legendgroup=attr, customdata=d['n'],
                         error_y=dict(type='data', symmetric=False,
                                      array=d['auc_hi'] - d['auc'],
                                      arrayminus=d['auc'] - d['auc_lo']),
                         hovertemplate="%{x}<br>AUC=%{y:.3f}<br>n=%{customdata}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=d['label'], y=d['sensitivity'], name=attr, marker_color=colors[attr],
                         legendgroup=attr, showlegend=False, customdata=d['n'],
                         error_y=dict(type='data', symmetric=False,
                                      array=d['sens_hi'] - d['sensitivity'],
                                      arrayminus=d['sensitivity'] - d['sens_lo']),
                         hovertemplate="%{x}<br>Sens=%{y:.3f}<br>n=%{customdata}<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=d['prevalence'], y=d['flag_rate'], mode='markers+text',
                             text=d['group'], textposition='top center',
                             marker=dict(size=np.clip(d['n'] / d['n'].max() * 30, 8, 30),
                                         color=colors[attr], opacity=0.8),
                             name=attr, legendgroup=attr, showlegend=False,
                             hovertemplate="%{text}<br>prev=%{x:.3f}<br>flag=%{y:.3f}<extra></extra>"),
                  row=1, col=3)

# Reference lines
fig.add_hline(y=overall['auc'], line_dash='dash', line_color='gray', row=1, col=1)
fig.add_hline(y=overall['sensitivity'], line_dash='dash', line_color='gray', row=1, col=2)

# Scatter reference: flag rate proportional to prevalence (points below = under-flagged)
xmax = max(plot_df['prevalence'].max(), overall['prevalence']) * 1.15
xs = np.linspace(0, xmax, 10)
fig.add_trace(go.Scatter(x=xs, y=xs * overall['flag_rate'] / overall['prevalence'],
                         mode='lines', line=dict(dash='dash', color='gray'),
                         name='flag ∝ prevalence', showlegend=False), row=1, col=3)

fig.update_xaxes(tickangle=-45, row=1, col=1)
fig.update_xaxes(tickangle=-45, row=1, col=2)
fig.update_xaxes(title_text='Prevalence (30-day readmission)', row=1, col=3)
fig.update_yaxes(title_text='ROC AUC', range=[0.4, 1.0], row=1, col=1)
fig.update_yaxes(title_text='Sensitivity', range=[0, 1], row=1, col=2)
fig.update_yaxes(title_text='Flag rate', row=1, col=3)
fig.update_layout(
    title=f"Analysis 9: Fairness audit at top-{FLAG_TOP_FRAC:.0%} operating point (threshold={thr:.3f})",
    height=550, 
    width=1700, 
    template='plotly_white', 
    legend_title_text='Attribute',
    margin=dict(b=170)
)
fig.write_html('analysis9_fairness_audit.html')
print("Saved: fairness_audit.csv, analysis9_fairness_audit.html")

#   - Discuss: which groups are under-flagged relative to their risk? Is that a model problem,
#     a data problem, or both? What would you do about it before deployment?
# Hint: pd.Series(p_best, index=X_test.index) lets you align probabilities with g_test
show = ['attribute','group','n','n_pos','prevalence','mean_pred','flag_rate',
        'sensitivity','sens_lo','sens_hi','auc','rel_flag_ratio']
rel = plot_df[plot_df['reliable']].sort_values('rel_flag_ratio')
print("\nUnder-flagged (n_pos >= 30):")
print(rel[show].head(4).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nOver-flagged (n_pos >= 30):")
print(rel[show].tail(4).iloc[::-1].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

worst_sens = rel.loc[rel['sensitivity'].idxmin()]
print(f"\nLowest sensitivity: {worst_sens['attribute']}={worst_sens['group']} "
      f"({worst_sens['sensitivity']:.3f} vs overall {overall['sensitivity']:.3f})")


print("Analysis 9 complete")

# %%
# Sandbox for mitigation
def sens_by(a, col):
    g = a.groupby(col)
    out = pd.DataFrame({
        'n': g.size(), 'n_pos': g['y'].sum(), 'flag_rate': g['flag'].mean(),
        'sensitivity': g.apply(lambda d: d.loc[d['flag'] == 1, 'y'].sum() / max(d['y'].sum(), 1))})
    return out[out['n'] >= MIN_GROUP_N].round(3)

# (a) Flag the top 15% WITHIN each age band (needs ethics/legal review: age enters the decision rule)
alt = audit.copy()
alt['flag'] = alt.groupby('age')['p'].transform(lambda s: (s >= s.quantile(1 - FLAG_TOP_FRAC)).astype(int))
alt_overall_sens = alt.loc[alt['flag'] == 1, 'y'].sum() / alt['y'].sum()
alt_age = sens_by(alt, 'age')
alt_age_rel = alt_age[alt_age['n_pos'] >= 30]            # same >=30 readmissions rule as the audit
alt_lo, alt_hi = alt_age_rel['sensitivity'].min(), alt_age_rel['sensitivity'].max()
print("Age-band thresholds, overall sensitivity:", round(alt_overall_sens, 3), "vs", round(overall['sensitivity'], 3))
print(alt_age)

# (b) Refit without age / sex / race inputs and re-audit
drop_cols = ['age_num', 'female'] + [c for c in X.columns if c.startswith('race__')]
m_abl = clone(best_model).fit(X_train.drop(columns=drop_cols), y_train)
alt2 = audit.copy()
alt2['p'] = m_abl.predict_proba(X_test.drop(columns=drop_cols))[:, 1]
alt2['flag'] = (alt2['p'] >= np.quantile(alt2['p'], 1 - FLAG_TOP_FRAC)).astype(int)
print("\nAUC without age/sex/race:", round(roc_auc_score(alt2['y'], alt2['p']), 3))

abl_auc = roc_auc_score(alt2['y'], alt2['p'])
abl_overall_sens = alt2.loc[alt2['flag'] == 1, 'y'].sum() / alt2['y'].sum()
abl_age = sens_by(alt2, 'age')
abl_age_rel = abl_age[abl_age['n_pos'] >= 30]
abl_lo, abl_hi = abl_age_rel['sensitivity'].min(), abl_age_rel['sensitivity'].max()

for c in ['age', 'gender', 'race']:
    print(sens_by(alt2, c))

# %%
# ==============================================================================
# ANALYSIS 10: DEPLOYMENT & CLINICAL INTEGRATION
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 15 + "ANALYSIS 10: DEPLOYMENT & CLINICAL INTEGRATION")
print("=" * 70)

CAPACITIES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]   # share of discharges the team can follow up
TIER_CUTS = {'High': 0.85, 'Medium': 0.60}          # percentile cut points (High = top 15%, matches Analysis 9)

# --- Rank all test patients by risk ------------------------------------------
ranked = audit[['p', 'y']].copy().sort_values('p', ascending=False, kind='mergesort')
y_sorted = ranked['y'].to_numpy(dtype=int)
N, P = len(ranked), int(y_sorted.sum())
prev = P / N
cum_tp = np.cumsum(y_sorted)

# %%
# TODO: Translate the model into an operational plan
#   - Capacity analysis: if the transitional-care team can follow up 5 / 10 / 15 / 20 / 30 / 50% of
cap_rows = []
for frac in CAPACITIES:
    k = int(math.ceil(frac * N))
    tp = int(cum_tp[k - 1])
    cap_rows.append({
        'capacity_pct': frac,
        'n_flagged': k,
        'risk_threshold': ranked['p'].iloc[k - 1],
        'readmissions_captured': tp,
        'readmissions_total': P,
        'pct_readmissions_captured': tp / P,           # = sensitivity
        'ppv': tp / k,
        'lift_vs_random': (tp / k) / prev,
        'flagged_per_readmission_captured': k / tp if tp else np.nan,
        'captured_per_1000_discharges': 1000 * tp / N,
        'flagged_per_1000_discharges': 1000 * k / N,
        'ppv_lo': wilson(tp, k)[0], 'ppv_hi': wilson(tp, k)[1],
        'sens_lo': wilson(tp, P)[0], 'sens_hi': wilson(tp, P)[1],
    })
cap = pd.DataFrame(cap_rows)
# ILLUSTRATIVE scenarios: assumed relative risk reduction among flagged patients (not a result)
for rrr in (0.10, 0.20):
    tag = int(rrr * 100)
    cap[f'prevented_per_1000_rrr{tag}'] = cap['captured_per_1000_discharges'] * rrr
    cap[f'flagged_per_prevented_rrr{tag}'] = cap['flagged_per_1000_discharges'] / cap[f'prevented_per_1000_rrr{tag}']
cap.to_csv('capacity_analysis.csv', index=False)

# Simple-rule baseline: any prior inpatient visit
rule = (X_test.loc[ranked.index, 'number_inpatient'] >= 1).to_numpy()
k_r, tp_r = int(rule.sum()), int(y_sorted[rule].sum())
print(f"Rule 'prior inpatient >= 1': flags {k_r/N:.1%}, captures {tp_r/P:.1%} (PPV {tp_r/k_r:.1%}); "
      f"model at the same volume captures {cum_tp[k_r-1]/P:.1%}")

#     discharges, how many 30-day readmissions would be captured, what is the PPV, and how many
#     patients must be flagged per readmission captured? Table -> 'capacity_analysis.csv'
print(f"Test set: {N:,} discharges, {P:,} 30-day readmissions (prevalence {prev:.3f})\n")
with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 220):
    print(cap.to_string(index=False))

#   - Cumulative gains chart (share flagged vs share of readmissions captured) with a random baseline
x_gain = np.r_[0, np.arange(1, N + 1) / N]
y_gain = np.r_[0, cum_tp / P]
x_perfect = np.array([0, prev, 1])
y_perfect = np.array([0, 1, 1])

#   - Risk tiers (Low / Medium / High) with observed readmission rate per tier
pct_rank = ranked['p'].rank(method='first', ascending=True, pct=True)
ranked['tier'] = np.select([pct_rank > TIER_CUTS['High'], pct_rank > TIER_CUTS['Medium']],
                           ['High', 'Medium'], default='Low')
TIER_CUTOFFS = {'High': ranked.loc[ranked['tier'] == 'High', 'p'].min(),
                'Medium': ranked.loc[ranked['tier'] == 'Medium', 'p'].min()}

tier_rows = []
for t in ['Low', 'Medium', 'High']:
    s = ranked[ranked['tier'] == t]
    k, n = int(s['y'].sum()), len(s)
    lo, hi = wilson(k, n)
    tier_rows.append({
        'tier': t, 'n': n, 'share_of_discharges': n / N,
        'p_min': s['p'].min(), 'p_max': s['p'].max(),
        'readmissions': k, 'observed_rate': k / n, 'ci_low': lo, 'ci_high': hi,
        'share_of_all_readmissions': k / P, 'lift_vs_random': (k / n) / prev,
    })
tiers = pd.DataFrame(tier_rows)
tiers.to_csv('risk_tiers.csv', index=False)
print("\nRisk tiers (observed 30-day readmission rate):")
with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 220):
    print(tiers.to_string(index=False))

# --- 4) Figure: gains | capacity trade-off | tiers ---------------------------
fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.10,
    subplot_titles=("Cumulative gains", "Capacity trade-off", "Observed readmission rate by tier"))

# Panel 1: gains
fig.add_trace(go.Scatter(x=x_gain, y=y_gain, mode='lines', name='Model',
                         line=dict(color='#1f77b4', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random baseline',
                         line=dict(color='gray', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=x_perfect, y=y_perfect, mode='lines', name='Perfect model',
                         line=dict(color='lightgray', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(
    x=cap['capacity_pct'], y=cap['pct_readmissions_captured'], mode='markers+text',
    text=[f"{c:.0%}" for c in cap['capacity_pct']], textposition='bottom right',
    marker=dict(size=10, color='#d62728'), name='Capacity options',
    hovertemplate="Follow up %{x:.0%}<br>Captured %{y:.1%}<extra></extra>"), row=1, col=1)

# Panel 2: sensitivity and PPV vs capacity
fig.add_trace(go.Scatter(x=cap['capacity_pct'], y=cap['pct_readmissions_captured'],
                         mode='lines+markers', name='% readmissions captured',
                         line=dict(color='#2ca02c')), row=1, col=2)
fig.add_trace(go.Scatter(x=cap['capacity_pct'], y=cap['ppv'], mode='lines+markers',
                         name='PPV', line=dict(color='#ff7f0e')), row=1, col=2)
fig.add_hline(y=prev, line_dash='dash', line_color='gray', row=1, col=2,
              annotation_text="prevalence (PPV if random)", annotation_position="top right")

# Panel 3: tiers
fig.add_trace(go.Bar(
    x=tiers['tier'], y=tiers['observed_rate'],
    error_y=dict(type='data', symmetric=False,
                 array=tiers['ci_high'] - tiers['observed_rate'],
                 arrayminus=tiers['observed_rate'] - tiers['ci_low']),
    marker_color=['#2ca02c', '#ffbf00', '#d62728'],
    text=[f"{r:.1%}<br>n={n:,}" for r, n in zip(tiers['observed_rate'], tiers['n'])],
    textposition='outside', name='Observed rate', showlegend=False), row=1, col=3)
fig.add_hline(y=prev, line_dash='dash', line_color='gray', row=1, col=3)

fig.update_xaxes(title_text='Share of discharges flagged', tickformat='.0%', row=1, col=1)
fig.update_yaxes(title_text='Share of readmissions captured', tickformat='.0%', row=1, col=1)
fig.update_xaxes(title_text='Follow-up capacity (% of discharges)', tickformat='.0%', row=1, col=2)
fig.update_yaxes(title_text='Proportion', tickformat='.0%', row=1, col=2)
fig.update_xaxes(title_text='Risk tier (Low <60th pct, Medium 60-85th, High >85th)', row=1, col=3)
fig.update_yaxes(title_text='30-day readmission rate', tickformat='.0%', row=1, col=3)
fig.update_layout(
    title="Analysis 10: Deployment planning",
    height=550, width=1700, template='plotly_white', margin=dict(b=100),
    legend=dict(orientation='h', y=-0.25))
#   - Figure -> 'analysis10_deployment.html'
fig.write_html('analysis10_deployment.html')
print("\nSaved: capacity_analysis.csv, risk_tiers.csv, analysis10_deployment.html")

# %%
#   - Write a MODEL CARD (model_card.md): intended use, training data, performance, fairness,
#     limitations, monitoring plan, who is accountable. A template is in the instructions.auc_all = roc_auc_score(ranked['y'], ranked['p'])
auc_all = roc_auc_score(ranked['y'], ranked['p'])
pr_auc = average_precision_score(ranked['y'], ranked['p'])
brier = brier_score_loss(ranked['y'], ranked['p'])
mean_p = ranked['p'].mean()
row15 = cap.loc[np.isclose(cap['capacity_pct'], 0.15)].iloc[0]
model_name = type(best_model).__name__

brier_ref = prev * (1 - prev)
bss = 1 - brier / brier_ref
p_clip = np.clip(ranked['p'].to_numpy(), 1e-6, 1 - 1e-6)
_cal = LogisticRegression(C=1e6).fit(np.log(p_clip / (1 - p_clip)).reshape(-1, 1), ranked['y'])
cal_slope, cal_int = _cal.coef_[0][0], _cal.intercept_[0]
auc_lo, auc_hi = overall['auc_lo'], overall['auc_hi']
lr_auc = model_results.loc[model_results['Model'].str.startswith('Logistic'), 'ROC AUC'].iloc[0]
cmp_lines = "\n".join(
    f"| {r['Model']} | {r['CV ROC AUC']:.3f} | {r['ROC AUC']:.3f} | {r['PR AUC']:.3f} | {r['Brier']:.4f} |"
    for _, r in model_results.iterrows())

#Fairness results for model card
fp = fair[(fair['attribute'] != 'ALL') & fair['reliable']].dropna(subset=['sensitivity'])
lo_s, hi_s = fp.loc[fp['sensitivity'].idxmin()], fp.loc[fp['sensitivity'].idxmax()]
lo_a, hi_a = fp.loc[fp['auc'].idxmin()], fp.loc[fp['auc'].idxmax()]
age_r = fair[(fair['attribute'] == 'age') & fair['reliable']]
gen = fair[fair['attribute'] == 'gender'].set_index('group')
fairness_txt = (
    f"- Audited at the top-15% operating point (score >= {thr:.3f}) across race, gender and age. "
    f"Not evaluated (n < {MIN_GROUP_N}): {skipped_txt}. † = fewer than 30 readmissions; unreliable.\n"
    f"- Among groups with >= 30 readmissions, sensitivity ranges {lo_s['sensitivity']:.1%} ({lo_s['attribute']}={lo_s['group']}) "
    f"to {hi_s['sensitivity']:.1%} ({hi_s['attribute']}={hi_s['group']}); overall {overall['sensitivity']:.1%}. "
    f"AUC ranges {lo_a['auc']:.3f} to {hi_a['auc']:.3f} over the same groups.\n"
    f"- Age is the main disparity: flag rate spans {age_r['flag_rate'].min():.1%}-{age_r['flag_rate'].max():.1%} "
    f"but prevalence only {age_r['prevalence'].min():.1%}-{age_r['prevalence'].max():.1%}, so true risk differences do not explain it. "
    f"Ages 30-50 are under-flagged and 80-90 over-flagged.\n"
    f"- Sex gap: sensitivity {gen.loc['Female','sensitivity']:.1%} (female) vs {gen.loc['Male','sensitivity']:.1%} (male).\n"
    f"- Age, sex and race are model inputs. Mitigations considered: (1) fixed-threshold monitoring by group; "
    f"(2) refit without age/sex/race inputs: test AUC {abl_auc:.3f} (vs {auc_all:.3f}), overall sensitivity {abl_overall_sens:.1%}, "
    f"age-group sensitivity range {abl_lo:.1%}-{abl_hi:.1%} (current model: {age_r['sensitivity'].min():.1%}-{age_r['sensitivity'].max():.1%}); "
    f"(3) age-band thresholds (flag the top {FLAG_TOP_FRAC:.0%} within each age band): overall sensitivity {alt_overall_sens:.1%} "
    f"(vs {overall['sensitivity']:.1%}), age-group range {alt_lo:.1%}-{alt_hi:.1%}; age would enter the decision rule, so this needs ethics/legal review; "
    f"(4) local validation with this same audit before go-live; (5) clinician override, and no use to deny services."
)
sub_lines = "\n".join(
    f"| {r.attribute}={r.group}{'' if r.n_pos >= 30 else ' †'} | {r.n:,} | {r.n_pos} | {r.flag_rate:.1%} | "
    f"{r.sensitivity:.1%} ({r.sens_lo:.0%}-{r.sens_hi:.0%}) | {r.ppv:.1%} | {r.auc:.3f} |"
    for r in fair[fair['attribute'] != 'ALL'].itertuples())

TIER_ACTION = {'High': 'Care-team call within 48 h and visit within 7 days (proposed)',
               'Medium': 'Automated outreach or pharmacist med-reconciliation call (proposed)',
               'Low': 'Standard discharge process'}

tier_lines = "\n".join(
    f"| {r.tier} | {r.p_min:.3f}-{r.p_max:.3f} | {r.share_of_discharges:.0%} | "
    f"{r.observed_rate:.1%} ({r.ci_low:.1%}-{r.ci_high:.1%}) | {r.share_of_all_readmissions:.0%} | {TIER_ACTION[r.tier]} |"
for r in tiers.itertuples())

#   - In the report: where in the workflow does the score appear, who acts on it, what is the
#     intervention, how will you know it is working, and what regulatory/privacy considerations apply
model_card = f"""# Model Card: 30-Day Hospital Readmission Risk Model

**Version:** 0.1 (draft) | **Date:** {date.today().isoformat()} | **Model:** {model_name} | **Team:** {TEAM_NAME}

**Model details:** HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True; {getattr(best_model, 'n_iter_', 'n/a')} iterations used), scikit-learn, random_state={RANDOM_STATE}. Chosen on training-set 5-fold CV (AUC within 0.005 of the best, then Brier), not on the test set.

| Model | CV ROC AUC | Test ROC AUC | Test PR AUC | Test Brier |
|---|---|---|---|---|
{cmp_lines}

-Gain over the logistic baseline is {auc_all - lr_auc:+.3f} test AUC; the choice rests on calibration and speed, not discrimination.

## 1. Intended use
- **Purpose:** Rank inpatients with diabetes at discharge by risk of any readmission within 30 days (the label cannot separate planned from unplanned) ...
- **Intended users:** Transitional-care nurses, care coordinators and discharge planners. The score supports, and does not replace, clinical judgment.
- **Intended population:** Diabetes-related inpatient encounters. Excluded from the cohort: discharges to death, hospice or unmapped disposition. Transfers were kept. The data include 655 patients under 20; no performance claim is made for them and they are out of scope.
- **Out of scope:** Denying, delaying or rationing care; pricing, insurance or coverage decisions; use on pediatric, obstetric or non-diabetic populations; use as the sole basis for any clinical decision.

## 2. Training data
- **Source:** Diabetes 130-US hospitals dataset (diabetic_data.csv), encounters 1999-2008.
- **Cohort rules:** {n0:,} raw encounters / {n1:,} patients -> first encounter per patient (prevents patient-level leakage across the split) -> {n2:,} after excluding death/hospice/unmapped discharges. Positive rate {df['readmit_30'].mean():.1%}.
- **Outcome:** `readmit_30` (readmitted within 30 days). Readmissions to other hospitals may be missed, which causes label noise.
- **Features:** {X_test.shape[1]} structured features (utilization, diagnosis group, medications, labs, discharge/admission codes, payer, specialty). Age, sex and race are included as inputs.
- **Split:** 80/20 stratified ({len(X_train):,} train / {len(X_test):,} test), random_state={RANDOM_STATE}; one row per patient. All numbers here are test-set.
- **Missing data (raw):** race 2.2% -> 'Unknown'; payer 39.6% and specialty 49.1% -> 'Missing'/'Unknown'; HbA1c 83% and glucose 95% -> 'not tested'; weight (97%) dropped.

## 3. Performance (held-out test set, n={N:,}, prevalence {prev:.1%})
| Metric | Value |
|---|---|
| ROC AUC (95% bootstrap CI) | {auc_all:.3f} ({auc_lo:.3f}-{auc_hi:.3f}) |
| PR AUC (average precision) | {pr_auc:.3f} |
| Brier score (base-rate reference {brier_ref:.4f}; skill {bss:.1%}) | {brier:.4f} |
| Calibration slope / intercept | {cal_slope:.2f} / {cal_int:.2f} |
| Mean predicted risk vs. observed prevalence | {mean_p:.3f} vs {prev:.3f} |

**At the operating point (follow up top 15% of discharges):** captures {row15['pct_readmissions_captured']:.1%} of readmissions, PPV {row15['ppv']:.1%} ({row15['lift_vs_random']:.1f}x random), about {row15['flagged_per_readmission_captured']:.1f} patients flagged per readmission captured.
Operating point: score >= {thr:.3f} (about the top {FLAG_TOP_FRAC:.0%} on test). Deploy this fixed threshold, not a rolling percentile. Tier cutoffs: High >= {TIER_CUTOFFS['High']:.3f}, Medium >= {TIER_CUTOFFS['Medium']:.3f}.

| Risk tier | Predicted risk | Share of discharges | Observed readmission rate (95% CI) | Share of all readmissions | Proposed action |
|---|---|---|---|---|---|
{tier_lines}

Full capacity table: `capacity_analysis.csv`.

## 4. Fairness
{fairness_txt}

| Group | n | Readmissions | Flag rate | Sensitivity (95% CI) | PPV | AUC |
|---|---|---|---|---|---|---|
{sub_lines}

† Fewer than 30 readmissions in the test set; estimates are very imprecise and are shown for completeness only.


## 5. Limitations
- Data are from 1999-2008 and from diabetes admissions only; coding practices, care pathways and population have since changed.
- The label is any readmission within 30 days (planned and unplanned are not distinguished) and misses readmissions elsewhere.
- Age, sex and race are inputs, and sensitivity varies materially by age (see Section 4).
- No social-needs, adherence or functional-status data.

## 6. Monitoring plan
| What | How often | Trigger for review |
|---|---|---|
| Input data quality (missingness, ranges, schema) | Weekly | Any feature missing > 2x baseline |
| Score distribution and flag rate | Weekly | Flag rate outside 12%-18% (15% target +/-20%) | Model owner |
| AUC, sensitivity, PPV, calibration by race/gender/age | Monthly | AUC drop > 0.03 (test CI is wider than small drops); any group sensitivity gap > 10 points over a rolling quarter (min 100 readmissions) | Model owner + governance |
| Intervention uptake (flagged patients reached within 48 h) | Weekly | < 80% reached |
| Outcome impact (High tier observed vs. risk-adjusted expected, or randomized hold-out) | Quarterly | No improvement after 2 quarters | Clinical owner |
| Full revalidation and retraining decision | Every 6-12 months or after major workflow/coding change | Any of the above persist |

## 7. Accountability
- **Technical (model) owner:** ** Director of Clinical Analytics (proposed role). Responsible for the scoring pipeline, the fixed threshold ({thr:.3f}) and tier cutoffs, the monitoring dashboard in Section 6, the monthly subgroup audit, retraining and revalidation decisions, versioned model and card documentation, and a rollback plan (switch the score off and revert to the standard discharge process).
- **Override:** clinicians can dismiss a flag with a logged reason code; override rates are reviewed monthly.
- **Clinical owner:** Transitional Care Medical Director (proposed role). Owns the tier-to-action protocol, staffing for follow-up capacity, clinical safety review, and the evaluation of whether the intervention reduces readmissions.
- **Governance:** AI/Model Governance Committee (clinical, ethics, privacy and compliance members, proposed). Approves go-live and any change to inputs or thresholds (including age-band thresholds), reviews the quarterly fairness audit, and can suspend or decommission the model.
- **Escalation:** Clinicians report concerns through the clinical owner or the hospital's patient-safety reporting system (proposed). Incidents are triaged within 2 business days and reviewed by the governance committee at its next meeting; a safety-critical issue can trigger an immediate suspension by the clinical or technical owner.


## 8. Clinical workflow, success measures and compliance
- **Where the score appears:** computed when the discharge order is signed and shown on the transitional-care worklist with the tier, not as an interruptive alert.
- **Who acts:** care coordinators and nurses contact High-tier patients; discharge planners see the tier before discharge.
- **Intervention:** see the tier table. The score does not show that an intervention will help.
- **Success measures:** primary is 30-day readmission in the High tier versus a randomized or risk-adjusted comparison. Process measures are the share reached within 48 h and seen within 7 days. Equity is reach and sensitivity by group. Balancing measures are staff hours per flagged patient and the override rate.
- **Evaluation design:** pilot with randomized or stepped-wedge rollout. Flagged versus unflagged is confounded because flagged patients are higher-risk by construction.
- **Privacy and regulatory (confirm current requirements with compliance):** HIPAA (PHI, minimum necessary, access logging); whether the tool qualifies as non-device clinical decision support under FDA guidance; EHR decision-support transparency requirements; nondiscrimination rules for tools using race, sex or age as inputs; IRB or quality-improvement review for the pilot. The training data are a public de-identified research dataset.
"""

with open('model_card.md', 'w', encoding='utf-8') as f:
    f.write(model_card)
print("Saved: model_card.md")

print("Analysis 10 complete")

# %%
from IPython.display import Markdown, display
display(Markdown(open('model_card.md', encoding='utf-8').read()))

# %%
# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

print("\n" + "=" * 70)
print(" " * 25 + "PROJECT SUMMARY")
print("=" * 70)

# TODO: Fill in from your analyses and export 'project_summary.csv'
# --- Top driver ---------------------------------------------------------------
def top_driver(model, X, y, n_repeats=5):
    """Return (feature_name, method) for the most influential feature."""
    if model is None:
        return "N/A (placeholder data)", "none"

    est, names = model, list(X.columns)
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
        try:                                   # names after preprocessing / one-hot encoding
            names = list(model[:-1].get_feature_names_out())
        except Exception:
            pass

    imp = None
    if hasattr(est, 'feature_importances_'):
        imp, method = np.asarray(est.feature_importances_), "model feature importance"
    elif hasattr(est, 'coef_'):
        imp, method = np.abs(np.ravel(est.coef_)), "absolute coefficient"

    if imp is not None and len(imp) == len(names):
        return names[int(np.argmax(imp))], method

    # Fallback: model-agnostic permutation importance on the test set
    r = permutation_importance(model, X, y, scoring='roc_auc', n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return X.columns[int(np.argmax(r.importances_mean))], "permutation importance (test AUC)"

driver, driver_method = top_driver(best_model, X_test, y_test)

# --- Summary table ------------------------------------------------------------
summary = pd.DataFrame({
    'Metric': ['Cohort size', '30-day readmission rate', 'Features', 'Best model', 'Test ROC AUC',
               'Test PR AUC', 'Readmissions captured at chosen capacity', 'Top driver'],
    'Value': [
        f"{len(df):,}",
        f"{df['readmit_30'].mean():.3%}",
        f"{X_test.shape[1]}",
        model_name,
        f"{auc_all:.3f}",
        f"{pr_auc:.3f}",
        (f"{row15['pct_readmissions_captured']:.1%} "
         f"({int(row15['readmissions_captured'])} of {int(row15['readmissions_total'])}) "
         f"at top {row15['capacity_pct']:.0%} follow-up capacity; PPV {row15['ppv']:.1%}"),
        f"{driver} ({driver_method})",
    ]
})

summary.to_csv('project_summary.csv', index=False)
print(summary.to_string(index=False))

# %%
# ==============================================================================
# NEXT STEPS
# ==============================================================================

print("""
REMAINING TASKS:

1. COMPLETE ALL ANALYSES
   [✅] Analyses 1-6 are started: finish the TODOs and refine
   [✅] Complete Analyses 7-10

2. CREATE THE DASHBOARD
   [✅] Decide on format (single HTML with subplots, Jupyter notebook, Dash/Streamlit)
   [✅] Executive summary with key metrics at the top
   [✅] 10-12 interconnected visualizations
   [✅] Clinical insights highlighted

3. CREATE THE SLIDE DECK (10-15 slides, executive-ready, no live presentation)
   [✅] Title slide with team info
   [✅] Clinical problem and why readmission matters (1-2)
   [✅] Data and cohort (1)
   [✅] Methods (1)
   [✅] Key findings (5-7)
   [✅] Fairness and deployment plan (2-3)
   [✅] Limitations and next steps (1)

4. WRITE THE TECHNICAL REPORT (4-6 pages)
   [✅] Executive summary, data and cohort, methods, results, fairness, deployment plan,
       limitations, model card (appendix), AI Tool Use Disclosure

5. FINALIZE CODE
   [✅] Comments throughout; README.md; requirements.txt; test a clean run

6. QUALITY CHECK
   [✅] All files open; properly named; zip ready; team review complete
""")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")




