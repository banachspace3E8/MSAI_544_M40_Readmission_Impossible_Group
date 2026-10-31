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

# Same five training folds for all three models.
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

# Cross-validation uses training data only.
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

# Selection rule defined before inspecting test results:
# Among models within 0.005 of the highest mean CV ROC AUC,
# prefer the lowest CV Brier score, then shorter fitting time.
eligible = cv_results[
    cv_results["CV ROC AUC"]
    >= cv_results["CV ROC AUC"].max() - 0.005
]

best_model_name = eligible.sort_values(
    ["CV Brier", "CV fit seconds"]
).iloc[0]["Model"]

# Fit each model on the full training set.
# Evaluate on the existing held-out test set.
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

# Required variables for Analyses 8-10.
best_model = trained_models[best_model_name]
p_best = test_probabilities[best_model_name].copy()

model_results = cv_results.merge(
    pd.DataFrame(test_rows), on="Model"
)
model_results["Selected"] = (
    model_results["Model"] == best_model_name
)
model_results.to_csv("model_results.csv", index=False)

# ROC and calibration curves.
fig7 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Test ROC curves",
        "Test calibration curves"
    )
)

colors = ["#3498db", "#2ecc71", "#9b59b6"]

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
            line=dict(color="gray", dash="dash")
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
print("Analysis 7 complete; best_model and p_best are ready.")



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

# Explain the selected model and compare it with logistic regression.
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

# Partial dependence for prior inpatient visits.
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

# Chart: feature importance and partial dependence.
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
        marker_color="#9b59b6",
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
        marker_color="#3498db",
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
        line=dict(color="#2ecc71")
    ),
    row=1, col=2
)

fig8.add_vline(
    x=0,
    line_dash="dash",
    line_color="gray",
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

# Agreement between the models.
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

# Save aligned probabilities for Shishir's fairness analysis.
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
#   - For each group with >= 100 test patients compute: n, prevalence, ROC AUC, flag rate,
#     sensitivity, false positive rate, PPV. Table -> 'fairness_audit.csv'
#   - 1x3 figure: AUC by group | sensitivity by group | flag rate vs prevalence scatter
#     -> 'analysis9_fairness_audit.html'
#   - Discuss: which groups are under-flagged relative to their risk? Is that a model problem,
#     a data problem, or both? What would you do about it before deployment?
# Hint: pd.Series(p_best, index=X_test.index) lets you align probabilities with g_test



print("Analysis 9 complete"


# %%
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


# %%
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


# %%
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


# %%






