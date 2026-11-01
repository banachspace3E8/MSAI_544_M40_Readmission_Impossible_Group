# Model Card: 30-Day Hospital Readmission Risk Model

**Version:** 0.1 (draft) | **Date:** 2026-11-01 | **Model:** HistGradientBoostingClassifier | **Team:** Readmission Impossible

**Model details:** HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True; 61 iterations used), scikit-learn, random_state=42. Chosen on training-set 5-fold CV (AUC within 0.005 of the best, then Brier), not on the test set.

| Model | CV ROC AUC | Test ROC AUC | Test PR AUC | Test Brier |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.641 | 0.644 | 0.158 | 0.2316 |
| Random Forest | 0.651 | 0.651 | 0.162 | 0.1729 |
| Histogram Gradient Boosting | 0.646 | 0.648 | 0.162 | 0.0797 |

-Gain over the logistic baseline is +0.005 test AUC; the choice rests on calibration and speed, not discrimination.

## 1. Intended use
- **Purpose:** Rank inpatients with diabetes at discharge by risk of any readmission within 30 days (the label cannot separate planned from unplanned) ...
- **Intended users:** Transitional-care nurses, care coordinators and discharge planners. The score supports, and does not replace, clinical judgment.
- **Intended population:** Diabetes-related inpatient encounters. Excluded from the cohort: discharges to death, hospice or unmapped disposition. Transfers were kept. The data include 655 patients under 20; no performance claim is made for them and they are out of scope.
- **Out of scope:** Denying, delaying or rationing care; pricing, insurance or coverage decisions; use on pediatric, obstetric or non-diabetic populations; use as the sole basis for any clinical decision.

## 2. Training data
- **Source:** Diabetes 130-US hospitals dataset (diabetic_data.csv), encounters 1999-2008.
- **Cohort rules:** 101,766 raw encounters / 71,518 patients -> first encounter per patient (prevents patient-level leakage across the split) -> 69,195 after excluding death/hospice/unmapped discharges. Positive rate 9.0%.
- **Outcome:** `readmit_30` (readmitted within 30 days). Readmissions to other hospitals may be missed, which causes label noise.
- **Features:** 65 structured features (utilization, diagnosis group, medications, labs, discharge/admission codes, payer, specialty). Age, sex and race are included as inputs.
- **Split:** 80/20 stratified (55,356 train / 13,839 test), random_state=42; one row per patient. All numbers here are test-set.
- **Missing data (raw):** race 2.2% -> 'Unknown'; payer 39.6% and specialty 49.1% -> 'Missing'/'Unknown'; HbA1c 83% and glucose 95% -> 'not tested'; weight (97%) dropped.

## 3. Performance (held-out test set, n=13,839, prevalence 9.0%)
| Metric | Value |
|---|---|
| ROC AUC (95% bootstrap CI) | 0.648 (0.633-0.664) |
| PR AUC (average precision) | 0.162 |
| Brier score (base-rate reference 0.0819; skill 2.7%) | 0.0797 |
| Calibration slope / intercept | 1.06 / 0.14 |
| Mean predicted risk vs. observed prevalence | 0.090 vs 0.090 |

**At the operating point (follow up top 15% of discharges):** captures 29.8% of readmissions, PPV 17.9% (2.0x random), about 5.6 patients flagged per readmission captured.
Operating point: score >= 0.129 (about the top 15% on test). Deploy this fixed threshold, not a rolling percentile. Tier cutoffs: High >= 0.129, Medium >= 0.087.

| Risk tier | Predicted risk | Share of discharges | Observed readmission rate (95% CI) | Share of all readmissions | Proposed action |
|---|---|---|---|---|---|
| Low | 0.029-0.087 | 60% | 6.1% (5.6%-6.6%) | 40% | Standard discharge process |
| Medium | 0.087-0.129 | 25% | 10.7% (9.7%-11.8%) | 30% | Automated outreach or pharmacist med-reconciliation call (proposed) |
| High | 0.129-0.506 | 15% | 17.9% (16.3%-19.6%) | 30% | Care-team call within 48 h and visit within 7 days (proposed) |

Full capacity table: `capacity_analysis.csv`.

## 4. Fairness
- Audited at the top-15% operating point (score >= 0.129) across race, gender and age. Not evaluated (n < 100): race: Asian; gender: Unknown/Invalid; age: [0-10). † = fewer than 30 readmissions; unreliable.
- Among groups with >= 30 readmissions, sensitivity ranges 8.9% (age=[30-40)) to 39.6% (age=[80-90)); overall 29.8%. AUC ranges 0.602 to 0.704 over the same groups.
- Age is the main disparity: flag rate spans 5.7%-24.2% but prevalence only 7.4%-12.3%, so true risk differences do not explain it. Ages 30-50 are under-flagged and 80-90 over-flagged.
- Sex gap: sensitivity 33.1% (female) vs 25.9% (male).
- Age, sex and race are model inputs. Mitigations considered: (1) fixed-threshold monitoring by group; (2) refit without age/sex/race inputs: test AUC 0.647 (vs 0.648), overall sensitivity 29.9%, age-group sensitivity range 17.8%-36.7% (current model: 8.9%-39.6%); (3) age-band thresholds (flag the top 15% within each age band): overall sensitivity 30.7% (vs 29.8%), age-group range 22.2%-33.3%; age would enter the decision rule, so this needs ethics/legal review; (4) local validation with this same audit before go-live; (5) clinician override, and no use to deny services.

| Group | n | Readmissions | Flag rate | Sensitivity (95% CI) | PPV | AUC |
|---|---|---|---|---|---|---|
| race=AfricanAmerican | 2,507 | 231 | 14.0% | 27.3% (22%-33%) | 18.0% | 0.651 |
| race=Caucasian | 10,328 | 945 | 15.6% | 30.5% (28%-33%) | 17.9% | 0.644 |
| race=Hispanic † | 301 | 22 | 12.0% | 36.4% (20%-57%) | 22.2% | 0.744 |
| race=Other † | 219 | 7 | 11.4% | 14.3% (3%-51%) | 4.0% | 0.546 |
| race=Unknown | 391 | 33 | 8.2% | 21.2% (11%-38%) | 21.9% | 0.704 |
| gender=Female | 7,380 | 671 | 16.6% | 33.1% (30%-37%) | 18.1% | 0.661 |
| gender=Male | 6,458 | 575 | 13.1% | 25.9% (22%-30%) | 17.6% | 0.633 |
| age=[10-20) † | 103 | 9 | 3.9% | 22.2% (6%-55%) | 50.0% | 0.759 |
| age=[20-30) † | 210 | 18 | 4.8% | 16.7% (6%-39%) | 30.0% | 0.677 |
| age=[30-40) | 526 | 45 | 5.7% | 8.9% (4%-21%) | 13.3% | 0.678 |
| age=[40-50) | 1,344 | 105 | 9.2% | 20.0% (13%-29%) | 17.1% | 0.616 |
| age=[50-60) | 2,433 | 181 | 10.6% | 22.1% (17%-29%) | 15.6% | 0.668 |
| age=[60-70) | 3,139 | 268 | 12.9% | 28.7% (24%-34%) | 19.1% | 0.651 |
| age=[70-80) | 3,499 | 367 | 19.5% | 37.1% (32%-42%) | 19.9% | 0.654 |
| age=[80-90) | 2,185 | 207 | 24.2% | 39.6% (33%-46%) | 15.5% | 0.611 |
| age=[90-100) | 365 | 45 | 9.6% | 13.3% (6%-26%) | 17.1% | 0.602 |

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
- **Technical (model) owner:** ** Director of Clinical Analytics (proposed role). Responsible for the scoring pipeline, the fixed threshold (0.129) and tier cutoffs, the monitoring dashboard in Section 6, the monthly subgroup audit, retraining and revalidation decisions, versioned model and card documentation, and a rollback plan (switch the score off and revert to the standard discharge process).
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
