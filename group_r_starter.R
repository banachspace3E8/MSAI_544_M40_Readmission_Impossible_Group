# ==============================================================================
# GROUP PROJECT: Hospital Readmission Risk - An End-to-End Clinical AI Solution
# Group Project - Starter Code (R)
# Saturday: 10 hours work time
#
# MSAI-544 Application of AI in Healthcare - Residency Group Project
#
# TEAM INFORMATION:
# Team Name: _______________________
# Members:   _______________________
#
# DELIVERABLES CHECKLIST:
# [ ] Interactive dashboard (HTML / R Markdown / flexdashboard)
# [ ] Slide deck (10-15 slides, .pptx)
# [ ] Technical report (4-6 pages) with model card
# [ ] Clean code with README
# [ ] All 10 analyses complete
#
# AI TOOL POLICY: AI tools may be used for debugging, ideating and report/slide clean-up only,
# never to write code. Disclose all use (with screenshots) in the technical report.
# ==============================================================================

# ==============================================================================
# SETUP & IMPORTS
# ==============================================================================

# install.packages(c("tidyverse", "caret", "glmnet", "ranger", "gbm", "pROC", "plotly", "htmlwidgets"))
library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(gbm)
library(pROC)
library(plotly)
library(htmlwidgets)

RANDOM_STATE <- 42
set.seed(RANDOM_STATE)
# Professional color palette: use consistently across every chart
BLUE <- '#3498db'; RED <- '#e74c3c'; GREEN <- '#2ecc71'; GRAY <- '#95a5a6'; PURPLE <- '#9b59b6'

cat(paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 12), collapse = ""), "HOSPITAL READMISSION RISK: END-TO-END CLINICAL AI\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Team: [YOUR TEAM NAME]\n")
cat("Start Time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# ==============================================================================
# DATA LOADING
# ==============================================================================

cat("\nLOADING DATA...\n")

# NOTE: both files are provided in the project folder. Update paths if needed.
# "?" is the missing-value marker; "None" in A1Cresult / max_glu_serum means the test was not done
df <- read_csv("diabetic_data.csv", na = c("?", "", "None"), show_col_types = FALSE, guess_max = 110000)
ids <- read_csv("ids_mapping.csv", show_col_types = FALSE, na = character())

cat("Encounters:", nrow(df), "rows x", ncol(df), "columns\n")
cat("Unique patients:", n_distinct(df$patient_nbr), "\n")
cat("\nColumn names:\n"); print(colnames(df))

# ==============================================================================
# DATA EXPLORATION
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 25), collapse = ""), "DATA EXPLORATION\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

cat("\nReadmitted (raw, three classes):\n"); print(table(df$readmitted))
cat("\nMissing fraction (columns with any missing):\n")
miss <- sort(colMeans(is.na(df)), decreasing = TRUE); print(round(miss[miss > 0], 3))
cat("\nEncounters per patient:\n"); print(summary(as.numeric(table(df$patient_nbr))))
cat("\nNumeric summary:\n")
print(summary(df[, c("time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                     "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses")]))
cat("\nID code descriptions (first rows):\n"); print(head(ids, 10))

# ==============================================================================
# ANALYSIS 1: DATA QUALITY & COHORT DEFINITION
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 1: DATA QUALITY & COHORT DEFINITION\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

n0 <- nrow(df)

# 1a. Drop unusable columns: weight is 97% missing; examide and citoglipton are never prescribed
df <- df %>% select(-weight, -examide, -citoglipton)
cat("Dropped weight, examide, citoglipton\n")

# 1b. TODO: Handle repeated patients
# Many patients have several encounters. Rows from the same patient are NOT independent:
# if one encounter lands in the training set and another in the test set, the model has
# effectively seen the test patient. Decision options:
#   Option A: keep only the FIRST encounter per patient (Strack et al., 2014)
#   Option B: keep all encounters but split train/test BY PATIENT (caret::groupKFold)
#   Option C: keep all encounters and ignore the issue (NOT acceptable; explain why in report)
# Hint for Option A: df %>% arrange(encounter_id) %>% distinct(patient_nbr, .keep_all = TRUE)

# YOUR DECISION HERE:



n1 <- nrow(df)

# 1c. TODO: Exclude encounters where readmission is impossible or undefined
# Look up discharge_disposition_id descriptions in `ids`. Patients who died or went to
# hospice cannot be readmitted. Decide what else, if anything, to exclude and justify it.
disch <- ids %>% filter(id_type == "discharge_disposition_id")
cat("\nDischarge dispositions:\n"); print(disch, n = 30)

# YOUR DECISION HERE:



n2 <- nrow(df)

# 1d. TODO: Handle remaining missing values (race, payer_code, medical_specialty, diag_1/2/3,
#     A1Cresult, max_glu_serum). Missing is often informative in clinical data ("not tested").
#     Decide: drop rows, fill with "Unknown"/"Missing", or treat as its own category.

# YOUR DECISION HERE:



n3 <- nrow(df)

# 1e. Define the target: readmitted within 30 days
df <- df %>% mutate(readmit_30 = as.integer(readmitted == "<30"))
cat(sprintf("\nCohort: %d rows; 30-day readmission rate = %.3f%%\n", n3, 100 * mean(df$readmit_30)))

# TODO: Build a cohort table (step, rows remaining) and save it as 'cohort_definition.csv'.
#       Plot it as a funnel chart (plot_ly(type = "funnel")) and save as 'analysis1_cohort_funnel.html'.
# Expected (Option A with expired/hospice excluded): roughly 101,766 -> 71,500 -> 70,000 rows, ~9% positive



cat("Analysis 1 complete\n")

# ==============================================================================
# ANALYSIS 2: DEMOGRAPHICS & READMISSION RATES
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 2: DEMOGRAPHICS & READMISSION RATES\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

df <- df %>% mutate(race = replace_na(race, "Unknown"))

# 30-day readmission rate and count by a grouping column
rate_table <- function(col) df %>% group_by(.data[[col]]) %>%
  summarise(rate = mean(readmit_30), n = n()) %>% rename(group = 1)

age_t <- rate_table("age"); race_t <- rate_table("race"); gender_t <- rate_table("gender")
cat("\nBy age:\n"); print(age_t); cat("\nBy race:\n"); print(race_t); cat("\nBy gender:\n"); print(gender_t)

overall <- mean(df$readmit_30)
mk_rate_bar <- function(t, color) {
  plot_ly(t, x = ~group, y = ~rate, type = "bar", text = ~n, textposition = "outside",
          marker = list(color = color), showlegend = FALSE) %>%
    layout(yaxis = list(tickformat = ".0%"),
           shapes = list(list(type = "line", x0 = -0.5, x1 = nrow(t) - 0.5, y0 = overall, y1 = overall,
                              line = list(dash = "dash", color = GRAY))))
}
fig2 <- subplot(mk_rate_bar(age_t, BLUE), mk_rate_bar(race_t, PURPLE), mk_rate_bar(gender_t, GREEN),
                margin = 0.05, titleY = TRUE) %>%
  layout(title = "Analysis 2: 30-Day Readmission Rate by Demographics (labels = n; dashed = overall)")
fig2
saveWidget(fig2, "analysis2_demographics.html")

# TODO: Add at least one more demographic view (e.g., age x gender heatmap, or payer type)



cat("Analysis 2 complete\n")

# ==============================================================================
# ANALYSIS 3: UTILIZATION & CLINICAL FACTORS
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 3: UTILIZATION & CLINICAL FACTORS\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

util_cols <- c("time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
               "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses")
cat("\nMean utilization by outcome:\n")
print(df %>% group_by(readmit_30) %>% summarise(across(all_of(util_cols), ~ round(mean(.x), 2))))

# Readmission rate by number of inpatient visits in the prior year (capped at 4+)
inp_t <- df %>% mutate(prior = if_else(number_inpatient >= 4, "4+", as.character(number_inpatient))) %>%
  group_by(prior) %>% summarise(rate = mean(readmit_30), n = n())
cat("\nBy prior inpatient visits:\n"); print(inp_t)

# TODO: Build a 1x3 figure (subplot):
#   Panel 1: bar of readmission rate by prior inpatient visits (inp_t, provided above)
#   Panel 2: line of readmission rate by length of stay (time_in_hospital, 1-14 days)
#   Panel 3: bar of readmission rate by HbA1c result (replace NA with "Not tested")
# Save as 'analysis3_utilization.html'
# Discussion point for the report: does testing HbA1c relate to readmission? (Strack et al. said yes)



cat("Analysis 3 complete\n")

# ==============================================================================
# ANALYSIS 4: DIAGNOSES & MEDICATIONS
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 4: DIAGNOSES & MEDICATIONS\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Group ICD-9 diagnosis codes into clinical categories following Strack et al. (2014)
icd9_group <- function(code) {
  if (is.na(code)) return("Missing")
  if (str_detect(code, "^[VE]")) return("Other")
  c <- suppressWarnings(as.numeric(code)); if (is.na(c)) return("Other")
  if ((c >= 390 & c <= 459) | c == 785) return("Circulatory")
  if ((c >= 460 & c <= 519) | c == 786) return("Respiratory")
  if ((c >= 520 & c <= 579) | c == 787) return("Digestive")
  if (floor(c) == 250) return("Diabetes")
  if (c >= 800 & c <= 999) return("Injury")
  if (c >= 710 & c <= 739) return("Musculoskeletal")
  if ((c >= 580 & c <= 629) | c == 788) return("Genitourinary")
  if (c >= 140 & c <= 239) return("Neoplasms")
  "Other"
}
df <- df %>% mutate(diag_group = map_chr(diag_1, icd9_group))
cat("\nPrimary diagnosis groups:\n"); print(table(df$diag_group))

MED_COLS <- c("metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
              "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
              "troglitazone", "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin",
              "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone")
# Each medication column is No / Steady / Up / Down
df <- df %>% mutate(n_meds_changed = rowSums(across(all_of(MED_COLS), ~ .x %in% c("Up", "Down"))),
                    n_diabetes_meds = rowSums(across(all_of(MED_COLS), ~ .x != "No")))

# TODO: Build a 1x3 figure:
#   Panel 1: readmission rate by primary diagnosis group (horizontal bars, sorted)
#   Panel 2: readmission rate by insulin order (No / Steady / Up / Down)
#   Panel 3: readmission rate by number of diabetes medications (0, 1, 2, 3+)
# Save as 'analysis4_diagnoses_medications.html'
# Optional: which individual medications are most associated with readmission?



cat("Analysis 4 complete\n")

# ==============================================================================
# ANALYSIS 5: FEATURE ENGINEERING
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 5: FEATURE ENGINEERING\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Provided helpers: look-ups for the three coded ID columns
adm_type <- ids %>% filter(id_type == "admission_type_id") %>% select(id, description) %>% deframe()
adm_src <- ids %>% filter(id_type == "admission_source_id") %>% select(id, description) %>% deframe()
disch_map <- disch %>% select(id, description) %>% deframe()

# Collapse 30 discharge codes into 4 clinically meaningful groups
group_discharge <- function(i) {
  d <- disch_map[as.character(i)]; if (is.na(d)) return("Other/Unknown")
  if (str_starts(d, "Discharged to home") & !str_detect(d, "health")) return("Home")
  if (str_detect(str_to_lower(d), "home health") | str_detect(d, "Home IV")) return("Home with services")
  if (str_detect(d, "SNF|ICF|inpatient care|hospital") | str_detect(str_to_lower(d), "rehab|nursing")) return("Facility transfer")
  "Other/Unknown"
}

# TODO: Build a modeling data frame `model_df` and then the feature matrix X (numeric matrix) and y.
# Requirements:
#   - Numeric: age midpoint (as.numeric(str_extract(age, "\\d+")) + 5), all util_cols, n_meds_changed,
#     n_diabetes_meds, total prior visits (sum of the three)
#   - Binary flags: A1C tested (!is.na(A1Cresult)), A1C high (>7 or >8), glucose tested, insulin ordered,
#     medication changed, any diabetes medication, female
#   - Categorical, one-hot encoded with caret::dummyVars: race, admission type (map with adm_type,
#     collapse Not Available/NULL/Not Mapped into Unknown), discharge group (group_discharge),
#     admission source (Emergency room / Referral / Transfer / Other), diag_group,
#     payer (top codes + Other + Missing), medical_specialty (top 8 + Other + Missing)
#   - Do NOT include: encounter_id, patient_nbr, readmitted, raw diag_2/diag_3 codes (900+ values)
#   - Keep race, gender, age in model_df (NOT in X) for the fairness audit
# Save 'feature_dictionary.csv' (feature, type, description) and 'cohort_features.csv' (X + y + race/gender/age)
# Chart: top 10 positive and 10 negative correlations with readmit_30 -> 'analysis5_feature_correlations.html'
# Expected: roughly 55-70 features



cat("Analysis 5 complete\n")

# ==============================================================================
# ANALYSIS 6: BASELINE MODEL
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 6: BASELINE MODEL (LOGISTIC REGRESSION)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# This block assumes you created model_df, X (data frame of numeric features) and y in Analysis 5.
if (!exists("X") | !exists("y") | !exists("model_df"))
  stop("Analysis 6 needs model_df, X and y from Analysis 5. Complete Analysis 5, then re-run.")

set.seed(RANDOM_STATE)
train_idx <- createDataPartition(factor(y), p = 0.8, list = FALSE)[, 1]
X_train <- as.matrix(X[train_idx, ]); X_test <- as.matrix(X[-train_idx, ])
y_train <- y[train_idx]; y_test <- y[-train_idx]
g_test <- model_df[-train_idx, c("race", "gender", "age")]   # for the fairness audit
cat("Train", length(y_train), "/ Test", length(y_test), "; positive rate", round(mean(y_train), 3), "\n")

# Balanced observation weights because only ~9% of patients are readmitted within 30 days
w <- if_else(y_train == 1, 0.5 / mean(y_train), 0.5 / (1 - mean(y_train)))
scaler <- preProcess(X_train, method = c("center", "scale"))
Xs_train <- predict(scaler, X_train); Xs_test <- predict(scaler, X_test)
lr_fit <- glmnet(Xs_train, y_train, family = "binomial", alpha = 0, lambda = 0.01, weights = w)
p_lr <- as.numeric(predict(lr_fit, Xs_test, type = "response"))

# Provided: standard metrics for an imbalanced clinical classifier (PR AUC = average precision)
summarize <- function(name, p, thr = 0.5) {
  pred <- as.integer(p >= thr)
  o <- order(-p); yt <- y_test[o]; prec <- cumsum(yt) / seq_along(yt)
  tibble(Model = name,
         `ROC AUC` = round(as.numeric(auc(y_test, p, quiet = TRUE)), 3),
         `PR AUC` = round(sum(prec * yt) / sum(yt), 3),
         Brier = round(mean((p - y_test)^2), 4),
         `Sensitivity@0.5` = round(sum(pred == 1 & y_test == 1) / sum(y_test == 1), 3),
         `Precision@0.5` = round(sum(pred == 1 & y_test == 1) / max(sum(pred == 1), 1), 3))
}
res_lr <- summarize("Logistic Regression (balanced)", p_lr); print(as.data.frame(res_lr))
# Expected: ROC AUC about 0.62-0.66. Readmission is a HARD prediction problem; published
# models on this dataset land in the same range. Do not "fix" this by leaking information.

roc_lr <- roc(y_test, p_lr, quiet = TRUE)
o <- order(-p_lr); yt <- y_test[o]; rec <- cumsum(yt) / sum(yt); prec <- cumsum(yt) / seq_along(yt)
fig6 <- subplot(
  plot_ly() %>%
    add_trace(x = 1 - roc_lr$specificities, y = roc_lr$sensitivities, type = "scatter", mode = "lines",
              name = paste0("LR (AUC ", res_lr$`ROC AUC`, ")"), line = list(color = BLUE)) %>%
    add_trace(x = c(0, 1), y = c(0, 1), type = "scatter", mode = "lines", line = list(dash = "dash", color = GRAY), showlegend = FALSE) %>%
    layout(xaxis = list(title = "False positive rate"), yaxis = list(title = "True positive rate")),
  plot_ly() %>%
    add_trace(x = rec, y = prec, type = "scatter", mode = "lines", name = paste0("LR (PR AUC ", res_lr$`PR AUC`, ")"), line = list(color = RED)) %>%
    add_trace(x = c(0, 1), y = rep(mean(y_test), 2), type = "scatter", mode = "lines", line = list(dash = "dash", color = GRAY), showlegend = FALSE) %>%
    layout(xaxis = list(title = "Recall"), yaxis = list(title = "Precision")),
  margin = 0.06, titleX = TRUE, titleY = TRUE) %>%
  layout(title = "Analysis 6: Baseline Logistic Regression (dashed = no-skill)")
fig6
saveWidget(fig6, "analysis6_baseline_model.html")

# TODO: Print the 10 largest positive and 10 largest negative coefficients (coef(lr_fit)) with feature names



cat("Analysis 6 complete\n")

# ==============================================================================
# ANALYSIS 7: ADVANCED MODELS & COMPARISON
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 7: ADVANCED MODELS & COMPARISON\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# TODO: Train at least TWO additional models and compare all of them:
#   ranger(x = X_train, y = factor(y_train), num.trees = 300, min.node.size = 20, probability = TRUE,
#          class.weights = c(1, (1 - mean(y_train)) / mean(y_train)))
#   gbm.fit(x = as.data.frame(X_train), y = y_train, distribution = "bernoulli", n.trees = 300,
#           interaction.depth = 4, shrinkage = 0.05, n.minobsinnode = 20)
#   (optional) tune one hyperparameter with a small grid; document what you tried
# Requirements:
#   - 5-fold CV ROC AUC on the TRAINING set for each model (createFolds on factor(y_train))
#   - Test-set ROC AUC, PR AUC, Brier via summarize()
#   - Table -> 'model_results.csv'
#   - 1x2 figure: ROC curves for all models | calibration curves (bin predictions into deciles,
#     plot mean predicted vs observed rate) -> 'analysis7_model_comparison.html'
#   - Choose a final model and justify it (discrimination, calibration, interpretability, speed)
#   - Keep its test probabilities in p_best and a prediction function best_predict(Xm)
# Hint: write one fit_predict(model_name, Xtr, ytr, Xte) function and loop over folds



cat("Analysis 7 complete\n")

# ==============================================================================
# ANALYSIS 8: EXPLAINABILITY
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 8: EXPLAINABILITY\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# TODO: Explain the final model
#   - Permutation importance on the TEST set: for each feature, shuffle its column, recompute AUC,
#     record the drop (repeat 3 times and average). Table -> 'feature_importance.csv'
#   - Partial dependence for at least one key feature (e.g., number_inpatient): set the column to
#     each value 0..7 on a sample of training rows and average the predicted probability
#   - 1x2 figure: top 15 features with error bars | partial dependence curve -> 'analysis8_explainability.html'
#   - Compare with the logistic regression coefficients from Analysis 6: do the models agree?
#   - Clinical sense check: is each top driver plausible, or is it a proxy for something else
#     (e.g., discharge to a facility = sicker patient)?



cat("Analysis 8 complete\n")

# ==============================================================================
# ANALYSIS 9: FAIRNESS & BIAS AUDIT
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 9: FAIRNESS & BIAS AUDIT\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# TODO: Audit the final model across race, gender and age groups (use g_test)
#   - Pick an operating point: e.g., flag the top 15% highest-risk patients
#     thr <- quantile(p_best, 0.85)
#   - For each group with >= 100 test patients compute: n, prevalence, ROC AUC, flag rate,
#     sensitivity, false positive rate, PPV. Table -> 'fairness_audit.csv'
#   - 1x3 figure: AUC by group | sensitivity by group | flag rate vs prevalence scatter
#     -> 'analysis9_fairness_audit.html'
#   - Discuss: which groups are under-flagged relative to their risk? Is that a model problem,
#     a data problem, or both? What would you do about it before deployment?



cat("Analysis 9 complete\n")

# ==============================================================================
# ANALYSIS 10: DEPLOYMENT & CLINICAL INTEGRATION
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 15), collapse = ""), "ANALYSIS 10: DEPLOYMENT & CLINICAL INTEGRATION\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

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



cat("Analysis 10 complete\n")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(paste(rep(" ", 25), collapse = ""), "PROJECT SUMMARY\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# TODO: Fill in from your analyses and export 'project_summary.csv'
summary_stats <- tibble(
  Metric = c("Cohort size", "30-day readmission rate", "Features", "Best model", "Test ROC AUC",
             "Test PR AUC", "Readmissions captured at chosen capacity", "Top driver"),
  Value = c(nrow(df), sprintf("%.1f%%", 100 * mean(df$readmit_30)), NA, NA, NA, NA, NA, NA))
write_csv(summary_stats, "project_summary.csv")
print(as.data.frame(summary_stats))

# ==============================================================================
# NEXT STEPS
# ==============================================================================

cat("
REMAINING TASKS:

1. COMPLETE ALL ANALYSES
   [ ] Analyses 1-6 are started: finish the TODOs and refine
   [ ] Complete Analyses 7-10

2. CREATE THE DASHBOARD
   [ ] Decide on format (single HTML with subplots, R Markdown, flexdashboard, Shiny)
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
   [ ] Comments throughout; README.md; package list; test a clean run

6. QUALITY CHECK
   [ ] All files open; properly named; zip ready; team review complete
")
cat("End Time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
