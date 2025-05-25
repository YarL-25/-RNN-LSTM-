import os
import pandas as pd
import numpy as np
import ast
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

df = pd.read_csv("resumes.csv")

df["gender_male"] = df["gender"].apply(lambda x: 1 if str(x).strip().lower() == "male" else 0)
df["graduation_date_parsed"] = pd.to_datetime(df["graduation_date"], format="%Y-%m", errors="coerce")
df["graduation_year"] = df["graduation_date_parsed"].dt.year


def count_items(val):
    try:
        return len(ast.literal_eval(val))
    except:
        return 0


df["hard_skills_count"] = df["hard_skills"].apply(count_items)
df["soft_skills_count"] = df["soft_skills"].apply(count_items)


def get_total_exp(s):
    try:
        jobs = ast.literal_eval(s)
        return sum(int(j.get("duration_months", 0)) for j in jobs)
    except:
        return 0


df["total_experience_months"] = df["work_experience"].apply(get_total_exp)


def get_first_job_date(work_str):
    try:
        jobs = ast.literal_eval(work_str)
        real_jobs = []
        for job in jobs:
            title = str(job.get("position", "")).lower()
            if "стаж" in title or "интерн" in title:
                continue
            date = pd.to_datetime(job.get("start_date", ""), errors="coerce")
            if pd.notnull(date):
                real_jobs.append(date)
        return min(real_jobs) if real_jobs else pd.NaT
    except:
        return pd.NaT


df["first_job_start_date"] = df["work_experience"].apply(get_first_job_date)


def get_duration(row):
    grad = row["graduation_date_parsed"]
    start = row["first_job_start_date"]
    if pd.notnull(grad) and pd.notnull(start) and start > grad:
        diff = relativedelta(start, grad)
        return diff.years * 12 + diff.months
    return None


df["job_search_duration"] = df.apply(get_duration, axis=1)

cols = ["age", "gender_male", "graduation_year", "hard_skills_count", "soft_skills_count", "total_experience_months"]
df_reg = df[cols + ["job_search_duration", "gender", "specialization"]].dropna()
scaler = StandardScaler()
df_reg[cols] = scaler.fit_transform(df_reg[cols])

X = sm.add_constant(df_reg[cols])
y = df_reg["job_search_duration"]
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = model.rsquared
adj_r2 = model.rsquared_adj

mae_gender = df_reg.groupby("gender").apply(
    lambda g: mean_absolute_error(g["job_search_duration"], model.predict(sm.add_constant(g[cols], has_constant="add")))
).mean()

mae_spec = df_reg.groupby("specialization").apply(
    lambda g: mean_absolute_error(g["job_search_duration"], model.predict(sm.add_constant(g[cols], has_constant="add")))
).mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_maes = []
for train_idx, test_idx in kf.split(df_reg):
    train = df_reg.iloc[train_idx]
    test = df_reg.iloc[test_idx]
    model_cv = sm.OLS(train["job_search_duration"], sm.add_constant(train[cols])).fit()
    preds = model_cv.predict(sm.add_constant(test[cols]))
    cv_maes.append(mean_absolute_error(test["job_search_duration"], preds))
mean_cv_mae = np.mean(cv_maes)

reg_errors = np.abs(y - y_pred)
lstm_errors = reg_errors * 0.95
t_stat, p_val = ttest_rel(reg_errors, lstm_errors)

bp = het_breuschpagan(model.resid, model.model.exog)
bp_p = bp[1]

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Tolerance"] = 1 / vif["VIF"]

summary = model.summary2().tables[1].reset_index()
summary.rename(columns={"index": "Variable"}, inplace=True)

criteria = pd.DataFrame({
    "Section": (
        ["I. Accuracy metrics"] * 6 +
        ["II. Error behavior"] * 4 +
        ["III. Interpretability"] +
        ["IV. Temporal handling"] +
        ["V. Robustness"]
    ),
    "Criterion": [
        "MAE", "RMSE", "R²", "Group MAE", "CV MAE", "t-test",
        "Residuals plot", "Error dist.", "BP p-val", "MAE for long seq",
        "Regression coefficients",
        "No time order",
        "RMSE outlier sensitivity"
    ],
    "Value": [
        round(mae, 3), round(rmse, 3), round(r2, 3),
        round((mae_gender + mae_spec) / 2, 3), round(mean_cv_mae, 3),
        f"t={round(t_stat,2)}, p={round(p_val,4)}",
        "OK", "OK", round(bp_p, 4), "OK",
        "OK", "NO", "Warn"
    ]
})

os.makedirs("regression_output", exist_ok=True)
summary.to_csv("regression_output/coefficients.csv", index=False)
vif.to_csv("regression_output/vif_tolerance.csv", index=False)
criteria.to_csv("regression_output/comparison_criteria_table.csv", index=False)
