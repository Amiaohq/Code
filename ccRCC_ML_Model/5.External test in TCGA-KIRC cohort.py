import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils import resample

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration
# ==========================================
OUTPUT_DIR = r"..."
EXTERNAL_DATA_PATH = r"..."
EXT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "External_Validation")
os.makedirs(EXT_OUTPUT_DIR, exist_ok=True)

TIME_COL = 'DFS_time'
EVENT_COL = 'DFS_event'
TIME_UNIT_TO_MONTH = 1.0
EVAL_TIME_POINTS = [36, 60]
TIME_POINT_MONTHS = 36
N_BOOTSTRAP = 1000
FIG_DPI = 300

MODEL_FILES = {
    'GBS': 'model_gbs.pkl',
    'RSF': 'model_rsf.pkl',
    'SSVM': 'model_ssvm.pkl'
}
PLOT_ORDER = ['GBS', 'RSF', 'SSVM']
MODEL_COLORS = {'GBS': '#ff7f0e', 'RSF': '#1f77b4', 'SSVM': '#2ca02c'}
MODEL_MARKERS = {'GBS': 's', 'RSF': 'o', 'SSVM': '^'}

print("\n" + "=" * 60)
print("External validation of three ML models (GBS, RSF, SSVM)")
print("Metrics: time-dependent ROC | C-index | calibration | confusion matrix")
print("=" * 60)

# ==========================================
# 2. External cohort: load and standardize
# ==========================================
print("\n[Step 1] Loading external validation data...")
ext_df = pd.read_excel(EXTERNAL_DATA_PATH)
ext_df.columns = [str(c).strip() for c in ext_df.columns]

if TIME_UNIT_TO_MONTH != 1.0:
    ext_df[TIME_COL] = ext_df[TIME_COL] * TIME_UNIT_TO_MONTH

features_df = pd.read_csv(os.path.join(OUTPUT_DIR, "feature_final_for_modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

missing_features = [f for f in final_features if f not in ext_df.columns]
if missing_features:
    raise ValueError("Features missing in external data: " + str(missing_features))

print("[Step 2] Standardizing with the internal training scaler (transform only)...")
scaler = joblib.load(os.path.join(OUTPUT_DIR, "data_standard_scaler.pkl"))
scaled_cols = [str(c) for c in getattr(scaler, 'feature_names_in_', [])]
scaled_cols = [c for c in scaled_cols if c in ext_df.columns]

needed_cols = list(dict.fromkeys(scaled_cols + final_features + [TIME_COL, EVENT_COL]))
n_before = len(ext_df)
ext_df = ext_df.dropna(subset=needed_cols).reset_index(drop=True)
print("  Samples kept: " + str(len(ext_df)) + " (dropped " + str(n_before - len(ext_df)) + ")")

ext_df[scaled_cols] = scaler.transform(ext_df[scaled_cols])
ext_df.to_csv(os.path.join(EXT_OUTPUT_DIR, "data_ext_scaled.csv"), index=False)

X_ext = ext_df[final_features]
T_ext = ext_df[TIME_COL].values
E_ext = ext_df[EVENT_COL].values

print("[Step 3] Loading trained models and the internal training cohort...")
models = {}
for name in PLOT_ORDER:
    try:
        models[name] = joblib.load(os.path.join(OUTPUT_DIR, MODEL_FILES[name]))
    except Exception as e:
        models[name] = None

train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "data_train_scaled.csv"))
X_train = train_df[final_features]
T_train = train_df[TIME_COL].values
E_train = train_df[EVENT_COL].values

print("  External cohort: n = " + str(len(ext_df))
      + ", median DFS = " + str(round(float(np.median(T_ext)), 1)) + " months")
print("  Event rate: " + str(round(float(E_ext.mean()) * 100, 2)) + "%")


# ==========================================
# 3. Shared statistics
# ==========================================
def event_indicator(T, E, time_point):
    y = np.zeros(len(T))
    y[(T <= time_point) & (E == 1)] = 1
    return y


def valid_mask(T, E, time_point):
    return (T > time_point) | ((T <= time_point) & (E == 1))


def risk_scores(model, X):
    return np.asarray(model.predict(X), dtype=float).ravel()


def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=N_BOOTSTRAP):
    scores = []
    for i in range(n_bootstrap):
        idx = resample(np.arange(len(y_true)), random_state=i)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(roc_auc_score(y_true[idx], y_scores[idx]))
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan, np.nan
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def time_dependent_roc(model, X, T, E, time_point):
    scores = risk_scores(model, X)
    y_all = event_indicator(T, E, time_point)
    mask = valid_mask(T, E, time_point)
    y_true = y_all[mask]
    y_score = scores[mask]
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_score)
    return fpr, tpr, auc_mean, ci_lower, ci_upper


def cindex_with_bootstrap(model, X, T, E, n_bootstrap=N_BOOTSTRAP):
    risk = risk_scores(model, X)
    c_index = concordance_index(T, -risk, E)
    rng = np.random.RandomState(42)
    c_indices = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(T), len(T), replace=True)
        try:
            c_indices.append(concordance_index(T[idx], -risk[idx], E[idx]))
        except Exception:
            continue
    if not c_indices:
        return c_index, np.nan, np.nan, np.nan
    return c_index, np.std(c_indices), np.percentile(c_indices, 2.5), np.percentile(c_indices, 97.5)


def predict_survival(model, X, time_point):
    if hasattr(model, 'predict_survival_function'):
        try:
            try:
                arr = model.predict_survival_function(X, times=[time_point], return_array=True)
            except TypeError:
                arr = model.predict_survival_function(X, times=[time_point])
            arr = np.atleast_2d(np.asarray(arr, dtype=float))
            if arr.shape[0] != len(X):
                arr = arr.T
            if arr.shape[0] == len(X) and 0.002 < float(np.nanmean(arr)) < 0.998:
                return np.clip(arr[:, 0], 0.001, 0.999)
        except Exception:
            pass
    risk = risk_scores(model, X)
    spread = risk.max() - risk.min()
    norm_risk = (risk - risk.min()) / spread if spread > 0 else np.zeros_like(risk)
    return np.clip(1.0 - norm_risk, 0.001, 0.999)


# ==========================================
# 4. Time-dependent ROC (36 and 60 months)
# ==========================================
print("\n[Step 4] Time-dependent ROC with bootstrap 95% CI...")
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

roc_rows = []
for ax_idx, time_point in enumerate(EVAL_TIME_POINTS):
    ax = axes[ax_idx]
    ax.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='Random guess')

    for name in PLOT_ORDER:
        if models[name] is None:
            continue
        result = time_dependent_roc(models[name], X_ext, T_ext, E_ext, time_point)
        if result is None or np.isnan(result[2]):
            continue
        fpr, tpr, auc_mean, ci_lower, ci_upper = result
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=3,
                label=f"{name} (AUC = {auc_mean:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        roc_rows.append({'Model': name, 'Time_Point_Months': time_point,
                         'AUC': auc_mean, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper})

    ax.set_title(f'External Validation ({time_point} months)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel('1 - Specificity', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('Sensitivity', fontsize=22, fontweight='bold', labelpad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(labelsize=20, width=1.5, length=6)

plt.savefig(os.path.join(EXT_OUTPUT_DIR, "roc_ext_comparison_36m_60m.eps"),
            format='eps', dpi=FIG_DPI, bbox_inches='tight')
plt.close()
pd.DataFrame(roc_rows).to_csv(os.path.join(EXT_OUTPUT_DIR, "roc_ext_auc_results.csv"), index=False)

# ==========================================
# 5. C-index with bootstrap SE and 95% CI
# ==========================================
print("[Step 5] C-index with bootstrap SE and 95% CI...")
cindex_rows = []
for name in PLOT_ORDER:
    if models[name] is None:
        continue
    c_index, c_se, c_lower, c_upper = cindex_with_bootstrap(models[name], X_ext, T_ext, E_ext)
    cindex_rows.append({'Model': name, 'External_C_index': c_index, 'External_SE': c_se,
                        'External_CI_Lower': c_lower, 'External_CI_Upper': c_upper})
    print("  " + name + ": C-index = " + f"{c_index:.4f}" + " (95% CI: "
          + f"{c_lower:.4f}-{c_upper:.4f}" + ")")

cindex_df = pd.DataFrame(cindex_rows)
cindex_df.to_csv(os.path.join(EXT_OUTPUT_DIR, "cindex_ext_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(14, 10))
x = np.arange(len(cindex_df))
bar_colors = [MODEL_COLORS[m] for m in cindex_df['Model']]
errors = np.array([
    [(row['External_C_index'] - row['External_CI_Lower']),
     (row['External_CI_Upper'] - row['External_C_index'])]
    for _, row in cindex_df.iterrows()
]).T

bars = ax.bar(x, cindex_df['External_C_index'], 0.5, yerr=errors, capsize=5,
              color=bar_colors, edgecolor='black', linewidth=1.5)
for idx, bar in enumerate(bars):
    height = bar.get_height()
    if not np.isnan(height):
        ax.text(bar.get_x() + bar.get_width() / 2., height + errors[1][idx] + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(cindex_df['Model'])
ax.set_xlabel('Model', fontsize=22, fontweight='bold', labelpad=15)
ax.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax.set_title('C-index: External Validation (TCGA)', fontsize=24, fontweight='bold', pad=20)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0, 1.1])
ax.tick_params(labelsize=20, width=1.5, length=6)

plt.savefig(os.path.join(EXT_OUTPUT_DIR, "cindex_ext_comparison.eps"),
            format='eps', dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ==========================================
# 6. Calibration curves (36 and 60 months)
# ==========================================
print("[Step 6] Calibration curves...")


def plot_calibration(models, df, time_point, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Ideal Calibration Curve', alpha=0.6)

    for name in PLOT_ORDER:
        if models[name] is None:
            continue
        pred_survival = predict_survival(models[name], df[final_features], time_point)
        df_temp = df.copy()
        df_temp['pred_survival'] = pred_survival
        try:
            df_temp['group'] = pd.qcut(df_temp['pred_survival'], q=4, duplicates='drop')
        except Exception:
            continue
        if df_temp['group'].nunique() < 2:
            continue

        group_means = df_temp.groupby('group', observed=True)['pred_survival'].mean()
        observed_survivals = []
        kmf = KaplanMeierFitter()
        for _, group in df_temp.groupby('group', observed=True):
            kmf.fit(group[TIME_COL], event_observed=group[EVENT_COL])
            observed_survivals.append(np.interp(time_point, kmf.survival_function_.index,
                                                kmf.survival_function_.iloc[:, 0]))
        observed_survivals = pd.Series(observed_survivals, index=group_means.index)

        ax.scatter(group_means, observed_survivals, color=MODEL_COLORS[name],
                   marker=MODEL_MARKERS[name], s=180, label=name,
                   alpha=0.8, edgecolors='black', linewidth=2)
        sorted_idx = np.argsort(group_means.values)
        ax.plot(group_means.values[sorted_idx], observed_survivals.values[sorted_idx],
                color=MODEL_COLORS[name], alpha=0.5, linewidth=2.5)

    ax.set_xlabel('Model Predicted Survival Probability', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel('Actual Survival Probability', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title(f'External Validation ({time_point} Months)', fontsize=28, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=22, width=1.5, length=6)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


for time_point in EVAL_TIME_POINTS:
    plot_calibration(models, ext_df, time_point,
                     os.path.join(EXT_OUTPUT_DIR, f"calibration_ext_{time_point}m.eps"))

# ==========================================
# 7. Confusion matrix at 36 months (training-set Youden threshold)
# ==========================================
print("[Step 7] Confusion matrix at 36 months...")

y_train_36 = event_indicator(T_train, E_train, TIME_POINT_MONTHS)
mask_train = valid_mask(T_train, E_train, TIME_POINT_MONTHS)

train_thresholds = {}
for name in PLOT_ORDER:
    if models[name] is None:
        continue
    fpr, tpr, thresholds = roc_curve(y_train_36[mask_train],
                                     risk_scores(models[name], X_train)[mask_train])
    train_thresholds[name] = thresholds[np.argmax(tpr - fpr)]
    print("  " + name + ": training-set threshold = " + f"{train_thresholds[name]:.4f}")

y_ext_36 = event_indicator(T_ext, E_ext, TIME_POINT_MONTHS)
mask_ext = valid_mask(T_ext, E_ext, TIME_POINT_MONTHS)
y_valid = y_ext_36[mask_ext]

cm_results = {}
cm_rows = []
for name in PLOT_ORDER:
    if models[name] is None or name not in train_thresholds:
        continue
    y_pred = (risk_scores(models[name], X_ext)[mask_ext] >= train_thresholds[name]).astype(int)
    cm = confusion_matrix(y_valid, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / cm.sum()
    cm_results[name] = cm
    cm_rows.append({'Model': name, 'Threshold': train_thresholds[name],
                    'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                    'Accuracy': accuracy, 'Precision': precision,
                    'Recall': recall, 'Specificity': specificity, 'F1_Score': f1_score})
    print("  " + name + f": Acc={accuracy:.3f} Sens={recall:.3f} Spec={specificity:.3f} F1={f1_score:.3f}")

pd.DataFrame(cm_rows).to_csv(os.path.join(EXT_OUTPUT_DIR, "confusion_ext_results.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(wspace=0.35, bottom=0.15, top=0.88)
for idx, name in enumerate(PLOT_ORDER):
    ax = axes[idx]
    if name not in cm_results:
        ax.axis('off')
        continue
    cm = cm_results[name]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    labels = [[f"{cm[r, c] / row_sums[r, 0] * 100:.1f}%\n(n={cm[r, c]})" for c in range(2)]
              for r in range(2)]
    sns.heatmap(cm / row_sums * 100, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['No Recurrence', 'Recurrence'],
                ax=ax, cbar=False, annot_kws={'size': 16}, vmin=0, vmax=100)
    ax.set_xlabel('Predicted', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('Actual', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title(f'External Validation ({name})', fontsize=26, fontweight='bold', pad=20)
    ax.tick_params(labelsize=20, width=1.5, length=6)

plt.savefig(os.path.join(EXT_OUTPUT_DIR, "confusion_ext_36m.eps"),
            format='eps', dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ==========================================
# 8. Summary table
# ==========================================
print("[Step 8] Summary of external validation...")
roc_lookup = {(row['Model'], row['Time_Point_Months']): row for _, row in pd.DataFrame(roc_rows).iterrows()}
cm_lookup = {row['Model']: row for row in cm_rows}

summary_rows = []
for name in PLOT_ORDER:
    row = {'Model': name}
    c = cindex_df[cindex_df['Model'] == name]
    if not c.empty:
        r = c.iloc[0]
        row['C_index_95CI'] = f"{r['External_C_index']:.4f} ({r['External_CI_Lower']:.4f}-{r['External_CI_Upper']:.4f})"
    else:
        row['C_index_95CI'] = 'N/A'
    for time_point in EVAL_TIME_POINTS:
        key = (name, time_point)
        if key in roc_lookup:
            r = roc_lookup[key]
            row[f'AUC_{time_point}m_95CI'] = f"{r['AUC']:.3f} ({r['CI_Lower']:.3f}-{r['CI_Upper']:.3f})"
        else:
            row[f'AUC_{time_point}m_95CI'] = 'N/A'
    if name in cm_lookup:
        r = cm_lookup[name]
        row['CM_Accuracy'] = f"{r['Accuracy']:.3f}"
        row['CM_Sensitivity'] = f"{r['Recall']:.3f}"
        row['CM_Specificity'] = f"{r['Specificity']:.3f}"
        row['CM_F1'] = f"{r['F1_Score']:.3f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(EXT_OUTPUT_DIR, "model_ext_performance_summary.csv"), index=False)

print("\nExternal validation summary:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 60)
print("Done. Outputs saved to: " + EXT_OUTPUT_DIR)
print("=" * 60)