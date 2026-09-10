import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

output_dir = "..."
time_col = 'DFS_time'
event_col = 'DFS_event'
time_points = [36, 60]
N_BOOTSTRAP = 1000
model_list = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
except Exception as e:
    gbs_model = None


def get_model_scores(df, model_name):
    scores = None
    if model_name == 'GBS':
        if gbs_model is None:
            return None
        risk_scores = gbs_model.predict(df[final_features])
        min_risk = risk_scores.min()
        max_risk = risk_scores.max()
        if max_risk > min_risk:
            norm_risk = (risk_scores - min_risk) / (max_risk - min_risk)
        else:
            norm_risk = np.zeros_like(risk_scores)
        scores = norm_risk
    else:
        if model_name in df.columns:
            scores = df[model_name].values
        else:
            return None
    return scores


def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=1000):
    bootstrapped_scores = []
    valid_indices = np.where(~np.isnan(y_true) & ~np.isnan(y_scores))[0]
    if len(valid_indices) == 0:
        return np.nan, np.nan, np.nan
    y_true_valid = y_true[valid_indices]
    y_scores_valid = y_scores[valid_indices]
    if len(np.unique(y_true_valid)) < 2:
        return np.nan, np.nan, np.nan
    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true_valid)), random_state=i)
        if len(np.unique(y_true_valid[indices])) < 2:
            continue
        try:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_valid[indices], y_scores_valid[indices])
            score = auc(fpr_boot, tpr_boot)
            bootstrapped_scores.append(score)
        except:
            continue
    if not bootstrapped_scores:
        return np.nan, np.nan, np.nan
    mean_auc = np.mean(bootstrapped_scores)
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)
    return mean_auc, ci_lower, ci_upper


def calculate_time_roc_with_ci(df, time_point, model_name):
    scores = get_model_scores(df, model_name)
    if scores is None:
        return None, None, np.nan, np.nan, np.nan
    T = df[time_col].values
    E = df[event_col].values
    y_true = np.zeros(len(T))
    y_true[(T <= time_point) & (E == 1)] = 1
    mask = (T > time_point) | ((T <= time_point) & (E == 1))
    if np.sum(mask) == 0:
        return None, None, np.nan, np.nan, np.nan
    y_true_valid = y_true[mask]
    scores_valid = scores[mask]
    fpr, tpr, thresholds = roc_curve(y_true_valid, scores_valid)
    roc_auc = auc(fpr, tpr)
    auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(
        y_true_valid, scores_valid, N_BOOTSTRAP
    )
    return fpr, tpr, auc_mean, ci_lower, ci_upper


def plot_comparison_roc_by_timepoint(train_df, test_df, time_point, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

    colors = {
        'GBS': '#ff7f0e',
        'Leibovich': '#d62728',
        'UISS': '#17becf',
        'KEYNOTE-564': '#9467bd'
    }

    linestyles = {
        'GBS': '-',
        'Leibovich': '--',
        'UISS': '-.',
        'KEYNOTE-564': ':'
    }

    linewidths = {
        'GBS': 3.0,
        'Leibovich': 2.0,
        'UISS': 2.0,
        'KEYNOTE-564': 2.0
    }

    datasets = [
        ('Training Set', train_df),
        ('Testing Set', test_df)
    ]

    for idx, (dataset_name, df) in enumerate(datasets):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)

        for model in model_list:
            fpr, tpr, auc_mean, ci_lower, ci_upper = calculate_time_roc_with_ci(df, time_point, model)
            if fpr is not None:
                label = f"{model} (AUC = {auc_mean:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f})"
                ax.plot(fpr, tpr, color=colors[model], lw=linewidths[model],
                        linestyle=linestyles[model], label=label)

        ax.set_xlabel('1 - Specificity', fontsize=22, fontweight='bold', labelpad=15)
        ax.set_ylabel('Sensitivity', fontsize=22, fontweight='bold', labelpad=15)
        ax.set_title(f'{dataset_name} ({time_point} months)', fontsize=26, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
    plt.close()


for time_point in time_points:
    save_path = os.path.join(output_dir, f"6_GBS_vs_Clinical_ROC_{time_point}m.eps")
    plot_comparison_roc_by_timepoint(train_df, test_df, time_point, save_path)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import warnings
from lifelines.utils import concordance_index
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

output_dir = "..."
time_col = 'DFS_time'
event_col = 'DFS_event'
model_list = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

colors = {
    'GBS': '#ff7f0e',
    'Leibovich': '#d62728',
    'UISS': '#17becf',
    'KEYNOTE-564': '#9467bd'
}

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
except Exception as e:
    gbs_model = None


def get_model_scores(df, model_name):
    scores = None
    if model_name == 'GBS':
        if gbs_model is None:
            return None
        risk_scores = gbs_model.predict(df[final_features])
        scores = risk_scores
    else:
        if model_name in df.columns:
            scores = df[model_name].values
        else:
            return None
    return scores


def calculate_c_index(df, model_name):
    scores = get_model_scores(df, model_name)
    if scores is None:
        return None, None
    T = df[time_col].values
    E = df[event_col].values
    c_index = concordance_index(T, -scores, E)
    n_bootstrap = 1000
    c_indices = []
    n_samples = len(T)
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        T_boot = T[indices]
        E_boot = E[indices]
        scores_boot = scores[indices]
        try:
            c_boot = concordance_index(T_boot, -scores_boot, E_boot)
            c_indices.append(c_boot)
        except:
            continue
    if len(c_indices) > 0:
        se = np.std(c_indices)
    else:
        se = 0.0
    return c_index, se


results = []
for model in model_list:
    train_c, train_se = calculate_c_index(train_df, model)
    if train_c is None:
        continue
    test_c, test_se = calculate_c_index(test_df, model)
    results.append({
        'Model': model,
        'Train_C_index': train_c,
        'Train_SE': train_se,
        'Test_C_index': test_c,
        'Test_SE': test_se,
        'Gap': train_c - test_c
    })

summary_df = pd.DataFrame(results)

summary_path = os.path.join(output_dir, "7_GBS_vs_Clinical_C_Index.csv")
summary_df.to_csv(summary_path, index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

order = model_list
summary_df_ordered = summary_df.set_index('Model').loc[order].reset_index()
x = np.arange(len(summary_df_ordered))
width = 0.35

train_errors = [row['Train_SE'] for _, row in summary_df_ordered.iterrows()]
train_colors = [colors[row['Model']] for _, row in summary_df_ordered.iterrows()]

bars1 = ax1.bar(x - width / 2, summary_df_ordered['Train_C_index'], width,
                yerr=train_errors, capsize=5, color=train_colors,
                edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Model', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_title('Training Set', fontsize=26, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df_ordered['Model'], rotation=45, ha='right')
ax1.legend(fontsize=10)
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([0, 1.1])
ax1.tick_params(labelsize=20)

for bar, err in zip(bars1, train_errors):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + err + 0.03,
             f'{height:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')

test_errors = [row['Test_SE'] for _, row in summary_df_ordered.iterrows()]
test_colors = [colors[row['Model']] for _, row in summary_df_ordered.iterrows()]

bars2 = ax2.bar(x + width / 2, summary_df_ordered['Test_C_index'], width,
                yerr=test_errors, capsize=5, color=test_colors,
                edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Model', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_title('Testing Set', fontsize=26, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(summary_df_ordered['Model'], rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim([0, 1.1])
ax2.tick_params(labelsize=20)

for bar, err in zip(bars2, test_errors):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + err + 0.03,
             f'{height:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(output_dir, "7_GBS_vs_Clinical_C_Index.eps")
plt.savefig(plot_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

TIME_POINT_MONTHS = 36
model_list = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

colors = {
    'GBS': '#ff7f0e',
    'Leibovich': '#d62728',
    'UISS': '#17becf',
    'KEYNOTE-564': '#9467bd'
}

output_dir = "..."

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

time_col = 'DFS_time'
event_col = 'DFS_event'

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

X_train = train_df[final_features]
X_test = test_df[final_features]

T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values

gbs_model = None
try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
except Exception as e:
    pass


def create_time_dependent_labels(T, E, time_point):
    y_true = np.zeros(len(T))
    y_true[(T <= time_point) & (E == 1)] = 1
    return y_true


def get_risk_scores(model_name, df, X):
    if model_name == 'GBS':
        if gbs_model is None:
            return np.zeros(len(X))
        return gbs_model.predict(X)
    else:
        if model_name in df.columns:
            return df[model_name].values
        else:
            return np.zeros(len(df))


def calculate_roc_and_youden(model_name, df, X, y_true, T, E, time_point):
    scores = get_risk_scores(model_name, df, X)
    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    scores_valid = scores[valid_mask]
    if len(np.unique(y_true_valid)) < 2:
        return None, None, None, None, None, None
    fpr, tpr, thresholds = roc_curve(y_true_valid, scores_valid)
    youden_index = tpr - fpr
    max_index = np.argmax(youden_index)
    optimal_threshold = thresholds[max_index]
    max_youden = youden_index[max_index]
    return fpr, tpr, thresholds, youden_index, optimal_threshold, max_youden


def calculate_confusion_matrix_with_threshold(model_name, df, X, y_true, T, E, threshold, time_point):
    scores = get_risk_scores(model_name, df, X)

    if model_name in ['Leibovich', 'UISS', 'KEYNOTE-564']:
        y_pred = (scores >= 1.5).astype(int)
    else:
        y_pred = (scores >= threshold).astype(int)

    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    cm = confusion_matrix(y_true_valid, y_pred_valid)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'threshold': threshold,
        'n_valid': len(y_true_valid)
    }


def format_cm_label(value, total):
    percentage = value / total * 100
    return f"{percentage:.1f}%\n(n={value})"


def cm_to_percentage_matrix(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    percentages = cm / row_sums * 100
    return percentages


y_train = create_time_dependent_labels(T_train, E_train, TIME_POINT_MONTHS)
y_test = create_time_dependent_labels(T_test, E_test, TIME_POINT_MONTHS)

all_results = {}

for model_name in model_list:
    fpr_train, tpr_train, thresholds_train, youden_train, optimal_threshold_train, max_youden_train = calculate_roc_and_youden(
        model_name, train_df, X_train, y_train, T_train, E_train, TIME_POINT_MONTHS
    )

    if optimal_threshold_train is None:
        continue

    metrics_train = calculate_confusion_matrix_with_threshold(
        model_name, train_df, X_train, y_train, T_train, E_train, optimal_threshold_train, TIME_POINT_MONTHS
    )

    metrics_test = calculate_confusion_matrix_with_threshold(
        model_name, test_df, X_test, y_test, T_test, E_test, optimal_threshold_train, TIME_POINT_MONTHS
    )

    all_results[model_name] = {
        'train': metrics_train,
        'test': metrics_test,
        'optimal_threshold': optimal_threshold_train,
        'youden_index': max_youden_train,
        'fpr_train': fpr_train,
        'tpr_train': tpr_train
    }

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.35, bottom=0.2, top=0.93)

axes_flat = axes.flatten()

for i, model_name in enumerate(model_list):
    if model_name not in all_results:
        continue

    ax = axes_flat[i]
    res = all_results[model_name]

    ax.plot(res['fpr_train'], res['tpr_train'], color=colors[model_name], lw=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')

    idx = np.argmax(res['tpr_train'] - res['fpr_train'])
    ax.plot(res['fpr_train'][idx], res['tpr_train'][idx], 'ro', markersize=8,
            label=f'Best Threshold ({res["optimal_threshold"]:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('True Positive Rate', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title(f'{model_name}\nYouden Index: {res["youden_index"]:.3f}', fontsize=26, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(labelsize=20)

for i in range(len(model_list), len(axes_flat)):
    axes_flat[i].axis('off')

plot_path = os.path.join(output_dir, "11_GBS_vs_Clinical_ROC_Youden_Exclude_Censored.eps")
plt.savefig(plot_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(4, 2, figsize=(16, 32))
plt.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.15, top=0.94)

for i, model_name in enumerate(model_list):
    if model_name not in all_results:
        continue

    cm_train = all_results[model_name]['train']['confusion_matrix']
    cm_train_pct = cm_to_percentage_matrix(cm_train)

    ax_train = axes[i, 0]

    labels_train = []
    for row in range(cm_train.shape[0]):
        row_labels = []
        for col in range(cm_train.shape[1]):
            value = cm_train[row, col]
            total = np.sum(cm_train[row, :])
            label = format_cm_label(value, total)
            row_labels.append(label)
        labels_train.append(row_labels)

    sns.heatmap(cm_train_pct, annot=labels_train, fmt='', cmap='Blues',
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['No Recurrence', 'Recurrence'],
                ax=ax_train, cbar=False, annot_kws={'size': 16},
                vmin=0, vmax=100)

    ax_train.set_xlabel('Predicted', fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_ylabel('Actual', fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_title(f'Training Set ({model_name})', fontsize=26, fontweight='bold', pad=20)
    ax_train.tick_params(labelsize=20)

    cm_test = all_results[model_name]['test']['confusion_matrix']
    cm_test_pct = cm_to_percentage_matrix(cm_test)

    ax_test = axes[i, 1]

    labels_test = []
    for row in range(cm_test.shape[0]):
        row_labels = []
        for col in range(cm_test.shape[1]):
            value = cm_test[row, col]
            total = np.sum(cm_test[row, :])
            label = format_cm_label(value, total)
            row_labels.append(label)
        labels_test.append(row_labels)

    sns.heatmap(cm_test_pct, annot=labels_test, fmt='', cmap='Blues',
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['No Recurrence', 'Recurrence'],
                ax=ax_test, cbar=False, annot_kws={'size': 16},
                vmin=0, vmax=100)

    ax_test.set_xlabel('Predicted', fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_ylabel('Actual', fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_title(f'Testing Set ({model_name})', fontsize=26, fontweight='bold', pad=20)
    ax_test.tick_params(labelsize=20)

confusion_matrix_path = os.path.join(output_dir, "11_GBS_vs_Clinical_Confusion_Matrix_Exclude_Censored.eps")
plt.savefig(confusion_matrix_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import joblib
import warnings

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

output_dir = "..."

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

time_col = 'DFS_time'
event_col = 'DFS_event'

model_names = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

colors = {
    'GBS': '#ff7f0e',
    'Leibovich': '#d62728',
    'UISS': '#17becf',
    'KEYNOTE-564': '#9467bd'
}

linestyles = {
    'GBS': '-',
    'Leibovich': '--',
    'UISS': '-.',
    'KEYNOTE-564': ':'
}

markers = {
    'GBS': 's',
    'Leibovich': 'o',
    'UISS': '^',
    'KEYNOTE-564': 'D'
}

gbs_model = None
try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
except Exception as e:
    pass


def predict_survival_probability(model_name, df, time_point):
    try:
        if model_name == 'GBS':
            if gbs_model is None:
                return None
            risk_scores = gbs_model.predict(df[final_features])
            min_risk = risk_scores.min()
            max_risk = risk_scores.max()
            if max_risk > min_risk:
                normalized_risk = (risk_scores - min_risk) / (max_risk - min_risk)
            else:
                normalized_risk = np.zeros_like(risk_scores)
            pred_survival = 1 - normalized_risk
            return pred_survival
        else:
            if model_name in df.columns:
                risk_scores = df[model_name].values
                risk_map = {1: 0.8, 2: 0.5, 3: 0.2}
                pred_survival = np.array([risk_map.get(r, 0.5) for r in risk_scores])
                return pred_survival
            else:
                return None
    except Exception as e:
        return None


def plot_calibration_comparison(df, features, time_col, event_col, time_point, title, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Ideal Calibration Curve', alpha=0.6)

    for model_name in model_names:
        pred_survival = predict_survival_probability(model_name, df, time_point)

        if pred_survival is None:
            continue

        pred_survival = np.clip(pred_survival, 0.001, 0.999)

        df_temp = df.copy()
        df_temp['pred_survival'] = pred_survival

        if model_name == 'GBS':
            try:
                df_temp['group'] = pd.qcut(df_temp['pred_survival'], q=4, duplicates='drop')
            except Exception as e:
                continue
        else:
            df_temp['group'] = df[model_name].astype(str)

        if len(df_temp['group'].unique()) < 2:
            continue

        group_means = df_temp.groupby('group')['pred_survival'].mean()
        observed_survivals = []

        kmf = KaplanMeierFitter()
        for name, group in df_temp.groupby('group'):
            kmf.fit(group[time_col], event_observed=group[event_col])
            if time_point > kmf.survival_function_.index[-1]:
                obs = kmf.survival_function_.iloc[-1, 0]
            elif time_point < kmf.survival_function_.index[0]:
                obs = 1.0
            else:
                obs = np.interp(time_point, kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0])
            observed_survivals.append(obs)

        observed_survivals = pd.Series(observed_survivals, index=group_means.index)

        ax.scatter(group_means, observed_survivals,
                   color=colors[model_name], marker=markers[model_name],
                   s=180 if model_name == 'GBS' else 150,
                   label=model_name, alpha=0.8, edgecolors='black', linewidth=2)

        sorted_idx = np.argsort(group_means.values)
        ax.plot(group_means.values[sorted_idx], observed_survivals.values[sorted_idx],
                color=colors[model_name], linestyle=linestyles[model_name],
                alpha=0.6, linewidth=2.5 if model_name == 'GBS' else 2.0)

    ax.set_xlabel("Model Predicted Survival Probability", fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel("Actual Survival Probability", fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
    plt.close()


time_point_3y = 36

plot_calibration_comparison(
    train_df, final_features,
    time_col, event_col,
    time_point_3y,
    'Training Set (36 Months)',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Train_3y.eps")
)

plot_calibration_comparison(
    test_df, final_features,
    time_col, event_col,
    time_point_3y,
    'Testing Set (36 Months)',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Test_3y.eps")
)

time_point_5y = 60

plot_calibration_comparison(
    train_df, final_features,
    time_col, event_col,
    time_point_5y,
    'Training Set (60 Months)',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Train_5y.eps")
)

plot_calibration_comparison(
    test_df, final_features,
    time_col, event_col,
    time_point_5y,
    'Testing Set (60 Months)',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Test_5y.eps")
)
