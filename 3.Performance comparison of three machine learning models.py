import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "..."
FIG_DPI = 300
N_BOOTSTRAP = 1000

MODEL_NAMES = {
    'GBS': 'GBS',
    'RSF': 'RSF',
    'SSVM': 'SSVM',
}

ROC_SETTINGS = {
    'time_points': [36, 60],
    'title_train': 'Training set ({time_point} months)',
    'title_test': 'Testing set ({time_point} months)',
    'xlabel': '1 - Specificity',
    'ylabel': 'Sensitivity',
    'legend_loc': 'lower right',
    'legend_template': '{model_name} (AUC = {auc}, 95% CI: {ci_lower}-{ci_upper})',
}


def format_title(template, time_point):
    time_year = int(time_point / 12)
    return template.replace('{time_year}', str(time_year)).replace('{time_point}', str(time_point))


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
            score = roc_auc_score(y_true_valid[indices], y_scores_valid[indices])
            bootstrapped_scores.append(score)
        except:
            continue
    if not bootstrapped_scores:
        return np.nan, np.nan, np.nan
    mean_auc = np.mean(bootstrapped_scores)
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)
    return mean_auc, ci_lower, ci_upper


def calculate_time_dependent_roc_with_ci(model, X, T, E, time_point, n_bootstrap=1000):
    try:
        risk_scores = model.predict(X)
        y_true = np.zeros(len(X))
        y_true[(T <= time_point) & (E == 1)] = 1
        valid_mask = ((T <= time_point) & (E == 1)) | (T > time_point)
        y_true_valid = y_true[valid_mask]
        y_score_valid = risk_scores[valid_mask]
        if len(np.unique(y_true_valid)) < 2:
            return None, None, np.nan, np.nan, np.nan
        fpr, tpr, _ = roc_curve(y_true_valid, y_score_valid)
        auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(
            y_true_valid, y_score_valid, n_bootstrap
        )
        return fpr, tpr, auc_mean, ci_lower, ci_upper
    except Exception as e:
        return None, None, np.nan, np.nan, np.nan


def format_legend_label(template, model_name, auc_mean, ci_lower, ci_upper):
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        ci_str = "N/A"
    else:
        ci_str = f"{ci_lower:.3f}-{ci_upper:.3f}"
    label = template.replace('{model_name}', model_name)
    label = label.replace('{auc}', f"{auc_mean:.3f}")
    label = label.replace('{ci_lower}', f"{ci_lower:.3f}" if not np.isnan(ci_lower) else "N/A")
    label = label.replace('{ci_upper}', f"{ci_upper:.3f}" if not np.isnan(ci_upper) else "N/A")
    return label


def plot_roc_comparison(models_dict, X_train, T_train, E_train, X_test, T_test, E_test, config, output_dir):
    model_keys = list(models_dict.keys())
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']

    for time_point in config['time_points']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

        ax1.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='Random guess')
        for idx, model_key in enumerate(model_keys):
            model = models_dict[model_key]
            if model is None:
                continue
            fpr, tpr, auc_mean, ci_lower, ci_upper = calculate_time_dependent_roc_with_ci(
                model, X_train, T_train, E_train, time_point, N_BOOTSTRAP
            )
            if fpr is not None and not np.isnan(auc_mean):
                display_name = MODEL_NAMES.get(model_key, model_key)
                label = format_legend_label(
                    config['legend_template'], display_name, auc_mean, ci_lower, ci_upper
                )
                ax1.plot(fpr, tpr, color=colors[idx], lw=3, label=label)

        title = format_title(config['title_train'], time_point)
        ax1.set_title(title, fontsize=26, fontweight='bold', pad=20)
        ax1.set_xlabel(config['xlabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax1.set_ylabel(config['ylabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax1.legend(loc=config['legend_loc'], fontsize=10)
        ax1.grid(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.tick_params(labelsize=20, width=1.5, length=6)

        ax2.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='Random guess')
        for idx, model_key in enumerate(model_keys):
            model = models_dict[model_key]
            if model is None:
                continue
            fpr, tpr, auc_mean, ci_lower, ci_upper = calculate_time_dependent_roc_with_ci(
                model, X_test, T_test, E_test, time_point, N_BOOTSTRAP
            )
            if fpr is not None and not np.isnan(auc_mean):
                display_name = MODEL_NAMES.get(model_key, model_key)
                label = format_legend_label(
                    config['legend_template'], display_name, auc_mean, ci_lower, ci_upper
                )
                ax2.plot(fpr, tpr, color=colors[idx], lw=3, label=label)

        title = format_title(config['title_test'], time_point)
        ax2.set_title(title, fontsize=26, fontweight='bold', pad=20)
        ax2.set_xlabel(config['xlabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax2.set_ylabel(config['ylabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax2.legend(loc=config['legend_loc'], fontsize=10)
        ax2.grid(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.tick_params(labelsize=20, width=1.5, length=6)

        save_name = f"ROC_Comparison_{time_point}m.eps"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, format='eps', dpi=FIG_DPI, bbox_inches='tight')
        plt.close()


def main():
    try:
        train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "2_train_data_scaled.csv"))
        test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "2_test_data_scaled.csv"))
        features_df = pd.read_csv(os.path.join(OUTPUT_DIR, "3_Final_Features_For_Modeling.csv"))
        final_features = features_df['Final_Feature_Name'].tolist()

        X_train = train_df[final_features]
        X_test = test_df[final_features]

        T_train = train_df['DFS_time'].values
        E_train = train_df['DFS_event'].values
        T_test = test_df['DFS_time'].values
        E_test = test_df['DFS_event'].values
    except Exception as e:
        return

    models = {}

    model_files = {
        'GBS': '4_gbs_model.pkl',
        'RSF': '4_rsf_model.pkl',
        'SSVM': '4_ssvm_model.pkl'
    }

    for name, file in model_files.items():
        try:
            models[name] = joblib.load(os.path.join(OUTPUT_DIR, file))
        except:
            models[name] = None

    plot_roc_comparison(
        models_dict=models,
        X_train=X_train,
        T_train=T_train,
        E_train=E_train,
        X_test=X_test,
        T_test=T_test,
        E_test=E_test,
        config=ROC_SETTINGS,
        output_dir=OUTPUT_DIR
    )


if __name__ == '__main__':
    main()


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
import joblib
import warnings
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

ml_models_config = {
    'Gradient Boosting Survival': {'file': '4_gbs_model.pkl', 'short': 'GBS'},
    'Random Survival Forest': {'file': '4_rsf_model.pkl', 'short': 'RSF'},
    'Survival SVM': {'file': '4_ssvm_model.pkl', 'short': 'SSVM'}
}

plot_order = ['GBS', 'RSF', 'SSVM']

model_colors = {
    'GBS': '#ff7f0e',
    'RSF': '#1f77b4',
    'SSVM': '#2ca02c'
}

N_BOOTSTRAP = 1000

output_dir = "..."

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

X_train = train_df[final_features]
X_test = test_df[final_features]

time_col = 'DFS_time'
event_col = 'DFS_event'

T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values


def calculate_cindex_with_bootstrap(model, X, T, E, n_bootstrap=1000):
    try:
        risk_scores = model.predict(X)
        c_index = concordance_index(T, -risk_scores, E)
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan

    c_indices = []
    n_samples = len(T)

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[indices]
        T_boot = T[indices]
        E_boot = E[indices]
        try:
            risk_boot = model.predict(X_boot)
            c_boot = concordance_index(T_boot, -risk_boot, E_boot)
            c_indices.append(c_boot)
        except:
            continue

    if len(c_indices) > 0:
        std_error = np.std(c_indices)
        ci_lower = np.percentile(c_indices, 2.5)
        ci_upper = np.percentile(c_indices, 97.5)
    else:
        std_error = 0.0
        ci_lower = c_index
        ci_upper = c_index

    return c_index, std_error, ci_lower, ci_upper


all_results = []

ordered_model_names = ['Gradient Boosting Survival', 'Random Survival Forest', 'Survival SVM']

for model_name in ordered_model_names:
    config = ml_models_config[model_name]

    model_path = os.path.join(output_dir, config['file'])
    if not os.path.exists(model_path):
        continue

    try:
        model = joblib.load(model_path)
    except Exception as e:
        continue

    train_c, train_se, train_ci_l, train_ci_u = calculate_cindex_with_bootstrap(model, X_train, T_train, E_train, N_BOOTSTRAP)
    test_c, test_se, test_ci_l, test_ci_u = calculate_cindex_with_bootstrap(model, X_test, T_test, E_test, N_BOOTSTRAP)

    all_results.append({
        'Model_Full': model_name,
        'Model': config['short'],
        'Train_C_index': train_c,
        'Train_SE': train_se,
        'Train_CI_Lower': train_ci_l,
        'Train_CI_Upper': train_ci_u,
        'Test_C_index': test_c,
        'Test_SE': test_se,
        'Test_CI_Lower': test_ci_l,
        'Test_CI_Upper': test_ci_u,
        'Gap': train_c - test_c
    })

summary_df = pd.DataFrame(all_results)
summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=plot_order, ordered=True)
summary_df = summary_df.sort_values('Model')

summary_path = os.path.join(output_dir, "9_C_Index_Comparison_ML_Only.csv")
summary_df.to_csv(summary_path, index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

x = np.arange(len(summary_df))
width = 0.35
bar_colors = [model_colors[m] for m in summary_df['Model']]

train_errors = np.array([
    [row['Train_C_index'] - row['Train_CI_Lower'], row['Train_CI_Upper'] - row['Train_C_index']]
    for _, row in summary_df.iterrows()
]).T

bars1 = ax1.bar(x - width / 2, summary_df['Train_C_index'], width,
                yerr=train_errors, capsize=5, color=bar_colors,
                edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Model', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_title('Training Set', fontsize=26, fontweight='bold', pad=20)
ax1.set_xticks(x - width / 2)
ax1.set_xticklabels(summary_df['Model'])
ax1.legend(fontsize=10)
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([0, 1.1])
ax1.tick_params(labelsize=20, width=1.5, length=6)

for idx, bar in enumerate(bars1):
    height = bar.get_height()
    if not np.isnan(height):
        y_pos = height + train_errors[1][idx] + 0.03
        ax1.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{height:.3f}', ha='center', va='bottom',
                 fontsize=20, fontweight='bold')

test_errors = np.array([
    [row['Test_C_index'] - row['Test_CI_Lower'], row['Test_CI_Upper'] - row['Test_C_index']]
    for _, row in summary_df.iterrows()
]).T

bars2 = ax2.bar(x + width / 2, summary_df['Test_C_index'], width,
                yerr=test_errors, capsize=5, color=bar_colors,
                edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Model', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_title('Testing Set', fontsize=26, fontweight='bold', pad=20)
ax2.set_xticks(x + width / 2)
ax2.set_xticklabels(summary_df['Model'])
ax2.legend(fontsize=10)
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim([0, 1.1])
ax2.tick_params(labelsize=20, width=1.5, length=6)

for idx, bar in enumerate(bars2):
    height = bar.get_height()
    if not np.isnan(height):
        y_pos = height + test_errors[1][idx] + 0.03
        ax2.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{height:.3f}', ha='center', va='bottom',
                 fontsize=20, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(output_dir, "9_C_Index_Comparison_ML_Only.eps")
plt.savefig(plot_path, format='eps', dpi=300, bbox_inches='tight')
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

TITLES = {
    'train_3y': 'Training Set (36 Months)',
    'test_3y': 'Testing Set (36 Months)',
    'train_5y': 'Training Set (60 Months)',
    'test_5y': 'Testing Set (60 Months)'
}

AXIS_LABELS = {
    'xlabel_3y': 'Model Predicted Survival Probability',
    'ylabel_3y': 'Actual Survival Probability',
    'xlabel_5y': 'Model Predicted Survival Probability',
    'ylabel_5y': 'Actual Survival Probability'
}

output_dir = "..."

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

time_col = 'DFS_time'
event_col = 'DFS_event'

models = {}

model_names = ['GBS', 'RSF', 'SSVM']

for name in model_names:
    try:
        if name == 'GBS':
            models[name] = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
        elif name == 'RSF':
            models[name] = joblib.load(os.path.join(output_dir, "4_rsf_model.pkl"))
        elif name == 'SSVM':
            models[name] = joblib.load(os.path.join(output_dir, "4_ssvm_model.pkl"))
    except Exception as e:
        models[name] = None


def predict_survival_probability(model, model_name, X, time_point):
    try:
        if model_name in ['RSF', 'GBS']:
            risk_scores = model.predict(X)
            min_risk = risk_scores.min()
            max_risk = risk_scores.max()
            if max_risk > min_risk:
                normalized_risk = (risk_scores - min_risk) / (max_risk - min_risk)
            else:
                normalized_risk = np.zeros_like(risk_scores)
            pred_survival = 1 - normalized_risk
            return pred_survival
        elif model_name == 'SSVM':
            risk_scores = model.predict(X)
            min_risk = risk_scores.min()
            max_risk = risk_scores.max()
            if max_risk > min_risk:
                normalized_risk = (risk_scores - min_risk) / (max_risk - min_risk)
            else:
                normalized_risk = np.zeros_like(risk_scores)
            pred_survival = 1 - normalized_risk
            return pred_survival
        else:
            return None
    except Exception as e:
        return None


def plot_calibration_comparison_fixed(models, model_names, df, features, time_col, event_col, time_point, title, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    markers = ['s', 'o', '^']

    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Ideal Calibration Curve', alpha=0.6)

    for idx, model_name in enumerate(model_names):
        if models[model_name] is None:
            continue

        pred_survival = predict_survival_probability(
            models[model_name], model_name, df[features], time_point
        )

        if pred_survival is None:
            continue

        pred_survival = np.clip(pred_survival, 0.001, 0.999)

        df_temp = df.copy()
        df_temp['pred_survival'] = pred_survival

        try:
            df_temp['group'] = pd.qcut(df_temp['pred_survival'], q=4, duplicates='drop')
        except Exception as e:
            continue

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
                   color=colors[idx], marker=markers[idx], s=180,
                   label=model_name, alpha=0.8, edgecolors='black', linewidth=2)

        sorted_idx = np.argsort(group_means.values)
        ax.plot(group_means.values[sorted_idx], observed_survivals.values[sorted_idx],
                color=colors[idx], alpha=0.5, linewidth=2.5)

    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=22, width=1.5, length=6)

    plt.tight_layout()
    plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
    plt.close()


time_point_3y = 36

plot_calibration_comparison_fixed(
    models, model_names,
    train_df, final_features,
    time_col, event_col,
    time_point_3y,
    TITLES['train_3y'],
    AXIS_LABELS['xlabel_3y'],
    AXIS_LABELS['ylabel_3y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Train_3y.eps")
)

plot_calibration_comparison_fixed(
    models, model_names,
    test_df, final_features,
    time_col, event_col,
    time_point_3y,
    TITLES['test_3y'],
    AXIS_LABELS['xlabel_3y'],
    AXIS_LABELS['ylabel_3y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Test_3y.eps")
)

time_point_5y = 60

plot_calibration_comparison_fixed(
    models, model_names,
    train_df, final_features,
    time_col, event_col,
    time_point_5y,
    TITLES['train_5y'],
    AXIS_LABELS['xlabel_5y'],
    AXIS_LABELS['ylabel_5y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Train_5y.eps")
)

plot_calibration_comparison_fixed(
    models, model_names,
    test_df, final_features,
    time_col, event_col,
    time_point_5y,
    TITLES['test_5y'],
    AXIS_LABELS['xlabel_5y'],
    AXIS_LABELS['ylabel_5y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Test_5y.eps")
)

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

ml_models = ['Gradient Boosting Survival', 'Random Survival Forest', 'Survival SVM']

model_names_short = {
    'Gradient Boosting Survival': 'GBS',
    'Random Survival Forest': 'RSF',
    'Survival SVM': 'SSVM'
}

ROC_SUBTITLE_TEMPLATE = '{model_short}\nYouden Index: {youden}'
ROC_XLABEL = 'False Positive Rate'
ROC_YLABEL = 'True Positive Rate'
CM_XLABEL = 'Predicted'
CM_YLABEL = 'Actual'

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

models = {}

model_files = {
    'Gradient Boosting Survival': '4_gbs_model.pkl',
    'Random Survival Forest': '4_rsf_model.pkl',
    'Survival SVM': '4_ssvm_model.pkl'
}

for model_name, file_name in model_files.items():
    try:
        models[model_name] = joblib.load(os.path.join(output_dir, file_name))
    except Exception as e:
        models[model_name] = None


def create_time_dependent_labels(T, E, time_point):
    y_true = np.zeros(len(T))
    y_true[(T <= time_point) & (E == 1)] = 1
    return y_true


def predict_risk_scores(model, X):
    if model is None:
        return np.zeros(len(X))
    risk_scores = model.predict(X)
    try:
        survival_probs = model.predict_survival_function(X, times=[TIME_POINT_MONTHS / 12]).iloc[:, 0]
        risk_scores = 1 - survival_probs
    except:
        pass
    return risk_scores


def calculate_roc_and_youden(model, X, y_true, T, E, time_point):
    risk_scores = predict_risk_scores(model, X)
    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    risk_scores_valid = risk_scores[valid_mask]
    if len(np.unique(y_true_valid)) < 2:
        return None, None, None, None, None, None
    fpr, tpr, thresholds = roc_curve(y_true_valid, risk_scores_valid)
    youden_index = tpr - fpr
    max_index = np.argmax(youden_index)
    optimal_threshold = thresholds[max_index]
    max_youden = youden_index[max_index]
    return fpr, tpr, thresholds, youden_index, optimal_threshold, max_youden


def calculate_confusion_matrix_with_threshold(model, X, y_true, T, E, threshold, time_point):
    risk_scores = predict_risk_scores(model, X)
    y_pred = (risk_scores >= threshold).astype(int)
    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    report = classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'threshold': threshold
    }


def format_cm_label(value, total):
    percentage = value / total * 100
    return f"{percentage:.1f}%\n(n={value})"


def cm_to_percentage_matrix(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    percentages = cm / row_sums * 100
    return percentages


def get_roc_subtitle(model_short, youden):
    try:
        return ROC_SUBTITLE_TEMPLATE.replace('{model_short}', model_short).replace('{youden}', f'{youden:.3f}')
    except:
        return model_short + '\nYouden Index: ' + f'{youden:.3f}'


y_train = create_time_dependent_labels(T_train, E_train, TIME_POINT_MONTHS)
y_test = create_time_dependent_labels(T_test, E_test, TIME_POINT_MONTHS)

all_results = {}

for model_name in ml_models:
    if models[model_name] is None:
        continue

    fpr_train, tpr_train, thresholds_train, youden_train, optimal_threshold_train, max_youden_train = calculate_roc_and_youden(
        models[model_name], X_train, y_train, T_train, E_train, TIME_POINT_MONTHS
    )

    if optimal_threshold_train is None:
        continue

    metrics_train = calculate_confusion_matrix_with_threshold(
        models[model_name], X_train, y_train, T_train, E_train, optimal_threshold_train, TIME_POINT_MONTHS
    )

    metrics_test = calculate_confusion_matrix_with_threshold(
        models[model_name], X_test, y_test, T_test, E_test, optimal_threshold_train, TIME_POINT_MONTHS
    )

    all_results[model_name] = {
        'train': metrics_train,
        'test': metrics_test,
        'optimal_threshold': optimal_threshold_train,
        'youren_index': max_youden_train
    }

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(wspace=0.35, bottom=0.2, top=0.88)

for i, model_name in enumerate(ml_models):
    if model_name not in all_results:
        continue

    model_short = model_names_short.get(model_name, model_name)

    fpr_train, tpr_train, thresholds_train, youden_train, optimal_threshold_train, max_youden_train = calculate_roc_and_youden(
        models[model_name], X_train, y_train, T_train, E_train, TIME_POINT_MONTHS
    )

    if fpr_train is None:
        continue

    ax = axes[i]

    ax.plot(fpr_train, tpr_train, color='blue', lw=3, label='ROC Curve')
    ax.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='Random Guess')

    idx = np.argmax(youden_train)
    ax.plot(fpr_train[idx], tpr_train[idx], 'ro', markersize=12,
            label=f'Best Threshold ({optimal_threshold_train:.3f})')

    ax.set_xlabel(ROC_XLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel(ROC_YLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title(get_roc_subtitle(model_short, max_youden_train), fontsize=26, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(labelsize=20, width=1.5, length=6)

plot_path = os.path.join(output_dir, "9_ROC_Youden_Index_Analysis.eps")
plt.savefig(plot_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(3, 2, figsize=(16, 24))
plt.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.15, top=0.94)

for i, model_name in enumerate(ml_models):
    if model_name not in all_results:
        continue

    model_short = model_names_short.get(model_name, model_name)

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

    ax_train.set_xlabel(CM_XLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_ylabel(CM_YLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_title('Training Set (' + model_short + ')', fontsize=26, fontweight='bold', pad=20)
    ax_train.tick_params(labelsize=20, width=1.5, length=6)

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

    ax_test.set_xlabel(CM_XLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_ylabel(CM_YLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_title('Testing Set (' + model_short + ')', fontsize=26, fontweight='bold', pad=20)
    ax_test.tick_params(labelsize=20, width=1.5, length=6)

confusion_matrix_path = os.path.join(output_dir, "9_Confusion_Matrix_Youden_Index.eps")
plt.savefig(confusion_matrix_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()
