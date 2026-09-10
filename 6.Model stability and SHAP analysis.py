import pandas as pd
import numpy as np
import os
import warnings
from scipy import stats
import matplotlib.pyplot as plt

try:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from lifelines.utils import concordance_index
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

output_dir = "..."
N_BOOTSTRAPS = 100
BASE_RANDOM_STATE = 42
time_col = 'DFS_time'
event_col = 'DFS_event'

GBS_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_samples_split': 10,
    'n_estimators': 200,
    'subsample': 0.7
}

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()


def get_structured_y(df):
    return np.array([(df.loc[i, event_col], df.loc[i, time_col]) for i in df.index],
                    dtype=[('event', '?'), ('time', '<f8')])


X_test = test_df[final_features].values
y_test_struct = get_structured_y(test_df)
T_test = test_df[time_col].values
E_test = test_df[event_col].values

X_train_full = train_df[final_features].values
y_train_struct_full = get_structured_y(train_df)


def calculate_icc_single(data):
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return np.nan
    k = data_clean.shape[1]
    n = data_clean.shape[0]
    grand_mean = data_clean.values.mean()
    row_means = data_clean.mean(axis=1).values
    col_means = data_clean.mean(axis=0).values
    SS_total = ((data_clean.values - grand_mean) ** 2).sum()
    SS_rows = k * ((row_means - grand_mean) ** 2).sum()
    SS_cols = n * ((col_means - grand_mean) ** 2).sum()
    SS_error = SS_total - SS_rows - SS_cols
    MS_rows = SS_rows / (n - 1) if n > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0
    if MS_rows + (k - 1) * MS_error == 0:
        return np.nan
    icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)
    return icc


predictions_cols = [f'Iter_{i}' for i in range(N_BOOTSTRAPS)]
predictions_df = pd.DataFrame(index=test_df.index, columns=predictions_cols)
c_index_list = []

if not SKSURV_AVAILABLE:
    pass
else:
    for i in range(N_BOOTSTRAPS):
        current_seed = BASE_RANDOM_STATE + i
        sample_indices = np.random.choice(len(X_train_full), size=len(X_train_full), replace=True)
        X_train_bs = X_train_full[sample_indices]
        y_train_bs = y_train_struct_full[sample_indices]
        current_params = GBS_PARAMS.copy()
        current_params['random_state'] = current_seed
        model = GradientBoostingSurvivalAnalysis(**current_params)
        model.fit(X_train_bs, y_train_bs)
        pred_risk = model.predict(X_test)
        c_val = concordance_index(T_test, -pred_risk, E_test)
        c_index_list.append(c_val)
        predictions_df[predictions_cols[i]] = pred_risk

icc_value = calculate_icc_single(predictions_df)
c_mean = np.mean(c_index_list)
c_std = np.std(c_index_list)

pred_save_path = os.path.join(output_dir, "15_GBS_Bootstrap_Predictions.csv")
predictions_df.to_csv(pred_save_path)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

np.random.seed(0)
sample_indices_to_plot = np.random.choice(len(predictions_df), size=min(5, len(predictions_df)), replace=False)

plot_data = predictions_df.iloc[sample_indices_to_plot].T
plot_data.plot(kind='box', ax=axes[0])
axes[0].set_title("Predicted Risk Score Variability Example", fontsize=26, fontweight='bold', pad=20)
axes[0].set_ylabel('Predicted Risk Score', fontsize=22, fontweight='bold', labelpad=15)
axes[0].set_xlabel('Patient ID (Test Set)', fontsize=22, fontweight='bold', labelpad=15)
axes[0].grid(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].tick_params(labelsize=20)

axes[1].plot(range(1, N_BOOTSTRAPS + 1), c_index_list, marker='o', linestyle='-', color='#1f77b4', markersize=4)
axes[1].axhline(y=c_mean, color='r', linestyle='--', linewidth=2, label=f'Mean ({c_mean:.4f})')
axes[1].fill_between(range(1, N_BOOTSTRAPS + 1), c_mean - c_std, c_mean + c_std, color='r', alpha=0.1, label=f'\u00b11 SD')
axes[1].set_title("Test Set C-index Variability Trend", fontsize=26, fontweight='bold', pad=20)
axes[1].set_xlabel('Iteration', fontsize=22, fontweight='bold', labelpad=15)
axes[1].set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
axes[1].legend(fontsize=10)
axes[1].grid(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(labelsize=20)
axes[1].set_ylim([max(0.5, c_mean - 0.1), min(1.0, c_mean + 0.1)])

plt.tight_layout()
plot_save_path = os.path.join(output_dir, "15_GBS_Stability_Analysis.eps")
plt.savefig(plot_save_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()

import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from textwrap import wrap

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

output_dir = "..."
time_col = 'PFS_time'
event_col = 'PFS_event'

test_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()
X_test = test_df[final_features]

try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
except Exception as e:
    exit()

shap_importance = None
try:
    explainer = shap.TreeExplainer(gbs_model)
    shap_values = explainer.shap_values(X_test)
    shap_importance = np.abs(shap_values).mean(axis=0)
except Exception as e:
    try:
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        explainer = shap.KernelExplainer(gbs_model.predict, X_sample)
        shap_values = explainer.shap_values(X_test, nsamples=100)
        shap_importance = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        exit()

if shap_importance is None:
    exit()

feature_importance = pd.DataFrame({
    'Feature': final_features,
    'SHAP_Importance': shap_importance
})

top_10_features = feature_importance.sort_values(by='SHAP_Importance', ascending=False).head(10)
top_10_feature_names = top_10_features['Feature'].tolist()

top_features_path = os.path.join(output_dir, "16_GBS_SHAP_Top10_Features.csv")
top_10_features.to_csv(top_features_path, index=False)

top_10_indices = [final_features.index(feature) for feature in top_10_feature_names]
top_10_shap_values = shap_values[:, top_10_indices]

plt.figure(figsize=(12, 10))
shap.summary_plot(
    top_10_shap_values,
    X_test[top_10_feature_names],
    feature_names=top_10_feature_names,
    plot_type="dot",
    show=False,
    color_bar=True
)

ax = plt.gca()
ax.set_xlabel('SHAP value', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Feature', fontsize=14, fontweight='bold', labelpad=15)

y_labels = ax.get_yticklabels()
new_y_labels = []
for label in y_labels:
    text = label.get_text()
    wrapped_text = '\n'.join(wrap(text, width=20))
    new_y_labels.append(wrapped_text)
ax.set_yticklabels(new_y_labels, fontsize=11)
ax.tick_params(axis='x', labelsize=12)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.gcf()
for current_ax in fig.axes:
    if current_ax != ax:
        if current_ax.get_ylabel():
            current_ax.set_ylabel('Feature value', fontsize=12, labelpad=15)
            yticklabels = current_ax.get_yticklabels()
            new_yticklabels = []
            for label in yticklabels:
                text = label.get_text()
                if 'low' in text.lower() or text == 'Low':
                    text = 'Low'
                elif 'high' in text.lower() or text == 'High':
                    text = 'High'
                new_yticklabels.append(text)
            current_ax.set_yticklabels(new_yticklabels, fontsize=10)

plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)

plot_path = os.path.join(output_dir, "16_GBS_SHAP_Top10_Summary_Plot_Wrapped.eps")
plt.savefig(plot_path, format='eps', dpi=300, bbox_inches='tight')
plt.close()
