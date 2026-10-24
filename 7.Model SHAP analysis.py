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