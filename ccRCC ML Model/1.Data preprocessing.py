import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

data_path = "..."
output_dir = "..."

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv(data_path)

id_var = df.columns[0]
outcome_vars = df.columns[1:4].tolist()
clinical_risk_models = df.columns[4:7].tolist()
clinical_vars = df.columns[7:16].tolist()
clinical_cat_vars = df.columns[7:13].tolist()
clinical_cont_vars = df.columns[13:16].tolist()
radiomics_vars = df.columns[16:].tolist()

from sklearn.model_selection import train_test_split

random_seed = 42

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=random_seed,
    stratify=df['Recurrence']
)

train_save_path = os.path.join(output_dir, "data_train.csv")
test_save_path = os.path.join(output_dir, "data_test.csv")

train_df.to_csv(train_save_path, index=False)
test_df.to_csv(test_save_path, index=False)

from sklearn.preprocessing import StandardScaler
import joblib

continuous_features = clinical_cont_vars + radiomics_vars

scaler = StandardScaler()

scaler.fit(train_df[continuous_features])

train_df[continuous_features] = scaler.transform(train_df[continuous_features])
test_df[continuous_features] = scaler.transform(test_df[continuous_features])

scaler_save_path = os.path.join(output_dir, "data_standard_scaler.pkl")
joblib.dump(scaler, scaler_save_path)

train_scaled_path = os.path.join(output_dir, "data_train_scaled.csv")
test_scaled_path = os.path.join(output_dir, "data_test_scaled.csv")

train_df.to_csv(train_scaled_path, index=False)
test_df.to_csv(test_scaled_path, index=False)

from scipy.stats import mannwhitneyu, chi2_contingency

group_0 = train_df[train_df['Recurrence'] == 0]
group_1 = train_df[train_df['Recurrence'] == 1]

features_to_screen = clinical_vars + radiomics_vars

p_values = []
significant_features = []

for feature in features_to_screen:
    p_val = np.nan
    method_used = "Unknown"
    try:
        if feature in clinical_cat_vars:
            crosstab = pd.crosstab(train_df[feature], train_df['Recurrence'])
            chi2, p, dof, expected = chi2_contingency(crosstab)
            p_val = p
            method_used = "Chi2"
        else:
            data_0 = group_0[feature]
            data_1 = group_1[feature]
            u_stat, p = mannwhitneyu(data_0, data_1, alternative='two-sided')
            p_val = p
            method_used = "MWU"
        p_values.append({'Feature': feature, 'P_Value': p_val, 'Method': method_used})
        if p_val < 0.05:
            significant_features.append(feature)
    except Exception as e:
        p_values.append({'Feature': feature, 'P_Value': np.nan, 'Method': "Error"})

univariate_results_df = pd.DataFrame(p_values)
univariate_results_df = univariate_results_df.sort_values(by='P_Value')

uni_save_path = os.path.join(output_dir, "feature_univariate_results.csv")
univariate_results_df.to_csv(uni_save_path, index=False)

features_list_path = os.path.join(output_dir, "feature_selected_univariate.csv")
pd.DataFrame(significant_features, columns=['Selected_Features']).to_csv(features_list_path, index=False)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

corr_matrix = train_df[significant_features].corr().abs()

plt.figure(figsize=(12, 10))
plot_features = significant_features[:50]
sns.heatmap(train_df[plot_features].corr().abs(), cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
plt.title("Correlation Matrix of Top 50 Features (Clinical + Radiomics)")
heatmap_path = os.path.join(output_dir, "feature_correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()

corr_threshold = 0.9

upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]

features_no_corr = [f for f in significant_features if f not in to_drop]

pd.DataFrame(features_no_corr, columns=['Uncorrelated_Features']).to_csv(
    os.path.join(output_dir, "feature_selected_uncorrelated.csv"),
    index=False
)

from sklearn.linear_model import LassoCV, lasso_path

X_train_lasso = train_df[features_no_corr]
y_train_lasso = train_df['Recurrence']

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
lasso_cv.fit(X_train_lasso, y_train_lasso)

lasso_coefs = pd.Series(lasso_cv.coef_, index=X_train_lasso.columns)
selected_lasso_features = lasso_coefs[lasso_coefs != 0].index.tolist()

if len(selected_lasso_features) > 0:
    final_lasso_results = pd.DataFrame({
        'Feature': selected_lasso_features,
        'Coefficient': lasso_coefs[selected_lasso_features]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(final_lasso_results['Feature'], final_lasso_results['Coefficient'], color='lightblue')
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Coefficient")
    plt.title("Lasso Regression Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_lasso_coefficients.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), marker='o', color='red')
    plt.axvline(lasso_cv.alpha_, color='black', linestyle='--', label=f'Optimal Alpha: {lasso_cv.alpha_:.6f}')
    plt.xlabel("Lambda (Alpha)")
    plt.ylabel("MSE")
    plt.title("MSE vs Lambda (LassoCV Cross-Validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_lasso_mse_vs_lambda.eps"), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    alphas, coefs, _ = lasso_path(X_train_lasso, y_train_lasso)
    plt.semilogx(alphas, coefs.T)
    plt.axvline(lasso_cv.alpha_, color='black', linestyle='--', label=f'Optimal Alpha: {lasso_cv.alpha_:.6f}')
    plt.xlabel("Lambda (Alpha)")
    plt.ylabel("Coefficients")
    plt.title("Lasso Coefficient Paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_lasso_coefficient_paths.eps"), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    lasso_result_path = os.path.join(output_dir, "feature_selected_lasso.csv")
    final_lasso_results.to_csv(lasso_result_path, index=False)
else:
    final_lasso_results = pd.DataFrame(columns=['Feature', 'Coefficient'])

final_features_for_modeling = selected_lasso_features

final_list_path = os.path.join(output_dir, "feature_final_for_modeling.csv")
pd.DataFrame(final_features_for_modeling, columns=['Final_Feature_Name']).to_csv(final_list_path, index=False)
