import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import joblib
from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

output_dir = "..."

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

if 'DFS_time' not in train_df.columns or 'DFS_event' not in train_df.columns:
    time_col = 'Time'
    event_col = 'Event'
else:
    time_col = 'DFS_time'
    event_col = 'DFS_event'

X_train = train_df[final_features].copy()
X_test = test_df[final_features].copy()

T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values

evaluation_times = [36, 60]


def bootstrap_auc(y_true, y_score, n_bootstraps=1000):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), random_state=i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = roc_auc_score(y_true[indices], y_score[indices])
            bootstrapped_scores.append(score)
        except:
            continue
    if bootstrapped_scores:
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        ci_lower = np.percentile(sorted_scores, 2.5)
        ci_upper = np.percentile(sorted_scores, 97.5)
        return np.mean(sorted_scores), ci_lower, ci_upper
    else:
        return np.nan, np.nan, np.nan


def time_dependent_auc(model, X, T, E, time_point):
    try:
        if hasattr(model, 'predict_partial_hazard'):
            risk_scores = model.predict_partial_hazard(X)
            if hasattr(risk_scores, 'values'):
                risk_scores = risk_scores.values
        elif hasattr(model, 'predict'):
            risk_scores = model.predict(X)
        else:
            return np.nan, np.nan, np.nan

        y_true_binary = np.zeros(len(X))
        y_true_binary[(T <= time_point) & (E == 1)] = 1
        y_true_binary[(T > time_point)] = 0

        valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
        y_true_binary = y_true_binary[valid_mask]
        y_score_binary = risk_scores[valid_mask]

        n_events = np.sum(y_true_binary == 1)
        n_non_events = np.sum(y_true_binary == 0)

        if n_events == 0 or n_non_events == 0:
            return np.nan, np.nan, np.nan

        if len(np.unique(y_true_binary)) < 2:
            return np.nan, np.nan, np.nan

        try:
            auc_value = roc_auc_score(y_true_binary, y_score_binary)
            se = np.sqrt(auc_value * (1 - auc_value) / (n_events + n_non_events))
            ci_lower = auc_value - 1.96 * se
            ci_upper = auc_value + 1.96 * se
            ci_lower = max(0, ci_lower)
            ci_upper = min(1, ci_upper)
            return auc_value, ci_lower, ci_upper
        except Exception as e:
            return np.nan, np.nan, np.nan
    except Exception as e:
        return np.nan, np.nan, np.nan


def prepare_survival_data(T, E):
    structured_array = np.array(
        [(E[i], T[i]) for i in range(len(T))],
        dtype=[('event', 'bool'), ('time', 'float64')]
    )
    return structured_array


try:
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored
        from sklearn.model_selection import ParameterGrid
    except ImportError:
        raise

    y_train_struct = prepare_survival_data(T_train, E_train)
    y_test_struct = prepare_survival_data(T_test, E_test)

    param_combinations = [
        {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 10, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 6, 'min_samples_leaf': 3, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 8, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 30, 'min_samples_leaf': 15, 'random_state': 42},
    ]

    best_score = -np.inf
    best_params = None
    best_model = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in param_combinations:
        fold_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train_struct[train_idx]
            y_val_fold = y_train_struct[val_idx]
            try:
                model_params = {k: v for k, v in params.items() if k != 'criterion'}
                rsf = RandomSurvivalForest(**model_params)
                rsf.fit(X_train_fold, y_train_fold)
                predictions = rsf.predict(X_val_fold)
                c_index = concordance_index_censored(
                    y_val_fold['event'], y_val_fold['time'], predictions
                )[0]
                fold_scores.append(c_index)
            except Exception as e:
                fold_scores.append(0)
                continue

        if fold_scores:
            avg_score = np.mean(fold_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    try:
        best_model_params = {k: v for k, v in best_params.items() if k != 'criterion'}
        best_model = RandomSurvivalForest(**best_model_params)
        best_model.fit(X_train, y_train_struct)
    except Exception as e:
        pass

    if best_model is None:
        best_model = RandomSurvivalForest(random_state=42)
        best_model.fit(X_train, y_train_struct)
        best_params = {'random_state': 42}

    rsf_model_path = os.path.join(output_dir, "4_rsf_model.pkl")
    joblib.dump(best_model, rsf_model_path)

    rsf_train_risk = best_model.predict(X_train)
    rsf_test_risk = best_model.predict(X_test)

    train_c_index = concordance_index_censored(
        y_train_struct['event'], y_train_struct['time'], rsf_train_risk
    )[0]
    test_c_index = concordance_index_censored(
        y_test_struct['event'], y_test_struct['time'], rsf_test_risk
    )[0]

    rsf_results = {'Model': 'Random Survival Forest',
                   'Train_C_index': f"{train_c_index:.4f}",
                   'Test_C_index': f"{test_c_index:.4f}",
                   'Gap': f"{train_c_index - test_c_index:.4f}"}

    for time_point in evaluation_times:
        train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
            best_model, X_train, T_train, E_train, time_point
        )
        test_auc, test_ci_lower, test_ci_upper = time_dependent_auc(
            best_model, X_test, T_test, E_test, time_point
        )
        rsf_results[f'Train_AUC_{time_point}m'] = f"{train_auc:.3f} ({train_ci_lower:.3f}-{train_ci_upper:.3f})" if not np.isnan(train_auc) else "NaN"
        rsf_results[f'Test_AUC_{time_point}m'] = f"{test_auc:.3f} ({test_ci_lower:.3f}-{test_ci_upper:.3f})" if not np.isnan(test_auc) else "NaN"

    rsf_results_df = pd.DataFrame([rsf_results])
    rsf_results_path = os.path.join(output_dir, "4_rsf_model_results.csv")
    rsf_results_df.to_csv(rsf_results_path, index=False)
except Exception as e:
    rsf_results = {'Model': 'Random Survival Forest', 'Error': str(e)}


try:
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    except ImportError:
        raise

    param_grid_gbs = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.005],
        'max_depth': [3],
        'min_samples_split': [10, 20],
        'subsample': [0.7, 0.8],
        'random_state': [42]
    }

    gbs = GradientBoostingSurvivalAnalysis(random_state=42)

    grid_search_gbs = GridSearchCV(
        gbs, param_grid_gbs, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
    )

    grid_search_gbs.fit(X_train, y_train_struct)

    gbs_best = grid_search_gbs.best_estimator_
    gbs_best.fit(X_train, y_train_struct)

    gbs_model_path = os.path.join(output_dir, "4_gbs_model.pkl")
    joblib.dump(gbs_best, gbs_model_path)

    gbs_train_risk = gbs_best.predict(X_train)
    gbs_test_risk = gbs_best.predict(X_test)

    train_c_index = concordance_index_censored(
        y_train_struct['event'], y_train_struct['time'], gbs_train_risk
    )[0]
    test_c_index = concordance_index_censored(
        y_test_struct['event'], y_test_struct['time'], gbs_test_risk
    )[0]

    gbs_results = {'Model': 'Gradient Boosting Survival',
                   'Train_C_index': f"{train_c_index:.4f}",
                   'Test_C_index': f"{test_c_index:.4f}",
                   'Gap': f"{train_c_index - test_c_index:.4f}"}

    for time_point in evaluation_times:
        train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
            gbs_best, X_train, T_train, E_train, time_point
        )
        test_auc, test_ci_lower, test_ci_upper = time_dependent_auc(
            gbs_best, X_test, T_test, E_test, time_point
        )
        gbs_results[f'Train_AUC_{time_point}m'] = f"{train_auc:.3f} ({train_ci_lower:.3f}-{train_ci_upper:.3f})" if not np.isnan(train_auc) else "NaN"
        gbs_results[f'Test_AUC_{time_point}m'] = f"{test_auc:.3f} ({test_ci_lower:.3f}-{test_ci_upper:.3f})" if not np.isnan(test_auc) else "NaN"

    gbs_results_df = pd.DataFrame([gbs_results])
    gbs_results_path = os.path.join(output_dir, "4_gbs_model_results.csv")
    gbs_results_df.to_csv(gbs_results_path, index=False)
except Exception as e:
    gbs_results = {'Model': 'Gradient Boosting Survival', 'Error': str(e)}


try:
    try:
        from sksurv.svm import FastSurvivalSVM
        from sksurv.metrics import concordance_index_censored
        use_sksurv_svm = True
    except ImportError:
        use_sksurv_svm = False

    if use_sksurv_svm:
        simplified_params = [
            {'alpha': 1.0, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 5.0, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 10.0, 'rank_ratio': 0.8, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 0.5, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
        ]

        best_score = -np.inf
        best_params = None
        best_model = None

        for params in simplified_params:
            try:
                ssvm_model = FastSurvivalSVM(**params)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                fold_scores = []
                for train_idx, val_idx in kf.split(X_train):
                    X_train_fold = X_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_train_fold = y_train_struct[train_idx]
                    y_val_fold = y_train_struct[val_idx]
                    ssvm_model.fit(X_train_fold, y_train_fold)
                    predictions = ssvm_model.predict(X_val_fold)
                    c_index = concordance_index_censored(
                        y_val_fold['event'], y_val_fold['time'], predictions
                    )[0]
                    fold_scores.append(c_index)
                avg_score = np.mean(fold_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                best_model = FastSurvivalSVM(**params)
                best_model.fit(X_train, y_train_struct)
            except Exception as e:
                pass

        if best_model is None:
            best_model = FastSurvivalSVM(random_state=42)
            best_model.fit(X_train, y_train_struct)
            best_params = {'random_state': 42}

        ssvm_model_path = os.path.join(output_dir, "4_ssvm_model.pkl")
        joblib.dump(best_model, ssvm_model_path)

        ssvm_train_risk = best_model.predict(X_train)
        ssvm_test_risk = best_model.predict(X_test)

        train_c_index = concordance_index_censored(
            y_train_struct['event'], y_train_struct['time'], ssvm_train_risk
        )[0]
        test_c_index = concordance_index_censored(
            y_test_struct['event'], y_test_struct['time'], ssvm_test_risk
        )[0]

        ssvm_results = {'Model': 'Survival SVM',
                        'Train_C_index': f"{train_c_index:.4f}",
                        'Test_C_index': f"{test_c_index:.4f}",
                        'Gap': f"{train_c_index - test_c_index:.4f}"}

        for time_point in evaluation_times:
            train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
                best_model, X_train, T_train, E_train, time_point
            )
            test_auc, test_ci_lower, test_ci_upper = time_dependent_auc(
                best_model, X_test, T_test, E_test, time_point
            )
            ssvm_results[f'Train_AUC_{time_point}m'] = f"{train_auc:.3f} ({train_ci_lower:.3f}-{train_ci_upper:.3f})" if not np.isnan(train_auc) else "NaN"
            ssvm_results[f'Test_AUC_{time_point}m'] = f"{test_auc:.3f} ({test_ci_lower:.3f}-{test_ci_upper:.3f})" if not np.isnan(test_auc) else "NaN"

        ssvm_results_df = pd.DataFrame([ssvm_results])
        ssvm_results_path = os.path.join(output_dir, "4_ssvm_model_results.csv")
        ssvm_results_df.to_csv(ssvm_results_path, index=False)
    else:
        def survival_to_classification(T, E, time_point):
            y_binary = np.zeros(len(T))
            y_binary[(T <= time_point) & (E == 1)] = 1
            y_binary[T > time_point] = 0
            mask = (T <= time_point) & (E == 0)
            y_binary = np.delete(y_binary, mask)
            return y_binary

        time_point = 36
        y_train_binary = survival_to_classification(T_train, E_train, time_point)
        mask_train = ~((T_train <= time_point) & (E_train == 0))
        X_train_binary = X_train[mask_train].copy()

        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        param_grid_svc = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'random_state': [42]
        }

        svc = SVC(probability=True)
        grid_search_svc = GridSearchCV(
            svc, param_grid_svc, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search_svc.fit(X_train_binary, y_train_binary)

        svc_best = grid_search_svc.best_estimator_

        ssvm_model_path = os.path.join(output_dir, "4_ssvm_svc_model.pkl")
        joblib.dump(svc_best, ssvm_model_path)

        y_test_binary = survival_to_classification(T_test, E_test, time_point)
        mask_test = ~((T_test <= time_point) & (E_test == 0))
        X_test_binary = X_test[mask_test].copy()

        y_train_pred = svc_best.predict_proba(X_train_binary)[:, 1]
        y_test_pred = svc_best.predict_proba(X_test_binary)[:, 1]

        train_auc = roc_auc_score(y_train_binary, y_train_pred)
        test_auc = roc_auc_score(y_test_binary, y_test_pred)

        ssvm_results = {'Model': 'Survival SVM (Binary at 36m)'}
        ssvm_results['Train_AUC_36m'] = f"{train_auc:.3f}"
        ssvm_results['Test_AUC_36m'] = f"{test_auc:.3f}"
        ssvm_results['Train_C_index'] = "N/A"
        ssvm_results['Test_C_index'] = "N/A"

        ssvm_results_df = pd.DataFrame([ssvm_results])
        ssvm_results_path = os.path.join(output_dir, "4_ssvm_model_results.csv")
        ssvm_results_df.to_csv(ssvm_results_path, index=False)
except Exception as e:
    ssvm_results = {'Model': 'Survival SVM', 'Error': str(e)}


all_results = []

model_files = {
    'Gradient Boosting Survival': '4_gbs_model_results.csv',
    'Random Survival Forest': '4_rsf_model_results.csv',
    'Survival SVM': '4_ssvm_model_results.csv'
}

for model_name, file_name in model_files.items():
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(file_path):
        try:
            results_df = pd.read_csv(file_path)
            all_results.append(results_df)
        except:
            pass

if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(output_dir, "4_model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)


def plot_simple_roc_curve(model, X, T, E, time_point, model_name, ax, set_name):
    try:
        if hasattr(model, 'predict_partial_hazard'):
            risk_scores = model.predict_partial_hazard(X)
            if hasattr(risk_scores, 'values'):
                risk_scores = risk_scores.values
        elif hasattr(model, 'predict'):
            risk_scores = model.predict(X)
        else:
            return

        y_true_binary = np.zeros(len(X))
        y_true_binary[(T <= time_point) & (E == 1)] = 1
        y_true_binary[(T > time_point)] = 0

        valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
        y_true_binary = y_true_binary[valid_mask]
        y_score_binary = risk_scores[valid_mask]

        if len(np.unique(y_true_binary)) < 2:
            return

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    except Exception as e:
        pass


models_to_plot = {}
model_paths = {
    'GBS': '4_gbs_model.pkl',
    'RSF': '4_rsf_model.pkl',
    'SSVM': '4_ssvm_model.pkl'
}

for model_name, file_name in model_paths.items():
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(file_path):
        try:
            model = joblib.load(file_path)
            models_to_plot[model_name] = model
        except:
            pass

if models_to_plot:
    for time_point in evaluation_times:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for model_name, model in models_to_plot.items():
            plot_simple_roc_curve(model, X_train, T_train, E_train, time_point, model_name, ax1, "Train")

        ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('假阳性率 (1-特异性)')
        ax1.set_ylabel('真阳性率 (敏感性)')
        ax1.set_title(f'{time_point // 12}年({time_point}月)时间依赖ROC曲线 - 训练集')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        for model_name, model in models_to_plot.items():
            plot_simple_roc_curve(model, X_test, T_test, E_test, time_point, model_name, ax2, "Test")

        ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假阳性率 (1-特异性)')
        ax2.set_ylabel('真阳性率 (敏感性)')
        ax2.set_title(f'{time_point // 12}年({time_point}月)时间依赖ROC曲线 - 测试集')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"4_time_dependent_roc_{time_point}m.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

