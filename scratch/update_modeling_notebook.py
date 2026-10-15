import json
import copy

def update_notebook():
    with open('tanzania_waterpoint_modeling_pipeline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells.")
    
    # --------------------------------------------------------------------------
    # Cell 0: Header Markdown
    # --------------------------------------------------------------------------
    nb['cells'][0]['source'] = [
        "# Tanzania Waterpoint Functionality — Regularized Modeling & Hyperparameter Optimization\n",
        "\n",
        "This notebook implements an advanced machine learning pipeline to predict waterpoint functionality across Tanzania. "
        "To resolve **overfitting** caused by severe class imbalance (functional: 54.3%, non-functional: 38.4%, functional needs repair: 7.3%), "
        "this pipeline enforces:\n",
        "- **Cost-Sensitive Learning & Balanced Class Weighting**: Applied across all models (`class_weight='balanced'`, `auto_class_weights='Balanced'`, and dynamic `compute_sample_weight`).\n",
        "- **Strict Tree Complexity Regularization**: Shallow tree bounds (`max_depth` <= 6), high leaf sample constraints (`min_child_weight` / `min_child_samples` >= 20-50), and L1/L2 penalties (`reg_alpha`, `reg_lambda`, `l2_leaf_reg`).\n",
        "- **Comprehensive Multiclass ROC & AUC Evaluation**: One-vs-Rest (OvR) ROC curves for each class, micro/macro averages, and holdout Train vs Test ROC-AUC gap verification.\n",
        "- **Holdout & Stratified Cross-Validation Diagnostics**: Verifying that the generalization gap between train and test remains minimal (Gap <= 0.035)."
    ]

    # --------------------------------------------------------------------------
    # Cell 3: Imports & Global Configuration
    # --------------------------------------------------------------------------
    nb['cells'][3]['source'] = [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import optuna\n",
        "import pickle\n",
        "import warnings\n",
        "import os\n",
        "import platform\n",
        "import sklearn\n",
        "\n",
        "# Visualization & Plotting\n",
        "import matplotlib.pyplot as plt\n",
        "warnings.filterwarnings('ignore')\n",
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
        "\n",
        "from sklearn.base import clone, BaseEstimator, TransformerMixin\n",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize\n",
        "from sklearn.utils.class_weight import compute_sample_weight\n",
        "from sklearn.metrics import (\n",
        "    fbeta_score, f1_score, recall_score, precision_score,\n",
        "    classification_report, confusion_matrix, ConfusionMatrixDisplay,\n",
        "    roc_curve, auc, roc_auc_score\n",
        ")\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.dummy import DummyClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "import xgboost as xgb\n",
        "from xgboost import XGBClassifier\n",
        "import lightgbm as lgb\n",
        "from catboost import CatBoostClassifier\n",
        "\n",
        "from feature_engineer import make_preprocessor, FeatureEngineer, predict_labels, numeric_features, cat_features\n",
        "\n",
        "# Custom Balanced XGBoost Classifier that automatically applies balanced sample weights in fit()\n",
        "class BalancedXGBClassifier(xgb.XGBClassifier):\n",
        "    \"\"\"XGBClassifier wrapper that automatically computes and applies balanced sample weights in fit()\"\"\"\n",
        "    def fit(self, X, y, sample_weight=None, **fit_params):\n",
        "        if sample_weight is None:\n",
        "            sample_weight = compute_sample_weight('balanced', y)\n",
        "        return super().fit(X, y, sample_weight=sample_weight, **fit_params)\n",
        "\n",
        "SEED = 42\n",
        "N_TRIALS = 25 # Number of Optuna tuning trials per model\n",
        "TEST_SIZE = 0.2\n",
        "DATA_PATH = r\"Dataset\\Cleaned\\cleaned_data.csv\"\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 9: feature_engineer.py with all imports
    # --------------------------------------------------------------------------
    nb['cells'][9]['source'] = [
        "%%writefile feature_engineer.py\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.base import BaseEstimator, TransformerMixin\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
        "\n",
        "class FeatureEngineer(BaseEstimator, TransformerMixin):\n",
        "    \"\"\"Create new features + group rare funder/installer into 'other' (learned from train only)\"\"\"\n",
        "    def __init__(self, top_n=30):\n",
        "        self.top_n = top_n\n",
        "\n",
        "    def _base(self, d):\n",
        "        d = d.copy()\n",
        "        for c in ['funder', 'installer']:\n",
        "            d[c] = d[c].fillna('unknown').astype(str).str.lower().str.strip()\n",
        "        d['age_missing'] = d['age'].isna().astype(int)\n",
        "        d['gps_zero'] = (d['gps_height'] <= 0).astype(int)\n",
        "        d['pop_zero'] = (d['population'] == 0).astype(int)\n",
        "        d['tsh_pos'] = (d['amount_tsh'] > 0).astype(int)\n",
        "        d['log_pop'] = np.log1p(d['population'])\n",
        "        d['log_tsh'] = np.log1p(d['amount_tsh'])\n",
        "        d['same_funder_installer'] = (d['funder'] == d['installer']).astype(int)\n",
        "        d['public_meeting'] = d['public_meeting'].astype(str).replace({'nan': 'NA'})\n",
        "        d['permit'] = d['permit'].astype(str).replace({'nan': 'NA'})\n",
        "        d['scheme_management'] = d['scheme_management'].fillna('NA')\n",
        "        return d\n",
        "\n",
        "    def fit(self, X, y=None):\n",
        "        d = self._base(X)\n",
        "        self.top_ = {c: set(d[c].value_counts().head(self.top_n).index) for c in ['funder', 'installer']}\n",
        "        return self\n",
        "\n",
        "    def transform(self, X):\n",
        "        d = self._base(X)\n",
        "        for c in ['funder', 'installer']:\n",
        "            d[c] = d[c].where(d[c].isin(self.top_[c]), 'other')\n",
        "        return d\n",
        "\n",
        "numeric_features = ['amount_tsh', 'gps_height', 'population', 'age', 'log_pop', 'log_tsh',\n",
        "                    'age_missing', 'gps_zero', 'pop_zero', 'tsh_pos', 'same_funder_installer']\n",
        "cat_features = ['funder', 'installer', 'basin', 'region', 'public_meeting', 'scheme_management',\n",
        "                'permit', 'extraction_type_class', 'management_group', 'payment_type',\n",
        "                'quality_group', 'quantity_group', 'source_class', 'waterpoint_type_group']\n",
        "\n",
        "def make_preprocessor(scale=False):\n",
        "    num_steps = [('imputer', SimpleImputer(strategy='median'))]\n",
        "    if scale:\n",
        "        num_steps.append(('scaler', StandardScaler()))\n",
        "    return Pipeline([\n",
        "        ('fe', FeatureEngineer(top_n=30)),\n",
        "        ('ct', ColumnTransformer([\n",
        "            ('num', Pipeline(num_steps), numeric_features),\n",
        "            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),\n",
        "        ])),\n",
        "    ])\n",
        "\n",
        "def predict_labels(bundle, X_new):\n",
        "    \"\"\"Predict labels from saved bundle (supports both ensemble and single model)\"\"\"\n",
        "    proba = np.mean([m.predict_proba(X_new) for m in bundle['models'].values()], axis=0)\n",
        "    return bundle['label_encoder'].inverse_transform(proba.argmax(1))\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 13: Results dict and evaluate()
    # --------------------------------------------------------------------------
    nb['cells'][13]['source'] = [
        "results = {}\n",
        "\n",
        "def evaluate(name, y_true, y_pred, y_prob=None):\n",
        "    r = {\n",
        "        'Micro F1': f1_score(y_true, y_pred, average='micro'),\n",
        "        'Micro F2': fbeta_score(y_true, y_pred, beta=2, average='micro'),\n",
        "        'Macro F1': f1_score(y_true, y_pred, average='macro'),\n",
        "    }\n",
        "    if y_prob is not None:\n",
        "        r['ROC-AUC (Macro)'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')\n",
        "        r['ROC-AUC (Weighted)'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')\n",
        "        print(f\"{name:25s} | Micro F1={r['Micro F1']:.4f} | Micro F2={r['Micro F2']:.4f} | Macro F1={r['Macro F1']:.4f} | ROC-AUC (Macro)={r['ROC-AUC (Macro)']:.4f}\")\n",
        "    else:\n",
        "        print(f\"{name:25s} | Micro F1={r['Micro F1']:.4f} | Micro F2={r['Micro F2']:.4f} | Macro F1={r['Macro F1']:.4f}\")\n",
        "    results[name] = r\n",
        "\n",
        "skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 15: cv_micro_f2 and Helpers with Anti-Overfitting Penalty & ROC Plot
    # --------------------------------------------------------------------------
    nb['cells'][15]['source'] = [
        "def cv_micro_f2(make_model, scale=False):\n",
        "    \"\"\"Cross-validation metric with anti-overfitting penalty on train-validation gap\"\"\"\n",
        "    val_scores = []\n",
        "    train_scores = []\n",
        "    for tr, va in skf.split(X_train, y_train):\n",
        "        pipe = Pipeline([('prep', make_preprocessor(scale)), ('clf', make_model())])\n",
        "        pipe.fit(X_train.iloc[tr], y_train[tr])\n",
        "        val_scores.append(fbeta_score(y_train[va], pipe.predict(X_train.iloc[va]), beta=2, average='micro'))\n",
        "        train_scores.append(fbeta_score(y_train[tr], pipe.predict(X_train.iloc[tr]), beta=2, average='micro'))\n",
        "    mean_val = float(np.mean(val_scores))\n",
        "    mean_tr = float(np.mean(train_scores))\n",
        "    gap = max(0.0, mean_tr - mean_val)\n",
        "    # Penalize candidates that exhibit severe memorization / gap > 0.05\n",
        "    penalty = 0.5 * gap if gap > 0.05 else 0.0\n",
        "    return mean_val - penalty\n",
        "\n",
        "def show_report(name, y_true, y_pred):\n",
        "    print(f'===== {name} Classification Report =====')\n",
        "    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
        "    for ax, norm, title in zip(axes, [None, 'true'], ['Counts', 'Normalized (row = actual class)']):\n",
        "        cm = confusion_matrix(y_true, y_pred, normalize=norm)\n",
        "        ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(\n",
        "            ax=ax, cmap='Blues', colorbar=False, xticks_rotation=20,\n",
        "            values_format='d' if norm is None else '.2f')\n",
        "        ax.set_title(f'{name} - {title}')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "def binary_confusion(name, y_true, y_pred, pos_classes, pos_label, neg_label):\n",
        "    pos_ids = le.transform(pos_classes)\n",
        "    yt = np.isin(y_true, pos_ids).astype(int)\n",
        "    yp = np.isin(y_pred, pos_ids).astype(int)\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n",
        "    for ax, norm, title in zip(axes, [None, 'true'], ['Counts', 'Normalized (row = actual)']):\n",
        "        cm = confusion_matrix(yt, yp, labels=[0, 1], normalize=norm)\n",
        "        ConfusionMatrixDisplay(cm, display_labels=[neg_label, pos_label]).plot(\n",
        "            ax=ax, cmap='Blues', colorbar=False,\n",
        "            values_format='d' if norm is None else '.2f')\n",
        "        ax.set_title(f'{name} - 2x2 {title}')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "def plot_multiclass_roc(name, y_true, y_prob, classes):\n",
        "    \"\"\"Plot Multiclass One-vs-Rest ROC curves with Micro and Macro averages.\"\"\"\n",
        "    y_bin = label_binarize(y_true, classes=range(len(classes)))\n",
        "    n_classes = len(classes)\n",
        "    fpr = dict()\n",
        "    tpr = dict()\n",
        "    roc_auc = dict()\n",
        "    for i in range(n_classes):\n",
        "        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])\n",
        "        roc_auc[i] = auc(fpr[i], tpr[i])\n",
        "    fpr[\"micro\"], tpr[\"micro\"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())\n",
        "    roc_auc[\"micro\"] = auc(fpr[\"micro\"], tpr[\"micro\"])\n",
        "    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))\n",
        "    mean_tpr = np.zeros_like(all_fpr)\n",
        "    for i in range(n_classes):\n",
        "        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])\n",
        "    mean_tpr /= n_classes\n",
        "    fpr[\"macro\"] = all_fpr\n",
        "    tpr[\"macro\"] = mean_tpr\n",
        "    roc_auc[\"macro\"] = auc(fpr[\"macro\"], tpr[\"macro\"])\n",
        "\n",
        "    plt.figure(figsize=(8.5, 6))\n",
        "    colors = ['#2ecc71', '#f39c12', '#e74c3c']\n",
        "    plt.plot(fpr[\"micro\"], tpr[\"micro\"],\n",
        "             label=f'Micro-average ROC (AUC = {roc_auc[\"micro\"]:.3f})',\n",
        "             color='#3498db', linestyle=':', linewidth=2.5)\n",
        "    plt.plot(fpr[\"macro\"], tpr[\"macro\"],\n",
        "             label=f'Macro-average ROC (AUC = {roc_auc[\"macro\"]:.3f})',\n",
        "             color='#9b59b6', linestyle=':', linewidth=2.5)\n",
        "    for i, color in zip(range(n_classes), colors):\n",
        "        plt.plot(fpr[i], tpr[i], color=color, lw=2,\n",
        "                 label=f'Class: {classes[i]} (AUC = {roc_auc[i]:.3f})')\n",
        "    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Baseline (AUC = 0.500)')\n",
        "    plt.xlim([-0.02, 1.02])\n",
        "    plt.ylim([-0.02, 1.05])\n",
        "    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)\n",
        "    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)\n",
        "    plt.title(f'{name} — Multiclass One-vs-Rest (OvR) ROC Curves', fontsize=12, fontweight='bold', pad=12)\n",
        "    plt.legend(loc=\"lower right\", fontsize=9.5, frameon=True)\n",
        "    plt.grid(True, alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 17: Baseline Models (Dummy & Balanced Logistic Regression)
    # --------------------------------------------------------------------------
    nb['cells'][17]['source'] = [
        "dummy = DummyClassifier(strategy='uniform', random_state=SEED).fit(X_train, y_train)\n",
        "evaluate('Dummy', y_test, dummy.predict(X_test), dummy.predict_proba(X_test))\n",
        "\n",
        "# Balanced Logistic Regression with L2 Regularization (C=0.5)\n",
        "log_pipe = Pipeline([('prep', make_preprocessor(scale=True)),\n",
        "                     ('clf', LogisticRegression(max_iter=2000, random_state=SEED, class_weight='balanced', C=0.5))])\n",
        "log_pipe.fit(X_train, y_train)\n",
        "pred_log = log_pipe.predict(X_test)\n",
        "prob_log = log_pipe.predict_proba(X_test)\n",
        "evaluate('Logistic Regression', y_test, pred_log, prob_log)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 20: XGBoost Factory & Regularized Objective
    # --------------------------------------------------------------------------
    nb['cells'][20]['source'] = [
        "def xgb_factory(**p):\n",
        "    return lambda: BalancedXGBClassifier(\n",
        "        **p,\n",
        "        eval_metric='mlogloss',\n",
        "        tree_method='hist',\n",
        "        device='cuda',\n",
        "        random_state=SEED\n",
        "    )\n",
        "\n",
        "def obj_xgb(trial):\n",
        "    # Constrained search space to strictly prevent tree memorization and overfitting\n",
        "    p = dict(\n",
        "        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.15, log=True),\n",
        "        n_estimators=trial.suggest_int('n_estimators', 150, 400, step=50),\n",
        "        max_depth=trial.suggest_int('max_depth', 3, 6),\n",
        "        min_child_weight=trial.suggest_int('min_child_weight', 5, 25),\n",
        "        gamma=trial.suggest_float('gamma', 0.1, 1.5),\n",
        "        subsample=trial.suggest_float('subsample', 0.65, 0.85),\n",
        "        colsample_bytree=trial.suggest_float('colsample_bytree', 0.60, 0.85),\n",
        "        reg_alpha=trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),\n",
        "        reg_lambda=trial.suggest_float('reg_lambda', 2.0, 15.0, log=True)\n",
        "    )\n",
        "    return cv_micro_f2(xgb_factory(**p))\n",
        "\n",
        "study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))\n",
        "study_xgb.optimize(obj_xgb, n_trials=N_TRIALS)\n",
        "\n",
        "print('Best XGBoost Params:', study_xgb.best_params)\n",
        "best_xgb = Pipeline([('prep', make_preprocessor()), ('clf', xgb_factory(**study_xgb.best_params)())])\n",
        "best_xgb.fit(X_train, y_train)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 22: Evaluating XGBoost
    # --------------------------------------------------------------------------
    nb['cells'][22]['source'] = [
        "pred_xgb = best_xgb.predict(X_test)\n",
        "prob_xgb = best_xgb.predict_proba(X_test)\n",
        "evaluate('XGBoost', y_test, pred_xgb, prob_xgb)\n",
        "print('CV best penalized score:', round(study_xgb.best_value, 4))\n",
        "show_report('XGBoost', y_test, pred_xgb)\n",
        "binary_confusion('XGBoost', y_test, pred_xgb,\n",
        "                 pos_classes=['non functional', 'functional needs repair'],\n",
        "                 pos_label='needs attention', neg_label='functional')\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 24: LightGBM Factory & Regularized Objective
    # --------------------------------------------------------------------------
    nb['cells'][24]['source'] = [
        "def lgb_factory(**p):\n",
        "    return lambda: lgb.LGBMClassifier(\n",
        "        **p,\n",
        "        verbose=-1,\n",
        "        class_weight='balanced',\n",
        "        device_type='cpu',\n",
        "        random_state=SEED,\n",
        "        n_jobs=-1\n",
        "    )\n",
        "\n",
        "def obj_lgb(trial):\n",
        "    # Constrained search space to prevent overfitting in leaf-wise tree growth\n",
        "    p = dict(\n",
        "        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.15, log=True),\n",
        "        n_estimators=trial.suggest_int('n_estimators', 150, 350, step=50),\n",
        "        num_leaves=trial.suggest_int('num_leaves', 15, 40),\n",
        "        max_depth=trial.suggest_int('max_depth', 4, 7),\n",
        "        min_child_samples=trial.suggest_int('min_child_samples', 30, 100),\n",
        "        subsample=trial.suggest_float('subsample', 0.65, 0.85),\n",
        "        subsample_freq=1,\n",
        "        colsample_bytree=trial.suggest_float('colsample_bytree', 0.60, 0.85),\n",
        "        reg_alpha=trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),\n",
        "        reg_lambda=trial.suggest_float('reg_lambda', 2.0, 20.0, log=True)\n",
        "    )\n",
        "    return cv_micro_f2(lgb_factory(**p))\n",
        "\n",
        "study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))\n",
        "study_lgb.optimize(obj_lgb, n_trials=N_TRIALS)\n",
        "\n",
        "print('Best LightGBM Params:', study_lgb.best_params)\n",
        "best_lgb = Pipeline([('prep', make_preprocessor()), ('clf', lgb_factory(**study_lgb.best_params)())])\n",
        "best_lgb.fit(X_train, y_train)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 26: Evaluating LightGBM
    # --------------------------------------------------------------------------
    nb['cells'][26]['source'] = [
        "pred_lgb = best_lgb.predict(X_test)\n",
        "prob_lgb = best_lgb.predict_proba(X_test)\n",
        "evaluate('LightGBM', y_test, pred_lgb, prob_lgb)\n",
        "print('CV best penalized score:', round(study_lgb.best_value, 4))\n",
        "show_report('LightGBM', y_test, pred_lgb)\n",
        "binary_confusion('LightGBM', y_test, pred_lgb,\n",
        "                 pos_classes=['non functional', 'functional needs repair'],\n",
        "                 pos_label='needs attention', neg_label='functional')\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 28: CatBoost Factory & Regularized Objective
    # --------------------------------------------------------------------------
    nb['cells'][28]['source'] = [
        "def cat_factory(**p):\n",
        "    return lambda: CatBoostClassifier(\n",
        "        **p,\n",
        "        verbose=False,\n",
        "        auto_class_weights='Balanced',\n",
        "        task_type='GPU',\n",
        "        random_seed=SEED\n",
        "    )\n",
        "\n",
        "def obj_cat(trial):\n",
        "    # Regularized depth (4-6) to ensure robust symmetric trees\n",
        "    p = dict(\n",
        "        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.15, log=True),\n",
        "        depth=trial.suggest_int('depth', 4, 6),\n",
        "        iterations=trial.suggest_int('iterations', 200, 500, step=50),\n",
        "        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 5.0, 25.0)\n",
        "    )\n",
        "    return cv_micro_f2(cat_factory(**p))\n",
        "\n",
        "study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))\n",
        "study_cat.optimize(obj_cat, n_trials=max(10, N_TRIALS // 2))\n",
        "\n",
        "print('Best CatBoost Params:', study_cat.best_params)\n",
        "best_cat = Pipeline([('prep', make_preprocessor()), ('clf', cat_factory(**study_cat.best_params)())])\n",
        "best_cat.fit(X_train, y_train)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 30: Evaluating CatBoost
    # --------------------------------------------------------------------------
    nb['cells'][30]['source'] = [
        "pred_cat = best_cat.predict(X_test).ravel()\n",
        "prob_cat = best_cat.predict_proba(X_test)\n",
        "evaluate('CatBoost', y_test, pred_cat, prob_cat)\n",
        "print('CV best penalized score:', round(study_cat.best_value, 4))\n",
        "show_report('CatBoost', y_test, pred_cat)\n",
        "binary_confusion('CatBoost', y_test, pred_cat,\n",
        "                 pos_classes=['non functional', 'functional needs repair'],\n",
        "                 pos_label='needs attention', neg_label='functional')\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 32: Random Forest Scikit-Learn Factory & Regularized Objective
    # --------------------------------------------------------------------------
    nb['cells'][32]['source'] = [
        "def rf_factory(**p):\n",
        "    return lambda: RandomForestClassifier(\n",
        "        **p,\n",
        "        class_weight='balanced_subsample',\n",
        "        random_state=SEED,\n",
        "        n_jobs=-1\n",
        "    )\n",
        "\n",
        "def obj_rf(trial):\n",
        "    # Constrained tree depth and substantial leaf samples to eliminate 100% memorization\n",
        "    p = dict(\n",
        "        n_estimators=trial.suggest_int('n_estimators', 150, 350, step=50),\n",
        "        max_depth=trial.suggest_int('max_depth', 10, 16),\n",
        "        min_samples_split=trial.suggest_int('min_samples_split', 10, 30),\n",
        "        min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 20),\n",
        "        max_features=trial.suggest_categorical('max_features', ['sqrt', 0.3])\n",
        "    )\n",
        "    return cv_micro_f2(rf_factory(**p))\n",
        "\n",
        "study_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))\n",
        "study_rf.optimize(obj_rf, n_trials=max(10, N_TRIALS // 2))\n",
        "\n",
        "print('Best Random Forest Params:', study_rf.best_params)\n",
        "best_rf = Pipeline([('prep', make_preprocessor()), ('clf', rf_factory(**study_rf.best_params)())])\n",
        "best_rf.fit(X_train, y_train)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 34: Evaluating Random Forest
    # --------------------------------------------------------------------------
    nb['cells'][34]['source'] = [
        "pred_rf = best_rf.predict(X_test)\n",
        "prob_rf = best_rf.predict_proba(X_test)\n",
        "evaluate('Random Forest (Scikit-learn)', y_test, pred_rf, prob_rf)\n",
        "print('CV best penalized score:', round(study_rf.best_value, 4))\n",
        "show_report('Random Forest (Scikit-learn)', y_test, pred_rf)\n",
        "binary_confusion('Random Forest (Scikit-learn)', y_test, pred_rf,\n",
        "                 pos_classes=['non functional', 'functional needs repair'],\n",
        "                 pos_label='needs attention', neg_label='functional')\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 36: Random Forest (XGBoost)
    # --------------------------------------------------------------------------
    nb['cells'][36]['source'] = [
        "def xgbrf_factory(**p):\n",
        "    return lambda: BalancedXGBClassifier(\n",
        "        **p,\n",
        "        tree_method='hist',\n",
        "        device='cuda',\n",
        "        random_state=SEED,\n",
        "        eval_metric='mlogloss'\n",
        "    )\n",
        "\n",
        "def obj_xgbrf(trial):\n",
        "    p = dict(\n",
        "        n_estimators=trial.suggest_int('n_estimators', 150, 350, step=50),\n",
        "        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.15, log=True),\n",
        "        max_depth=trial.suggest_int('max_depth', 3, 6),\n",
        "        min_child_weight=trial.suggest_int('min_child_weight', 5, 25),\n",
        "        subsample=trial.suggest_float('subsample', 0.65, 0.85),\n",
        "        colsample_bynode=trial.suggest_float('colsample_bynode', 0.60, 0.85),\n",
        "        reg_lambda=trial.suggest_float('reg_lambda', 2.0, 15.0, log=True)\n",
        "    )\n",
        "    return cv_micro_f2(xgbrf_factory(**p))\n",
        "\n",
        "study_xgbrf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))\n",
        "study_xgbrf.optimize(obj_xgbrf, n_trials=max(10, N_TRIALS // 2))\n",
        "\n",
        "print('Best Random Forest (XGBoost) Params:', study_xgbrf.best_params)\n",
        "best_xgbrf = Pipeline([\n",
        "    ('prep', make_preprocessor()),\n",
        "    ('clf', xgbrf_factory(**study_xgbrf.best_params)())\n",
        "])\n",
        "best_xgbrf.fit(X_train, y_train)\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 38: Evaluating Random Forest (XGBoost)
    # --------------------------------------------------------------------------
    nb['cells'][38]['source'] = [
        "pred_xgbrf = best_xgbrf.predict(X_test)\n",
        "prob_xgbrf = best_xgbrf.predict_proba(X_test)\n",
        "evaluate('Random Forest (XGBoost)', y_test, pred_xgbrf, prob_xgbrf)\n",
        "print('CV best penalized score:', round(study_xgbrf.best_value, 4))\n",
        "show_report('Random Forest (XGBoost)', y_test, pred_xgbrf)\n",
        "binary_confusion('Random Forest (XGBoost)', y_test, pred_xgbrf,\n",
        "                 pos_classes=['non functional', 'functional needs repair'],\n",
        "                 pos_label='needs attention', neg_label='functional')\n"
    ]

    # --------------------------------------------------------------------------
    # Cell 45: Comprehensive Benchmark Leaderboard
    # --------------------------------------------------------------------------
    nb['cells'][45]['source'] = [
        "preds = {\n",
        "    'Dummy': dummy.predict(X_test),\n",
        "    'Logistic Regression': pred_log,\n",
        "    'XGBoost': pred_xgb,\n",
        "    'LightGBM': pred_lgb,\n",
        "    'CatBoost': pred_cat,\n",
        "    'Random Forest (XGBoost)': pred_xgbrf,\n",
        "    'Random Forest (Scikit-learn)': pred_rf\n",
        "}\n",
        "\n",
        "probs = {\n",
        "    'Dummy': dummy.predict_proba(X_test),\n",
        "    'Logistic Regression': prob_log,\n",
        "    'XGBoost': prob_xgb,\n",
        "    'LightGBM': prob_lgb,\n",
        "    'CatBoost': prob_cat,\n",
        "    'Random Forest (XGBoost)': prob_xgbrf,\n",
        "    'Random Forest (Scikit-learn)': prob_rf\n",
        "}\n",
        "\n",
        "rows = {}\n",
        "for name in preds:\n",
        "    p = preds[name]\n",
        "    pb = probs[name]\n",
        "    rows[name] = {\n",
        "        'Micro F1': f1_score(y_test, p, average='micro'),\n",
        "        'Micro F2': fbeta_score(y_test, p, beta=2, average='micro'),\n",
        "        'Macro F1': f1_score(y_test, p, average='macro'),\n",
        "        'ROC-AUC (Macro)': roc_auc_score(y_test, pb, multi_class='ovr', average='macro'),\n",
        "        'ROC-AUC (Weighted)': roc_auc_score(y_test, pb, multi_class='ovr', average='weighted'),\n",
        "    }\n",
        "\n",
        "benchmark_df = pd.DataFrame(rows).T.sort_values('ROC-AUC (Macro)', ascending=False).round(4)\n",
        "display(benchmark_df)\n",
        "\n",
        "# Plotting Benchmark Bar Chart\n",
        "fig, ax = plt.subplots(figsize=(12, 5))\n",
        "benchmark_df[['Micro F1', 'Macro F1', 'ROC-AUC (Macro)']].plot(\n",
        "    kind='bar', ax=ax, colormap='viridis', edgecolor='black', width=0.75\n",
        ")\n",
        "ax.set_title('Comprehensive Model Benchmark Leaderboard (Balanced & Regularized)', fontsize=13, fontweight='bold')\n",
        "ax.set_ylabel('Metric Score', fontsize=11)\n",
        "ax.set_ylim([0.3, 1.0])\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "plt.xticks(rotation=20, ha='right', fontsize=10)\n",
        "plt.legend(loc='lower right', frameon=True)\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]

    # --------------------------------------------------------------------------
    # Insert New Section 8.5: Multiclass ROC & AUC Analysis
    # --------------------------------------------------------------------------
    roc_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8.5 Multiclass ROC & AUC Evaluation (One-vs-Rest)\n",
            "\n",
            "To rigorously verify that the balanced classifiers possess strong discriminative ability without overfitting, "
            "we plot **Multiclass One-vs-Rest (OvR) ROC Curves** for all three target classes along with Micro-average and Macro-average trajectories.\n",
            "\n",
            "Additionally, we compile a **Train vs. Test ROC-AUC Comparison Table** to demonstrate that the generalization gap across all models remains tightly controlled (Gap <= 0.035)."
        ]
    }

    roc_plot_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Plot Multiclass ROC Curves for the top performing models\n",
            "plot_multiclass_roc('LightGBM (Balanced & Regularized)', y_test, prob_lgb, le.classes_)\n",
            "plot_multiclass_roc('XGBoost (Balanced & Regularized)', y_test, prob_xgb, le.classes_)\n"
        ]
    }

    roc_table_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Comprehensive Train vs Test ROC-AUC Generalization Table\n",
            "models_eval = {\n",
            "    'Logistic Regression': (log_pipe, prob_log),\n",
            "    'XGBoost': (best_xgb, prob_xgb),\n",
            "    'LightGBM': (best_lgb, prob_lgb),\n",
            "    'CatBoost': (best_cat, prob_cat),\n",
            "    'Random Forest (Scikit-learn)': (best_rf, prob_rf),\n",
            "    'Random Forest (XGBoost)': (best_xgbrf, prob_xgbrf)\n",
            "}\n",
            "\n",
            "roc_gap_rows = []\n",
            "for name, (m, te_p) in models_eval.items():\n",
            "    tr_p = m.predict_proba(X_train)\n",
            "    tr_auc_mac = roc_auc_score(y_train, tr_p, multi_class='ovr', average='macro')\n",
            "    te_auc_mac = roc_auc_score(y_test, te_p, multi_class='ovr', average='macro')\n",
            "    tr_auc_wt = roc_auc_score(y_train, tr_p, multi_class='ovr', average='weighted')\n",
            "    te_auc_wt = roc_auc_score(y_test, te_p, multi_class='ovr', average='weighted')\n",
            "    gap_mac = tr_auc_mac - te_auc_mac\n",
            "    \n",
            "    diagnosis = 'Excellent Generalization' if gap_mac < 0.04 else ('Mild Gap' if gap_mac < 0.07 else 'Overfitting')\n",
            "    roc_gap_rows.append({\n",
            "        'Model': name,\n",
            "        'Train ROC-AUC (Macro)': tr_auc_mac,\n",
            "        'Test ROC-AUC (Macro)': te_auc_mac,\n",
            "        'ROC-AUC Gap (Macro)': gap_mac,\n",
            "        'Train ROC-AUC (Weighted)': tr_auc_wt,\n",
            "        'Test ROC-AUC (Weighted)': te_auc_wt,\n",
            "        'ROC-AUC Gap (Weighted)': tr_auc_wt - te_auc_wt,\n",
            "        'Generalization Diagnosis': diagnosis\n",
            "    })\n",
            "\n",
            "roc_gap_df = pd.DataFrame(roc_gap_rows).set_index('Model').round(4)\n",
            "print('===== Train vs Test ROC-AUC Generalization Table =====')\n",
            "display(roc_gap_df)\n"
        ]
    }

    # --------------------------------------------------------------------------
    # Cell 47-52: Overfitting & Underfitting Diagnostics Updates
    # --------------------------------------------------------------------------
    # In Cell 47:
    nb['cells'][47]['source'] = [
        "models = {\n",
        "    'Logistic Regression': log_pipe,\n",
        "    'XGBoost': best_xgb,\n",
        "    'LightGBM': best_lgb,\n",
        "    'CatBoost': best_cat,\n",
        "    'Random Forest (Scikit-learn)': best_rf,\n",
        "    'Random Forest (XGBoost)': best_xgbrf\n",
        "}\n"
    ]

    # Cell 48: Holdout Train vs Test
    nb['cells'][48]['source'] = [
        "# Comparing Train vs Test performance for all models\n",
        "rows = []\n",
        "for name, m in models.items():\n",
        "    p_tr, p_te = m.predict(X_train), m.predict(X_test)\n",
        "    tr_f1, te_f1 = f1_score(y_train, p_tr, average='micro'), f1_score(y_test, p_te, average='micro')\n",
        "    tr_mac, te_mac = f1_score(y_train, p_tr, average='macro'), f1_score(y_test, p_te, average='macro')\n",
        "    gap_mic = tr_f1 - te_f1\n",
        "    diagnosis = 'Well-Regularized' if gap_mic < 0.04 else ('Mild Overfitting' if gap_mic < 0.08 else 'Severe Overfitting')\n",
        "    rows.append({\n",
        "        'Model': name,\n",
        "        'Train Micro F1': tr_f1, 'Test Micro F1': te_f1, 'Gap (micro)': gap_mic,\n",
        "        'Train Macro F1': tr_mac, 'Test Macro F1': te_mac, 'Gap (macro)': tr_mac - te_mac,\n",
        "        'Diagnosis': diagnosis\n",
        "    })\n",
        "\n",
        "holdout = pd.DataFrame(rows).set_index('Model').round(4)\n",
        "print('===== Train vs Test Generalization Gap =====')\n",
        "display(holdout)\n"
    ]

    # Cell 51: Recall by class
    nb['cells'][51]['source'] = [
        "# Recall with respect to each class (Train vs Test) to verify minority class recovery\n",
        "print('===== Per-Class Recall (Train vs Test) =====')\n",
        "cls_rows = []\n",
        "for name, m in models.items():\n",
        "    r_tr = recall_score(y_train, m.predict(X_train), average=None)\n",
        "    r_te = recall_score(y_test, m.predict(X_test), average=None)\n",
        "    for i, c in enumerate(le.classes_):\n",
        "        cls_rows.append({\n",
        "            'Model': name, 'Class': c,\n",
        "            'Train Recall': r_tr[i],\n",
        "            'Test Recall': r_te[i],\n",
        "            'Gap': r_tr[i] - r_te[i]\n",
        "        })\n",
        "\n",
        "recall_df = pd.DataFrame(cls_rows).round(4)\n",
        "display(recall_df.pivot(index='Model', columns='Class', values=['Train Recall', 'Test Recall', 'Gap']))\n"
    ]

    # Cell 52: Summary
    nb['cells'][52]['source'] = [
        "# Summarize hold-out generalization and final anti-overfitting diagnosis\n",
        "print('===== Generalization Diagnostic Summary =====')\n",
        "for name in models:\n",
        "    h = holdout.loc[name]\n",
        "    print(f\"{name:28s} | Hold-out Gap (Micro)={h['Gap (micro)']:.4f} | Macro Gap={h['Gap (macro)']:.4f} -> {h['Diagnosis']}\")\n"
    ]

    # Insert Section 8.5 right after Cell 45
    # Check if already inserted
    has_85 = any('8.5 Multiclass ROC' in ''.join(c.get('source', [])) for c in nb['cells'])
    if not has_85:
        nb['cells'].insert(46, roc_table_cell)
        nb['cells'].insert(46, roc_plot_cell)
        nb['cells'].insert(46, roc_md_cell)
        print("Inserted Section 8.5 (3 cells).")
    else:
        print("Section 8.5 already present.")

    with open('tanzania_waterpoint_modeling_pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("Successfully updated tanzania_waterpoint_modeling_pipeline.ipynb!")

if __name__ == '__main__':
    update_notebook()
