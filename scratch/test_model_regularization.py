import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Test importing feature_engineer
import sys
sys.path.append('.')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Make sure feature_engineer works
import feature_engineer
from feature_engineer import make_preprocessor, FeatureEngineer

class BalancedXGBClassifier(xgb.XGBClassifier):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is None:
            sample_weight = compute_sample_weight('balanced', y)
        return super().fit(X, y, sample_weight=sample_weight, **fit_params)

DATA_PATH = r"Dataset\Cleaned\cleaned_data.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop('status_group', axis=1)
le = LabelEncoder()
y = le.fit_transform(df['status_group'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data split: Train {X_train.shape}, Test {X_test.shape}")

# Test 1: LightGBM with balanced class weights and tight regularization
print("\n--- Training Regularized Balanced LightGBM ---")
pipe_lgb = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
])
pipe_lgb.fit(X_train, y_train)

p_tr_lgb = pipe_lgb.predict(X_train)
p_te_lgb = pipe_lgb.predict(X_test)
prob_tr_lgb = pipe_lgb.predict_proba(X_train)
prob_te_lgb = pipe_lgb.predict_proba(X_test)

tr_f1 = f1_score(y_train, p_tr_lgb, average='micro')
te_f1 = f1_score(y_test, p_te_lgb, average='micro')
tr_auc = roc_auc_score(y_train, prob_tr_lgb, multi_class='ovr', average='macro')
te_auc = roc_auc_score(y_test, prob_te_lgb, multi_class='ovr', average='macro')

print(f"LGBM Train Micro F1: {tr_f1:.4f} | Test Micro F1: {te_f1:.4f} | Gap: {tr_f1 - te_f1:.4f}")
print(f"LGBM Train ROC-AUC:  {tr_auc:.4f} | Test ROC-AUC:  {te_auc:.4f} | Gap: {tr_auc - te_auc:.4f}")

# Test 2: Balanced XGBoost with tight regularization
print("\n--- Training Regularized Balanced XGBoost ---")
pipe_xgb = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', BalancedXGBClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_child_weight=15,
        gamma=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    ))
])
pipe_xgb.fit(X_train, y_train)

p_tr_xgb = pipe_xgb.predict(X_train)
p_te_xgb = pipe_xgb.predict(X_test)
prob_tr_xgb = pipe_xgb.predict_proba(X_train)
prob_te_xgb = pipe_xgb.predict_proba(X_test)

tr_f1_x = f1_score(y_train, p_tr_xgb, average='micro')
te_f1_x = f1_score(y_test, p_te_xgb, average='micro')
tr_auc_x = roc_auc_score(y_train, prob_tr_xgb, multi_class='ovr', average='macro')
te_auc_x = roc_auc_score(y_test, prob_te_xgb, multi_class='ovr', average='macro')

print(f"XGB Train Micro F1: {tr_f1_x:.4f} | Test Micro F1: {te_f1_x:.4f} | Gap: {tr_f1_x - te_f1_x:.4f}")
print(f"XGB Train ROC-AUC:  {tr_auc_x:.4f} | Test ROC-AUC:  {te_auc_x:.4f} | Gap: {tr_auc_x - te_auc_x:.4f}")

# Test 3: Random Forest with balanced subsample and bounded depth
print("\n--- Training Regularized Balanced Random Forest ---")
pipe_rf = Pipeline([
    ('prep', make_preprocessor()),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.3,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ))
])
pipe_rf.fit(X_train, y_train)

p_tr_rf = pipe_rf.predict(X_train)
p_te_rf = pipe_rf.predict(X_test)
prob_tr_rf = pipe_rf.predict_proba(X_train)
prob_te_rf = pipe_rf.predict_proba(X_test)

tr_f1_rf = f1_score(y_train, p_tr_rf, average='micro')
te_f1_rf = f1_score(y_test, p_te_rf, average='micro')
tr_auc_rf = roc_auc_score(y_train, prob_tr_rf, multi_class='ovr', average='macro')
te_auc_rf = roc_auc_score(y_test, prob_te_rf, multi_class='ovr', average='macro')

print(f"RF Train Micro F1:  {tr_f1_rf:.4f} | Test Micro F1:  {te_f1_rf:.4f} | Gap: {tr_f1_rf - te_f1_rf:.4f}")
print(f"RF Train ROC-AUC:   {tr_auc_rf:.4f} | Test ROC-AUC:   {te_auc_rf:.4f} | Gap: {tr_auc_rf - te_auc_rf:.4f}")
