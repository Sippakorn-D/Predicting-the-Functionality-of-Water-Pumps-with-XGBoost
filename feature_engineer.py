import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create new features + group rare funder/installer into 'other' (learned from train only)"""
    def __init__(self, top_n=30):
        self.top_n = top_n

    def _base(self, d):
        d = d.copy()
        for c in ['funder', 'installer']:
            d[c] = d[c].fillna('unknown').astype(str).str.lower().str.strip()
        d['age_missing'] = d['age'].isna().astype(int)
        d['gps_zero'] = (d['gps_height'] <= 0).astype(int)
        d['pop_zero'] = (d['population'] == 0).astype(int)
        d['tsh_pos'] = (d['amount_tsh'] > 0).astype(int)
        d['log_pop'] = np.log1p(d['population'])
        d['log_tsh'] = np.log1p(d['amount_tsh'])
        d['same_funder_installer'] = (d['funder'] == d['installer']).astype(int)
        d['public_meeting'] = d['public_meeting'].astype(str).replace({'nan': 'NA'})
        d['permit'] = d['permit'].astype(str).replace({'nan': 'NA'})
        d['scheme_management'] = d['scheme_management'].fillna('NA')
        return d

    def fit(self, X, y=None):
        d = self._base(X)
        self.top_ = {c: set(d[c].value_counts().head(self.top_n).index) for c in ['funder', 'installer']}
        return self

    def transform(self, X):
        d = self._base(X)
        for c in ['funder', 'installer']:
            d[c] = d[c].where(d[c].isin(self.top_[c]), 'other')
        return d

numeric_features = ['amount_tsh', 'gps_height', 'population', 'age', 'log_pop', 'log_tsh',
                    'age_missing', 'gps_zero', 'pop_zero', 'tsh_pos', 'same_funder_installer']
cat_features = ['funder', 'installer', 'basin', 'region', 'public_meeting', 'scheme_management',
                'permit', 'extraction_type_class', 'management_group', 'payment_type',
                'quality_group', 'quantity_group', 'source_class', 'waterpoint_type_group']

def make_preprocessor(scale=False):
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        num_steps.append(('scaler', StandardScaler()))
    return Pipeline([
        ('fe', FeatureEngineer(top_n=30)),
        ('ct', ColumnTransformer([
            ('num', Pipeline(num_steps), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ])),
    ])

def predict_labels(bundle, X_new):
    """Predict labels from saved bundle (supports both ensemble and single model)"""
    proba = np.mean([m.predict_proba(X_new) for m in bundle['models'].values()], axis=0)
    return bundle['label_encoder'].inverse_transform(proba.argmax(1))
