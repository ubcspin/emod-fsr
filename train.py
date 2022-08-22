from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.rfes = {}


    
    def cv(self, n_splits=3, random_state=0, shuffle=True):
        return StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    
    def rfe(self, clf, cv=None, step=1, scoring='f1_macro'):
        if cv is None:
            cv = self.cv()
        return RFECV(estimator=clf, step=step, cv=cv, scoring = scoring)


    def pipeline(self, rfecv, gs_cv):
        return Pipeline([('feature_sele', rfecv), ('clf_cv',gs_cv)])
    
    def fit(self, X, y, cv=None, n_jobs=3, verbose=1, scoring=None, refit=False):
        if cv is None:
            cv = self.cv()
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            rfecv = self.rfe(model)
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            pipe = self.pipeline(rfecv, gs)
            pipe.fit(X,y)
            self.grid_searches[key] = gs 
            self.rfes[key] = rfecv

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv.get_n_splits()):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

models1 = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier(),
    # 'SVC': SVC()
}

params1 = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'XGBClassifier': { 'max_depth': (4, 6, 8), 'min_child_weight': (1, 5, 10) },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ],
}

import numpy as np 
import pandas as pd 

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


subject_results = []
for subject in loaded_data:
    X = subject['features'].loc[:, subject['features'].columns.str.contains('smooth_a6')]
    y_pos = subject['pos']
    y_angle = subject['angle']
    y_acc = subject['acc']
    y_cw = label_encoder.fit_transform(subject['calibrated_values'].astype('str'))
    
    helper_pos = EstimatorSelectionHelper(models1, params1)
    helper_pos.fit(X, y_pos, scoring='f1_macro', n_jobs=6)
    scores_pos = helper_pos.score_summary(sort_by='max_score')
    
    helper_angle = EstimatorSelectionHelper(models1, params1)
    helper_angle.fit(X, y_angle, scoring='f1_macro', n_jobs=6)
    scores_angle = helper_angle.score_summary(sort_by='max_score')
    
    helper_acc = EstimatorSelectionHelper(models1, params1)
    helper_acc.fit(X, y_acc, scoring='f1_macro', n_jobs=6)
    scores_acc = helper_acc.score_summary(sort_by='max_score')
    
    helper_cw = EstimatorSelectionHelper(models1, params1)
    helper_cw.fit(X, y_cw, scoring='f1_macro', n_jobs=6)
    scores_cw = helper_cw.score_summary(sort_by='max_score')
    
    subject_results.append({
        'helper_angle': helper_angle, 'scores_angle': scores_angle, 
        'helper_acc': helper_acc, 'scores_acc': scores_acc, 
        'helper_pos': helper_pos, 'scores_pos': scores_pos, 
        'helper_cw': helper_cw, 'scores_cw': scores_cw, 
        'pnum': subject['pnum']
    })