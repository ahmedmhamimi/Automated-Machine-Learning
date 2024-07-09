import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

class VamimiClassifier:
  def __init__(self):
    self.best_model = None
    self.best_params = ""
    self.best_accuracy = 0.0
    self.models_df = pd.DataFrame()

  def getBestEstimator(self, X_train, y_train):
    final = {}

    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=5000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga', 'lbfgs']
            }
        },
        'SVC': {
            'model': SVC(),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05]
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True]
            }
        },
        'XGBClassifier': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 6],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'GradientBoostingClassifier': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 6],
                'subsample': [0.8, 1.0]
            }
        }
    }

    for name, info in models.items():
        grid = GridSearchCV(estimator=info['model'], param_grid=info['params'], cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        best_score = grid.best_score_

        print(f"{round(best_score, 4)}  {name}: {best_params}")

        final[name] = {
            'params': best_params,
            'accuracy': best_score
        }

    sorted_models = sorted(final.items(), key=lambda item: item[1]['accuracy'], reverse=True)

    self.best_model = sorted_models[0][0]
    self.best_params = str(sorted_models[0][1]['params'])
    self.best_accuracy = sorted_models[0][1]['accuracy']

    top_models = {k: v for k, v in sorted_models[:5]}
    self.models_df = pd.DataFrame(top_models).T  # Transpose to have models as columns

"""
Usage

top_classifier = VamimiClassifier()
top_classifier.getBestEstimator(X_train, y_train)
print(top_classifier.best_model)
print(top_classifier.best_params)
print(top_classifier.best_accuracy)
print(top_classifier.models_df)
"""

class VamimiRegressor:
    def __init__(self):
        self.best_model = None
        self.best_params = ""
        self.best_score = 0.0
        self.models_df = pd.DataFrame()

    def getBestEstimator(self, X_train, y_train):
        final = {}

        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                }
            },
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1, 10],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.01, 0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'DecisionTreeRegressor': {
                'model': DecisionTreeRegressor(),
                'params': {
                    'criterion': ['absolute_error', 'squared_error'],
                    'splitter': ['best', 'random'],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [2, 4]
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': [100],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True]
                }
            },
            'XGBRegressor': {
                'model': XGBRegressor(objective='reg:squarederror'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 6],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'GradientBoostingRegressor': {
                'model': GradientBoostingRegressor(),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 6],
                    'subsample': [0.8, 1.0]
                }
            }
        }

        for name, info in models.items():
            grid = GridSearchCV(estimator=info['model'], param_grid=info['params'], cv=5, scoring='neg_mean_squared_error')
            grid.fit(X_train, y_train)

            best_params = grid.best_params_
            best_score = grid.best_score_

            print(f"{round(best_score, 4)}  {name}: {best_params}")

            final[name] = {
                'params': best_params,
                'mse': best_score
            }

        sorted_models = sorted(final.items(), key=lambda item: item[1]['mse'], reverse=True)

        self.best_model = sorted_models[0][0]
        self.best_params = str(sorted_models[0][1]['params'])
        self.best_score = sorted_models[0][1]['mse']

        top_models = {k: v for k, v in sorted_models[:5]}
        self.models_df = pd.DataFrame(top_models).T  # Transpose to have models as columns

"""
Usage

top_regressor = VamimiRegressor()
top_regressor.getBestEstimator(X_train, y_train)
print(top_regressor.best_model)
print(top_regressor.best_params)
print(top_regressor.best_score)
print(top_regressor.models_df)
"""
