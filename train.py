import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import time
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import os
import time
import logging

def build_pipe(USE_PCA=False, POLY_ENABLED=False, seed=1234):
    steps = []
    if POLY_ENABLED:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("scaler", StandardScaler()))
    if USE_PCA:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    steps.append(("logreg", LogisticRegression(max_iter=4000, random_state=seed)))
    pipe = Pipeline(steps)
    param_grid = [
        {
            "logreg__solver": ["liblinear"],
            "logreg__penalty": ["l1", "l2"],
            "logreg__C": [0.01, 0.1, 1, 10],
        },
        {
            "logreg__solver": ["lbfgs"],
            "logreg__penalty": ["l2"],
            "logreg__C": [0.01, 0.1, 1, 10],
        },
    ]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",#roc_auc#accuracy
        n_jobs=4,        # use 4 cores in parallel
        cv=kfold,            # 5-fold cross-validation, more on this later
        refit=True,      # retrain the best model on the full training set
        return_train_score=True
    )
    return grid_search  # not fitted yet — caller will call `fit(X, y)`
def predict_and_submit(test_df, features, pipe, prefix=""):
    os.makedirs("output", exist_ok=True)
    # Make predictions on the real test data
    X_test = test_df[features]
    print("Generating predictions on the test set...")
    test_predictions = pipe.predict(X_test)
    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        "battle_id": test_df["battle_id"],
        "player_won": test_predictions
    })
    submission_df.to_csv(f"output/{prefix}_submission.csv", index=False)
    print("\nsubmission.csv file created successfully!")
def train_regularization(X, y, USE_PCA=False, POLY_ENABLED=False, seed=1234):
    grid_search = build_pipe(USE_PCA=USE_PCA, POLY_ENABLED=POLY_ENABLED, seed=seed)
    grid_search.fit(X, y)
    print(f"Best params: {grid_search.best_params_}")
    mean_score = grid_search.best_score_
    std_score = grid_search.cv_results_["std_test_score"][grid_search.best_index_]
    print(f"Best CV mean: {mean_score:.4f} ± {std_score:.4f}")
    best_model = grid_search.best_estimator_
    return best_model
def get_power_set_non_empty_as_list(array):
  n = len(array)
  combinations_iterators = (
      itertools.combinations(array, k) for k in range(1, n + 1)
  )
  non_empty_subsets_tuples = itertools.chain.from_iterable(combinations_iterators)
  non_empty_subsets_lists = [
      list(subset_tuple) for subset_tuple in non_empty_subsets_tuples
  ]
  return non_empty_subsets_lists
def select_top_features(model, X, y, k=50, scoring="roc_auc"):
    print(f"\nCalcolo permutation importances (Top {k})")
    t0 = time.time()
    model.fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=10,
        random_state=89351,
        n_jobs=-1
    )
    importances = result.importances_mean
    feature_names = np.array(X.columns)
    idx_sorted = np.argsort(importances)[::-1]
    top_features = feature_names[idx_sorted][:k]
    top_scores = importances[idx_sorted][:k]
    importance_df = pd.DataFrame({
        "feature": top_features,
        "importance": top_scores
    })
    #print(importance_df.head(20))
    print(f"[Permutation Importance completato in {time.time()-t0:.2f}s]")
    return list(top_features), importance_df
def build_voting_model():
    #Logistic Regression (regolarizzata)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            max_iter=1500,
            random_state=1234
        ))
    ])
    #Random Forest (meno overfitting)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=4,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=1234
    )
    #XGBoost (modello principale)
    xgb = XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=1234,
        n_jobs=-1
    )
    #Voting Ensemble
    model = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf", rf),
            ("lr", lr)
        ],
        voting="soft",
        weights=[4, 1, 3],   #XGB più influente
        n_jobs=-1
    )
    return model
def train_with_feature_selection(X, y, k=50):
    print("\nFASE 1: Training iniziale con tutte le feature")
    base_model = build_voting_model()
    t0 = time.time()
    base_model.fit(X, y)
    print(f"Modello iniziale addestrato in {time.time()-t0:.2f}s")
    #Feature Selection
    selected_features, importance_df = select_top_features(base_model, X, y, k=k)
    print(f"\nTop-{k} feature selezionate:")
    print(selected_features)
    print("\nFASE 2: Retraining con feature selezionate")
    final_model = build_voting_model()
    X_sel = X[selected_features]
    t1 = time.time()
    final_model.fit(X_sel, y)
    print(f"Retraining completato in {time.time()-t1:.2f}s\n")
    #Performance
    y_pred = final_model.predict(X_sel)
    y_proba = final_model.predict_proba(X_sel)[:, 1]
    acc_cv = cross_val_score(final_model, X_sel, y, cv=5, scoring="accuracy")
    auc_cv = cross_val_score(final_model, X_sel, y, cv=5, scoring="roc_auc")
    print("\nRISULTATI FINALI")
    print(f"Training Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Training AUC: {roc_auc_score(y, y_proba):.4f}")
    print(f"CV Accuracy: {acc_cv.mean():.4f} ± {acc_cv.std():.4f}")
    print(f"CV AUC: {auc_cv.mean():.4f} ± {auc_cv.std():.4f}")
    return final_model, selected_features, importance_df
def build_stacking_model():
    #Definizione dei Modelli Base
    # Logistic Regression (regolarizzata)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            max_iter=1500,
            random_state=1234
        ))
    ])
    
    # Random Forest (meno overfitting)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=4,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=1234
    )
    
    # XGBoost (modello principale)
    xgb = XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=1234,
        n_jobs=-1
    )
    
    #Ensemble Stacking
    # Lista degli estimatori base
    estimators = [
        ('xgb', xgb),
        ('rf', rf),
        ('lr', lr)
    ]
    
    # Meta-modello: 
    meta_model = LogisticRegression(max_iter=1000, C=1.0, random_state=1234)

    # Stacking Classifier
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,               
        passthrough=False, 
        n_jobs=-1
    )
    
    return model
def correlation_pruning(X, threshold=0.90):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropped {len(to_drop)} correlated features: {to_drop} (>{threshold}).")
    return [f for f in X.columns if f not in to_drop]
def fit_predict_submit(model, features, X, y, start_time, middle_time, test_df, prefix=""):
    selected = features
    X_selected = X[selected]
    model.fit(X_selected, y)
    final_pipe = model
    y_train_pred = final_pipe.predict(X_selected)
    y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]
    acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring="accuracy")
    auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring="roc_auc")
    end_time = time.time()
    logging.info(f"[{prefix}]featureArray,accuracy_score_training,roc_auc_score,accuracy_cross_val_score,roc_auc_cross_val_score")
    log_msg = f"{[f for f in selected]},\n[{int(end_time-middle_time)}sec-{len(selected)}feat]\n{accuracy_score(y, y_train_pred)}->{acc.mean():.4f} ± {acc.std():.4f}, {roc_auc_score(y, y_train_proba)}->{auc.mean():.4f} ± {auc.std():.4f}"
    logging.info(log_msg)
    #print("featureArray,accuracy_score_training,roc_auc_score,accuracy_cross_val_score,roc_auc_cross_val_score")
    #print(log_msg)
    #complete_prefix = prefix+str(int(10000*accuracy_score(y, y_train_pred)))+"_"+str(int(10000*acc.mean()))
    predict_and_submit(test_df, selected, final_pipe, prefix=prefix)
    #print(f"Total execution time: {int(end_time-start_time)} seconds")
    logging.info(f"Total execution time: {int(end_time-start_time)} seconds")
