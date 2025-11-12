import os
import time
from start_utils import (
    create_features, read_train_data, read_test_data, 
    simple_train, build_pipe, train_regularization, 
    #greedy_feature_selection,
    greedy_feature_selection_dynamicC,
    #random_bucket_feature_search, 
    random_bucket_feature_search_robust,
    extract_features_by_importance,
    make_mi_scores,
    predict_and_submit)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

#1 read and prepare
COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('input', COMPETITION_NAME)

train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')

train_data = read_train_data(train_file_path)
test_data = read_test_data(test_file_path)

# Create feature DataFrames for both training and test sets
print("Processing training data...")
train_df = create_features(train_data)

print("\nProcessing test data...")
test_df = create_features(test_data, is_test=True)
    
print("\nTraining features preview:")
train_df.head().to_json("train_df_head.json", orient="records", indent=2)

features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']

#3. TRAIN
mi_scores = make_mi_scores(X, y)
#X_sel = X[mi_scores.head(15).keys()]
"""
10
Stacked model training accuracy: 0.8614
Stacked model training AUC: 0.9469520800000001
CV Accuracy: 0.8317 ± 0.0108
CV AUC: 0.9053 ± 0.0077

15
Stacked model training accuracy: 0.8753
Stacked model training AUC: 0.9595996
CV Accuracy: 0.8381 ± 0.0097
CV AUC: 0.9106 ± 0.0073

50
Stacked model training accuracy: 0.8916
Stacked model training AUC: 0.9713142400000001
CV Accuracy: 0.8394 ± 0.0088
CV AUC: 0.9136 ± 0.0073

70
Stacked model training accuracy: 0.8996
Stacked model training AUC: 0.9761216799999999
CV Accuracy: 0.8406 ± 0.0083
CV AUC: 0.9136 ± 0.0070

ALL
Stacked model training accuracy: 0.9007 
Stacked model training AUC: 0.9775587600000001 
CV Accuracy: 0.8405 ± 0.0092 
CV AUC: 0.9136 ± 0.0072
TEST 0.8386

A SAMPLE OF 10 AMONG THE FIRST BY MI
Stacked model training accuracy: 0.8758
Stacked model training AUC: 0.9621964
CV Accuracy: 0.8359 ± 0.0116
CV AUC: 0.9062 ± 0.0070

TAIL 15
Stacked model training accuracy: 0.9956
Stacked model training AUC: 0.9999178400000001
CV Accuracy: 0.5784 ± 0.0130
CV AUC: 0.6048 ± 0.0167
"""

# selected = mi_scores.index.tolist()

# #.head(15).keys()#features
# with open('mi_scores.txt', 'w') as f:
#     for feature in selected:
#         f.write(feature + "\n")
# exit()

selected = ["diff_final_hp",
"status_change_diff",
"p1_status_change",
"net_major_status_suffering",
"diff_final_schieramento",
"nr_pokemon_sconfitti_diff",
"p1_bad_status_advantage",
"battle_duration",
"p1_hp_std",
"p2_status_change",
"p2_cumulative_major_status_turns_pct", "p1_type_advantage"]
X_selected = X[selected]
print(f"selected shape={X_selected.shape}")

# --- Build base logistic regression model using your existing pipeline ---
logreg_search = build_pipe(USE_PCA=False, POLY_ENABLED=False)
logreg_search.fit(X_selected, y)

best_lr = logreg_search.best_estimator_
print("Best Logistic Regression params:", logreg_search.best_params_)

# --- Build random forest base model ---
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=1234,
    n_jobs=-1
)

# --- Combine them into a stacking classifier ---
stacked_model = StackingClassifier(
    estimators=[
        ('logreg', best_lr),
        ('rf', rf)
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=1234),
    passthrough=False,  # set to True if you want to include original features in meta-model
    n_jobs=-1
)

# --- Train the stacked model ---
print("\nTraining stacking ensemble (LogReg + RF)...")
stacked_model.fit(X_selected, y)

final_pipe = stacked_model

#EVALUATE
from sklearn.metrics import accuracy_score, roc_auc_score

y_train_pred = final_pipe.predict(X_selected)
y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]

print("Stacked model training accuracy:", accuracy_score(y, y_train_pred))
print("Stacked model training AUC:", roc_auc_score(y, y_train_proba))


#CHECK OVERFITTING
from sklearn.model_selection import cross_val_score

acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='accuracy')
auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='roc_auc')

print(f"CV Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
print(f"CV AUC: {auc.mean():.4f} ± {auc.std():.4f}")

predict_and_submit(test_df, selected, final_pipe)