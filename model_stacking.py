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
selected = mi_scores.index.tolist()

#.head(15).keys()#features
with open('mi_scores.txt', 'w') as f:
    for feature in selected:
        f.write(feature + "\n")
# exit()
import numpy as np

# Define the maximum acceptable correlation threshold
# A common starting point is 0.7 or 0.8.
CORRELATION_THRESHOLD = 0.5 
MAX_FEATURES = 50 # Set a limit for the final feature count

# 1. Get features ordered by MI score (highest first)
mi_ranked_features = mi_scores.index.tolist()

start = time.time()
# 2. Get the correlation matrix (absolute values)
correlation_matrix = X.corr().abs()
end = time.time()
print(f"Took {end-start} time to calculate correlation matrix")
# 3. Select features
final_selected = []
for feature in mi_ranked_features:
    is_highly_correlated = False
    
    # Check correlation with features already selected
    for selected_feature in final_selected:
        # Check if the correlation between the current feature and any selected feature 
        # is above the threshold
        if correlation_matrix.loc[feature, selected_feature] > CORRELATION_THRESHOLD:
            is_highly_correlated = True
            # Optional: print the features being excluded
            # print(f"Excluding {feature} (Corr={correlation_matrix.loc[feature, selected_feature]:.2f} with {selected_feature})")
            break
            
    if not is_highly_correlated:
        final_selected.append(feature)
        
    # Stop if we hit the max feature limit
    if len(final_selected) >= MAX_FEATURES:
        break

print(f"\n✅ Final Selected Features ({len(final_selected)}):")
# for feature in final_selected:
#     print(f"- {feature}")

new_end = time.time()
print(f"Took {new_end-end} time to do everything")    
# Use the new list for your model:
# X_selected = X[final_selected]
#exit()
"""
LOGISTIC + RF
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


"""
Stacked model training accuracy: 0.9086
Stacked model training AUC: 0.9828837199999999
CV Accuracy: 0.8327 ± 0.0084
CV AUC: 0.9045 ± 0.0079
"""
#11 features
# selected = ["diff_final_hp",
# "status_change_diff",
# "p1_status_change",
# "net_major_status_suffering",
# "diff_final_schieramento",
# "nr_pokemon_sconfitti_diff",
# "p1_bad_status_advantage",
# "battle_duration",
# "p1_hp_std",
# "p2_status_change",
# "p2_cumulative_major_status_turns_pct", "p1_type_advantage"]

"""
Stacked model training accuracy: 0.8922
Stacked model training AUC: 0.9753844
CV Accuracy: 0.8311 ± 0.0094
CV AUC: 0.9034 ± 0.0094
"""
#7 features

"""
- diff_final_hp
- status_change_diff
- p1_final_hp_per_ko
- net_major_status_suffering
- diff_final_schieramento
- p1_bad_status_advantage
- hp_diff_mean
- p2_pct_final_hp
- p2_n_pokemon_use
- p1_n_pokemon_use
- hp_diff_min
- late_hp_min_diff
- net_major_status_infliction
- diff_mean_stab
- p1_hp_advantage_std

15 top MI uncorrelated (corr threshold 75%)
Stacked model training accuracy: 0.9221
Stacked model training AUC: 0.98827732
CV Accuracy: 0.8392 ± 0.0098
CV AUC: 0.9086 ± 0.0077
"""

"""
50 top MI uncorrelated (corr threshold 75%)
Stacked model training accuracy: 0.9245
Stacked model training AUC: 0.98803736
CV Accuracy: 0.8380 ± 0.0093
CV AUC: 0.9097 ± 0.0079


15 top MI uncorrelated (corr threshold 50%)
Stacked model training accuracy: 0.8982
Stacked model training AUC: 0.9761671600000001
CV Accuracy: 0.8344 ± 0.0099
CV AUC: 0.9054 ± 0.0069

50 top MI uncorrelated (corr threshold 50%)
Stacked model training accuracy: 0.8888
Stacked model training AUC: 0.97028564
CV Accuracy: 0.8359 ± 0.0106
CV AUC: 0.9075 ± 0.0068
"""
"""

Stacked model training AUC: 0.5437/0.5660829 vs 0.5213 ± 0.0129/0.5326 ± 0.0151

p1_avg_high_speed_stat_battaglia
Stacked model training accuracy: 0.5
Stacked model training AUC: 0.5
CV Accuracy: 0.5020 ± 0.0040
CV AUC: 0.5010 ± 0.0021
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Build Random Forest base model ---
rf = RandomForestClassifier(
    n_estimators=300,
    # 🛑 KEY CHANGE 1: Restrict Max Depth for Regularization 
    max_depth=8,  
    random_state=1234,
    n_jobs=-1
)

# --- Build Gradient Boosting base model (ADDED REGULARIZATION) ---
gb = GradientBoostingClassifier(
    n_estimators=100,      
    # 🛑 KEY CHANGE 2: Reduce Learning Rate for Regularization
    learning_rate=0.05,     # Reduced from 0.1 to 0.05
    # 🛑 KEY CHANGE 3: Restrict Max Depth for Regularization
    max_depth=3,           # Kept shallow, which is good for GB
    random_state=1234
)

# --- Combine them into a Stacking Classifier ---
stacked_model = StackingClassifier(
    estimators=[
        ('rf', rf),         
        ('gb', gb)          
    ],
    # 🛑 Optional/Minor Change: Add regularization (C) to meta-model
    final_estimator=LogisticRegression(
        max_iter=2000, 
        C=0.1,  # Lower C means stronger L2 regularization
        random_state=1234
    ), 
    passthrough=False, 
    n_jobs=-1
)
"""
#final_selected
strong: "diff_final_hp"
Feature,Training_Accuracy,Training_AUC,CV_Accuracy,CV_AUC
Feature,Training_Accuracy,Training_AUC,CV_Accuracy,CV_AUC
p1_major_status_infliction_rate,0.6084, 0.61727986, 0.6076 ± 0.0064, 0.6147 ± 0.0074
p1_cumulative_major_status_turns_pct,0.6679, 0.7112755799999999, 0.6679 ± 0.0073, 0.7109 ± 0.0089
p2_major_status_infliction_rate,0.6025, 0.61424808, 0.6000 ± 0.0061, 0.6122 ± 0.0048
p2_cumulative_major_status_turns_pct,0.682, 0.72956716, 0.6820 ± 0.0063, 0.7290 ± 0.0057
p1_max_offense_boost_diff,0.5623, 0.5674023, 0.5615 ± 0.0054, 0.5628 ± 0.0070
p1_team_super_effective_moves,0.5, 0.5, 0.5000 ± 0.0000, 0.5000 ± 0.0000
expected_damage_ratio_turn_1,0.5367, 0.5607778400000001, 0.5148 ± 0.0056, 0.5217 ± 0.0061
p1_max_offensive_stat,0.5077, 0.51082224, 0.4913 ± 0.0111, 0.4927 ± 0.0093
p1_max_speed_stat,0.5166, 0.5212405, 0.5103 ± 0.0103, 0.5190 ± 0.0068
p1_mean_hp,0.5575, 0.58474574, 0.5373 ± 0.0151, 0.5495 ± 0.0175
p1_mean_spe,0.5379, 0.5577506200000001, 0.5165 ± 0.0079, 0.5242 ± 0.0102
p1_mean_atk,0.5546, 0.5850239399999999, 0.5413 ± 0.0074, 0.5643 ± 0.0096
p1_mean_def,0.5428, 0.5660964400000001, 0.5282 ± 0.0114, 0.5448 ± 0.0176
p1_mean_sp,0.5391, 0.5601696, 0.5295 ± 0.0138, 0.5408 ± 0.0153
p1_max_hp,0.5127, 0.51713568, 0.5127 ± 0.0057, 0.5171 ± 0.0080
p1_max_spe,0.5166, 0.5212405, 0.5103 ± 0.0103, 0.5190 ± 0.0068
p1_max_atk,0.5132, 0.51878682, 0.5080 ± 0.0079, 0.5179 ± 0.0117
p1_max_def,0.5342, 0.55574596, 0.5299 ± 0.0079, 0.5550 ± 0.0098
p1_max_spd,0.4925, 0.49073456, 0.5024 ± 0.0112, 0.4930 ± 0.0083
p1_min_hp,0.5198, 0.5311068600000002, 0.5100 ± 0.0113, 0.5233 ± 0.0121
p1_min_spe,0.505, 0.50840044, 0.5009 ± 0.0049, 0.5047 ± 0.0026
p1_min_atk,0.5084, 0.5095462399999999, 0.5076 ± 0.0049, 0.5070 ± 0.0054
p1_min_def,0.51, 0.51200106, 0.5089 ± 0.0061, 0.5099 ± 0.0070
p1_min_spd,0.521, 0.52638748, 0.5176 ± 0.0059, 0.5260 ± 0.0100
p1_std_hp,0.5819, 0.6317848400000001, 0.5566 ± 0.0066, 0.5775 ± 0.0103
p1_std_spe,0.5852, 0.62534256, 0.5461 ± 0.0123, 0.5690 ± 0.0079
p1_std_atk,0.5753, 0.62060092, 0.5517 ± 0.0094, 0.5734 ± 0.0067
p1_std_def,0.5838, 0.63220744, 0.5445 ± 0.0051, 0.5789 ± 0.0056
p1_std_spd,0.5798, 0.61774876, 0.5523 ± 0.0131, 0.5724 ± 0.0080
diff_hp,0.5363, 0.5614444999999999, 0.5292 ± 0.0097, 0.5471 ± 0.0136
diff_spe,0.5366, 0.5597836, 0.5307 ± 0.0101, 0.5498 ± 0.0111
diff_atk,0.5322, 0.5504808600000001, 0.5224 ± 0.0090, 0.5368 ± 0.0072
diff_def,0.5323, 0.55257406, 0.5216 ± 0.0109, 0.5358 ± 0.0103
diff_spd,0.5229, 0.5373775000000001, 0.5123 ± 0.0066, 0.5208 ± 0.0078
p1_avg_move_priority,0.5236, 0.5305598200000001, 0.5184 ± 0.0066, 0.5227 ± 0.0046
p2_avg_move_priority,0.5245, 0.53056306, 0.5203 ± 0.0027, 0.5225 ± 0.0028
p1_std_move_priority,0.5233, 0.53054104, 0.5183 ± 0.0068, 0.5230 ± 0.0049
p2_std_move_priority,0.5245, 0.53056306, 0.5203 ± 0.0027, 0.5225 ± 0.0028
priority_diff,0.5331, 0.5602166, 0.5233 ± 0.0050, 0.5427 ± 0.0066
priority_rate_advantage,0.5189, 0.5219993399999999, 0.5149 ± 0.0025, 0.5153 ± 0.0040
mean_base_atk_diff_timeline,0.5589, 0.6042188000000001, 0.5185 ± 0.0036, 0.5335 ± 0.0040
std_base_atk_diff_timeline,0.5967, 0.64342888, 0.5135 ± 0.0049, 0.5111 ± 0.0066
min_base_atk_diff_timeline,0.5235, 0.53886184, 0.5108 ± 0.0045, 0.5212 ± 0.0068
max_base_atk_diff_timeline,0.5293, 0.5440524, 0.5166 ± 0.0150, 0.5190 ± 0.0174
mean_base_spa_diff_timeline,0.5784, 0.61416778, 0.5280 ± 0.0104, 0.5386 ± 0.0106
std_base_spa_diff_timeline,0.577, 0.6212225200000001, 0.5015 ± 0.0102, 0.5074 ± 0.0183
min_base_spa_diff_timeline,0.5318, 0.54517816, 0.5228 ± 0.0032, 0.5303 ± 0.0067
max_base_spa_diff_timeline,0.5337, 0.5486489999999999, 0.5227 ± 0.0077, 0.5347 ± 0.0085
mean_base_spe_diff_timeline,0.5703, 0.6106049, 0.5358 ± 0.0061, 0.5531 ± 0.0129
std_base_spe_diff_timeline,0.5526, 0.59650298, 0.5037 ± 0.0072, 0.5046 ± 0.0090
min_base_spe_diff_timeline,0.5322, 0.5465145800000001, 0.5175 ± 0.0065, 0.5275 ± 0.0066
max_base_spe_diff_timeline,0.5322, 0.54338846, 0.5167 ± 0.0105, 0.5235 ± 0.0135
hp_diff_mean,0.6964, 0.76602782, 0.6806 ± 0.0078, 0.7379 ± 0.0135
hp_diff_max,0.6006, 0.64698962, 0.5742 ± 0.0101, 0.6037 ± 0.0131
hp_diff_min,0.6517, 0.6763335200000001, 0.6309 ± 0.0074, 0.6635 ± 0.0054
hp_diff_std,0.615, 0.6682574000000001, 0.5737 ± 0.0059, 0.5982 ± 0.0092
p1_hp_advantage_mean,0.6389, 0.6915764400000001, 0.6388 ± 0.0054, 0.6900 ± 0.0054
p1_hp_advantage_max,0.5016, 0.5016, 0.5016 ± 0.0011, 0.5016 ± 0.0011
p1_hp_advantage_min,0.5, 0.5, 0.5000 ± 0.0000, 0.5000 ± 0.0000
p1_hp_advantage_std,0.6136, 0.6452215000000001, 0.6136 ± 0.0050, 0.6436 ± 0.0061
p1_n_pokemon_use,0.6474, 0.67577798, 0.6474 ± 0.0109, 0.6756 ± 0.0144
p2_n_pokemon_use,0.6333, 0.66666738, 0.6333 ± 0.0052, 0.6665 ± 0.0049
diff_final_schieramento,0.6981, 0.7777952200000001, 0.6979 ± 0.0102, 0.7772 ± 0.0136
nr_pokemon_sconfitti_p1,0.6987, 0.76299662, 0.6987 ± 0.0078, 0.7630 ± 0.0072
nr_pokemon_sconfitti_p2,0.5, 0.5, 0.4998 ± 0.0004, 0.4972 ± 0.0057
nr_pokemon_sconfitti_diff,0.7045, 0.77685078, 0.7045 ± 0.0088, 0.7768 ± 0.0085
p1_pct_final_hp,0.7253, 0.80726816, 0.7103 ± 0.0098, 0.7861 ± 0.0094
p2_pct_final_hp,0.6647, 0.7376866200000001, 0.6546 ± 0.0069, 0.7106 ± 0.0060
diff_final_hp,0.8205, 0.90056532, 0.8138 ± 0.0134, 0.8854 ± 0.0085
battle_duration,0.6937, 0.75429436, 0.6937 ± 0.0071, 0.7543 ± 0.0074
hp_loss_rate,0.8204, 0.90249944, 0.8134 ± 0.0127, 0.8856 ± 0.0089
early_hp_mean_diff,0.6176, 0.6761159999999999, 0.6032 ± 0.0089, 0.6354 ± 0.0098
late_hp_mean_diff,0.669, 0.7320293, 0.6507 ± 0.0082, 0.7014 ± 0.0104
early_hp_min_diff,0.6204, 0.66794948, 0.5967 ± 0.0079, 0.6360 ± 0.0105
late_hp_min_diff,0.6493, 0.69653606, 0.6337 ± 0.0071, 0.6746 ± 0.0074
early_hp_max_diff,0.5974, 0.63908942, 0.5604 ± 0.0058, 0.5815 ± 0.0061
late_hp_max_diff,0.6091, 0.66035114, 0.5897 ± 0.0071, 0.6237 ± 0.0118
early_hp_std_diff,0.5914, 0.6407757600000001, 0.5495 ± 0.0101, 0.5568 ± 0.0133
late_hp_std_diff,0.5967, 0.65178436, 0.5401 ± 0.0073, 0.5584 ± 0.0086
hp_delta_trend,0.5967, 0.65238796, 0.5730 ± 0.0076, 0.5922 ± 0.0084
hp_advantage_trend,0.5967, 0.65238798, 0.5730 ± 0.0076, 0.5922 ± 0.0084
p1_hp_std,0.6912, 0.76526808, 0.6671 ± 0.0024, 0.7280 ± 0.0067
p2_hp_std,0.5978, 0.6594011999999999, 0.5669 ± 0.0103, 0.5860 ± 0.0117
hp_delta_std,0.615, 0.6682573599999999, 0.5737 ± 0.0059, 0.5982 ± 0.0092
p1_bad_status_advantage,0.7167, 0.7772237, 0.7150 ± 0.0159, 0.7740 ± 0.0148
p1_bad_status_advantage_min,0.5, 0.5, 0.5000 ± 0.0000, 0.5000 ± 0.0000
,0.7597, 0.8291447200000001, 0.7597 ± 0.0057, 0.8279 ± 0.0103
"""
print("\nTraining stacking ensemble (RF + GB)...")
print("Feature,Training_Accuracy,Training_AUC,CV_Accuracy,CV_AUC")
#for f in features:
# print(f"{[f for f in selected]}")
selected = ["status_change_diff"]#[ f]

X_selected = X[selected]

# --- Train the stacked model ---

stacked_model.fit(X_selected, y)

final_pipe = stacked_model

#EVALUATE


y_train_pred = final_pipe.predict(X_selected)
y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]

# print("Stacked model training accuracy:", accuracy_score(y, y_train_pred))
# print("Stacked model training AUC:", roc_auc_score(y, y_train_proba))


#CHECK OVERFITTING
from sklearn.model_selection import cross_val_score

acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='accuracy')
auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='roc_auc')

# print(f"CV Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
# print(f"CV AUC: {auc.mean():.4f} ± {auc.std():.4f}")

print(f"{[f for f in selected]},{accuracy_score(y, y_train_pred)}, {roc_auc_score(y, y_train_proba)}, {acc.mean():.4f} ± {acc.std():.4f}, {auc.mean():.4f} ± {auc.std():.4f}")

#predict_and_submit(test_df, selected, final_pipe)