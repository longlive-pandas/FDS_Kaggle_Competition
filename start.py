import os
import time
from start_utils import (
    create_features, read_train_data, read_test_data, 
    simple_train, build_pipe, train_regularization, 
    #greedy_feature_selection,
    greedy_feature_selection_dynamicC,
    #random_bucket_feature_search, 
    random_bucket_feature_search_robust,
    tune_threshold,
    extract_features_by_importance,
    predict_and_submit)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#1 read and prepare
COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('input', COMPETITION_NAME)

train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')

train_data = read_train_data(train_file_path)
test_data = read_test_data(test_file_path)

#2 create features
# Create feature DataFrames for both training and test sets
print("Processing training data...")
train_df = create_features(train_data)

print("\nProcessing test data...")
test_df = create_features(test_data, is_test=True)
    
print("\nTraining features preview:")
#display(train_df.head())
train_df.head().to_json("train_df_head.json", orient="records", indent=2)
#train_df.describe()

# Define features and target
#features = ['p1_type_diversity', 'p1_type_resistance', 'p1_type_weakness']
#features = ['diff_mean_stab']
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
# print(f"Using {len(features)} features")
X = train_df[features]
y = train_df['player_won']

#3. TRAIN
"""
Using bucket=35 computed features
Goal Score: 0.8420 with 34 features (min mean)

Fitting 5 folds for each of 34 candidates, totalling 170 fits
Best params: {'logreg__C': 1, 'logreg__penalty': 'l1', 'logreg__solver': 'liblinear'}
Best CV mean: 0.8434 ± 0.0045
Seed 42: 0.8426 ± 0.0063
Seed 1234: 0.8434 ± 0.0045
Seed 999: 0.8419 ± 0.0063
Seed 2023: 0.8419 ± 0.0027
It took 33.034090995788574 time
---
#Algorithm: bucket(35)
Fitting 5 folds for each of 34 candidates, totalling 170 fits
Best params: {'logreg__C': 1, 'logreg__penalty': 'l1', 'logreg__solver': 'liblinear'}
Best CV mean: 0.8434 ± 0.0045
Seed 1039284721: 0.8434 ± 0.0065
Seed 398172634: 0.8412 ± 0.0073
Seed 2750193806: 0.8427 ± 0.0081
Seed 198234176: 0.8414 ± 0.0027
Seed 4129837512: 0.8420 ± 0.0073
Seed 1298374650: 0.8440 ± 0.0088
Seed 3029487619: 0.8421 ± 0.0066
Seed 718236451: 0.8429 ± 0.0045
Seed 2543197682: 0.8431 ± 0.0099
Seed 1765432987: 0.8421 ± 0.0101
Seed 389124765: 0.8420 ± 0.0067
Seed 612984372: 0.8418 ± 0.0080
Seed 2983716540: 0.8433 ± 0.0079
Seed 830174562: 0.8426 ± 0.0068
Seed 1229837465: 0.8422 ± 0.0069
Seed 4198372651: 0.8431 ± 0.0052
Seed 2378164529: 0.8425 ± 0.0041
Seed 3487612098: 0.8426 ± 0.0061
Seed 954613287: 0.8425 ± 0.0034
Seed 1864293754: 0.8408 ± 0.0046
It took 94.43389391899109 time
"""

# selected = ['p1_cumulative_major_status_turns_pct', 'p1_avg_high_speed_stat_battaglia', 'p2_major_status_infliction_rate', 
# 'p1_mean_atk', 'p1_bad_status_advantage', 'p2_type_advantage', 
# 'diff_mean_stab', 'priority_diff', 'p1_mean_sp', 
# 'p1_type_resistance', 'diff_spd', 'p1_type_advantage', 
# 'std_base_spe_diff_timeline', 'diff_atk', 'late_hp_mean_diff', 
# 'p2_n_pokemon_use', 'diff_final_hp', 'std_base_atk_diff_timeline', 
# 'diff_hp', 'diff_final_schieramento', 'p1_status_change', 
# 'nr_pokemon_sconfitti_p2', 'p1_major_status_infliction_rate', 'net_major_status_suffering', 
# 'hp_delta_std', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 
# 'p1_mean_def', 'p1_avg_speed_stat_battaglia', 'p1_final_hp_per_ko', 
# 'diff_def', 'p1_type_weakness', 'battle_duration', 
# 'p1_n_pokemon_use']
#-----------------
"""
Algorithm: Bucket(67), C=30
Fitting 5 folds for each of 34 candidates, totalling 170 fits
Best params: {'logreg__C': 3, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}
Best CV mean: 0.8442 ± 0.0041 (da 8419 a 8451)
It took 33.62293004989624 time
"""
# selected = ['p2_status_change', 'std_base_spa_diff_timeline', 'status_change_diff', 'p1_hp_advantage_mean', 'nr_pokemon_sconfitti_p2', 'net_major_status_suffering', 'p1_max_offense_boost_diff', 'p1_mean_hp', 'diff_hp', 'p1_mean_def', 'p1_cumulative_major_status_turns_pct', 'p1_mean_atk', 'diff_atk', 'p2_n_pokemon_use', 'p1_status_change', 'battle_duration', 'p2_mean_stab', 'p1_type_advantage', 'p1_mean_sp', 'p2_cumulative_major_status_turns_pct', 'diff_final_hp', 'diff_final_schieramento', 'hp_advantage_trend', 'diff_type_advantage', 'std_base_spe_diff_timeline', 'p1_mean_spe', 'p2_type_advantage', 'expected_damage_ratio_turn_1', 'std_base_atk_diff_timeline', 'p1_max_speed_offense_product', 'mean_base_spe_diff_timeline', 'nr_pokemon_sconfitti_p1', 'p1_pct_final_hp', 'p2_major_status_infliction_rate', 'p1_bad_status_advantage', 'p1_mean_stab', 'p1_max_speed_stat', 'diff_def', 'hp_diff_mean', 'priority_diff', 'hp_delta_trend', 'late_hp_mean_diff', 'p1_avg_high_speed_stat_battaglia', 'hp_delta_std', 'p1_type_weakness', 'diff_mean_stab']

# selected = features#reset
# start = time.time()
# X_selected = X[selected]
# print(f"Selected {len(selected)} features, X_selected.shape={X_selected.shape}")
# final_pipe = train_regularization(X_selected,y)
# end = time.time()
# print(f"It took {end-start} time")
# exit()
# simple_train(X,y)

# ###GRID SEARCH AND GREEDY
"""
--- Iteration 1 ---
Remaining features: 64 | Selected so far: 0
Best candidate: status_change_diff | robust score = 0.7582 | Δ = 0.7582
 ✅ Accepted feature 'status_change_diff'. New best robust score: 0.7582

--- Iteration 2 ---
Remaining features: 63 | Selected so far: 1
Best candidate: diff_final_schieramento | robust score = 0.7915 | Δ = 0.0333
 ✅ Accepted feature 'diff_final_schieramento'. New best robust score: 0.7915

--- Iteration 3 ---
Remaining features: 62 | Selected so far: 2
Best candidate: diff_final_hp | robust score = 0.8259 | Δ = 0.0344
 ✅ Accepted feature 'diff_final_hp'. New best robust score: 0.8259

--- Iteration 4 ---
Remaining features: 61 | Selected so far: 3
Best candidate: net_major_status_suffering | robust score = 0.8309 | Δ = 0.0050
 ✅ Accepted feature 'net_major_status_suffering'. New best robust score: 0.8309

--- Iteration 5 ---
Remaining features: 60 | Selected so far: 4
Best candidate: p1_bad_status_advantage | robust score = 0.8351 | Δ = 0.0042
 ✅ Accepted feature 'p1_bad_status_advantage'. New best robust score: 0.8351

--- Iteration 6 ---
Remaining features: 59 | Selected so far: 5
Best candidate: priority_diff | robust score = 0.8365 | Δ = 0.0014
 ✅ Accepted feature 'priority_diff'. New best robust score: 0.8365

--- Iteration 7 ---
Remaining features: 58 | Selected so far: 6
Best candidate: p1_n_pokemon_use | robust score = 0.8375 | Δ = 0.0010
 ✅ Accepted feature 'p1_n_pokemon_use'. New best robust score: 0.8375

--- Iteration 8 ---
Remaining features: 57 | Selected so far: 7
Best candidate: p2_major_status_infliction_rate | robust score = 0.8386 | Δ = 0.0011
 ✅ Accepted feature 'p2_major_status_infliction_rate'. New best robust score: 0.8386

--- Iteration 9 ---
Remaining features: 56 | Selected so far: 8
Best candidate: p1_hp_advantage_mean | robust score = 0.8405 | Δ = 0.0019
 ✅ Accepted feature 'p1_hp_advantage_mean'. New best robust score: 0.8405

--- Iteration 10 ---
Remaining features: 55 | Selected so far: 9
Best candidate: diff_mean_stab | robust score = 0.8408 | Δ = 0.0003
 ⏹️ No meaningful improvement. Stopping.
"""
# start1 = time.time()
# selected, history = greedy_feature_selection_dynamicC(
#     X, y,
#     cv=5,
#     seed_list=[42, 1234, 999, 2023],
#     C_grid=[0.1, 1, 3, 10, 30],
#     min_delta=0.0005,
#     verbose=True
# )
# start2 = time.time()
# print(f"{start2-start1} Time")
# # Run random bucket search
# """
# n_buckets 100
# bucket_size 25
# Score: 0.8400 with 25 features
# Top features: [
# 'p1_bad_status_advantage', 'diff_hp', 'status_change_diff', 'hp_diff_mean', 'p2_n_pokemon_use',
# 'nr_pokemon_sconfitti_diff', 'early_hp_mean_diff', 'std_base_spa_diff_timeline', 'p1_final_hp_per_ko', 'p1_type_resistance', 
# 'p1_pct_final_hp', 'p1_mean_atk', 'diff_atk', 'p1_type_diversity', 'diff_type_advantage',
# 'net_major_status_infliction', 'diff_spd', 'hp_loss_rate', 'p1_mean_sp', 'p2_mean_stab',
# 'p1_type_advantage', 'net_major_status_suffering', 'mean_base_atk_diff_timeline', 'p2_status_change', 'diff_final_schieramento']
# Best features: ['p1_bad_status_advantage', 'diff_hp', 'status_change_diff', 'hp_diff_mean', 'p2_n_pokemon_use', 'nr_pokemon_sconfitti_diff', 'early_hp_mean_diff', 'std_base_spa_diff_timeline', 'p1_final_hp_per_ko', 'p1_type_resistance', 'p1_pct_final_hp', 'p1_mean_atk', 'diff_atk', 'p1_type_diversity', 'diff_type_advantage', 'net_major_status_infliction', 'diff_spd', 'hp_loss_rate', 'p1_mean_sp', 'p2_mean_stab', 'p1_type_advantage', 'net_major_status_suffering', 'mean_base_atk_diff_timeline', 'p2_status_change', 'diff_final_schieramento'] {'best_score': 0.8400000000000001, 'best_features': ['p1_bad_status_advantage', 'diff_hp', 'status_change_diff', 'hp_diff_mean', 'p2_n_pokemon_use', 'nr_pokemon_sconfitti_diff', 'early_hp_mean_diff', 'std_base_spa_diff_timeline', 'p1_final_hp_per_ko', 'p1_type_resistance', 'p1_pct_final_hp', 'p1_mean_atk', 'diff_atk', 'p1_type_diversity', 'diff_type_advantage', 'net_major_status_infliction', 'diff_spd', 'hp_loss_rate', 'p1_mean_sp', 'p2_mean_stab', 'p1_type_advantage', 'net_major_status_suffering', 'mean_base_atk_diff_timeline', 'p2_status_change', 'diff_final_schieramento']
# """
# results = random_bucket_feature_search_robust(
#     X, y,
#     n_buckets=100,
#     bucket_size=25,
#     cv=5,
#     seed_list=[42, 1234, 999, 2023],
#     C=30,
#     try_subsets=True,
#     verbose=True
# )

# print("Best features:", results["best_features"], results)

# """
# 🏆 Best bucket found:
# Score: 0.8420 with 34 features
# Top features: ['p1_cumulative_major_status_turns_pct', 'p1_avg_high_speed_stat_battaglia', 'p2_major_status_infliction_rate', 
# 'p1_mean_atk', 'p1_bad_status_advantage', 'p2_type_advantage', 
# 'diff_mean_stab', 'priority_diff', 'p1_mean_sp', 
# 'p1_type_resistance', 'diff_spd', 'p1_type_advantage', 
# 'std_base_spe_diff_timeline', 'diff_atk', 'late_hp_mean_diff', 
# 'p2_n_pokemon_use', 'diff_final_hp', 'std_base_atk_diff_timeline', 
# 'diff_hp', 'diff_final_schieramento', 'p1_status_change', 
# 'nr_pokemon_sconfitti_p2', 'p1_major_status_infliction_rate', 'net_major_status_suffering', 
# 'hp_delta_std', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 
# 'p1_mean_def', 'p1_avg_speed_stat_battaglia', 'p1_final_hp_per_ko', 
# 'diff_def', 'p1_type_weakness', 'battle_duration', 
# 'p1_n_pokemon_use']
# """
# results = random_bucket_feature_search_robust(
#     X, y,
#     n_buckets=100,
#     bucket_size=35,
#     cv=5,
#     seed_list=[42, 1234, 999, 2023],
#     C=30,
#     try_subsets=True,
#     verbose=True
# )

# print("Best features:", results["best_features"], results)

###
# """
# C=1
# """
# C=1
# seed_list = [
#         1039284721,
#         # 398172634,
#         # 2750193806,
#         198234176,
#         4129837512,
#         1298374650,
#         # 3029487619,
#         # 718236451,
#         # 2543197682,
#         # 1765432987,
#         389124765,
#         612984372,
#         # 2983716540,
#         830174562,
#         1229837465,
#         # 4198372651,
#         # 2378164529,
#         # 3487612098,
#         954613287,
#         1864293754,
#     ]
# n_buckets = 80
# results = random_bucket_feature_search_robust(
#     X, y,
#     n_buckets=n_buckets,
#     bucket_size=len(features),
#     cv=5,
#     seed_list=seed_list,#[42, 1234, 999, 2023],
#     C=C,
#     try_subsets=True,
#     verbose=True
# )

# print(f"Best features with C={C}:{results["best_features"]}")
# """
# C=30
# Score: 0.8441 with 46 features
# Top features: 
# ['p2_status_change', 'std_base_spa_diff_timeline', 'status_change_diff', 'p1_hp_advantage_mean', 'nr_pokemon_sconfitti_p2', 'net_major_status_suffering', 'p1_max_offense_boost_diff', 'p1_mean_hp', 'diff_hp', 'p1_mean_def', 'p1_cumulative_major_status_turns_pct', 'p1_mean_atk', 'diff_atk', 'p2_n_pokemon_use', 'p1_status_change', 'battle_duration', 'p2_mean_stab', 'p1_type_advantage', 'p1_mean_sp', 'p2_cumulative_major_status_turns_pct', 'diff_final_hp', 'diff_final_schieramento', 'hp_advantage_trend', 'diff_type_advantage', 'std_base_spe_diff_timeline', 'p1_mean_spe', 'p2_type_advantage', 'expected_damage_ratio_turn_1', 'std_base_atk_diff_timeline', 'p1_max_speed_offense_product', 'mean_base_spe_diff_timeline', 'nr_pokemon_sconfitti_p1', 'p1_pct_final_hp', 'p2_major_status_infliction_rate', 'p1_bad_status_advantage', 'p1_mean_stab', 'p1_max_speed_stat', 'diff_def', 'hp_diff_mean', 'priority_diff', 'hp_delta_trend', 'late_hp_mean_diff', 'p1_avg_high_speed_stat_battaglia', 'hp_delta_std', 'p1_type_weakness', 'diff_mean_stab']

# """
# C=30
# results = random_bucket_feature_search_robust(
#     X, y,
#     n_buckets=n_buckets,
#     bucket_size=len(features),
#     cv=5,
#     seed_list=seed_list,#
#     C=C,
#     try_subsets=True,
#     verbose=True
# )

# print(f"Best features with C={C}:{results["best_features"]}")
# """
# C=3
# Score: 0.8440 with 55 features
# Top features: ['p2_cumulative_major_status_turns_pct', 'diff_spe', 'diff_final_hp', 'hp_advantage_trend', 'diff_final_schieramento', 'p1_type_resistance', 'diff_type_advantage', 'p1_major_status_infliction_rate', 'priority_rate_advantage', 'mean_base_spa_diff_timeline', 'p1_n_pokemon_use', 'mean_base_atk_diff_timeline', 'priority_diff', 'p1_mean_hp', 'p1_max_speed_stat', 'p2_mean_stab', 'p2_type_advantage', 'p1_bad_status_advantage', 'std_base_spa_diff_timeline', 'p1_hp_std', 'p1_final_hp_per_ko', 'nr_pokemon_sconfitti_diff', 'p1_type_diversity', 'diff_hp', 'p1_cumulative_major_status_turns_pct', 'diff_mean_stab', 'p1_hp_advantage_mean', 'mean_base_spe_diff_timeline', 'expected_damage_ratio_turn_1', 'net_major_status_infliction', 'hp_delta_trend', 'p2_pct_final_hp', 'p2_n_pokemon_use', 'p2_status_change', 'p1_mean_spe', 'nr_pokemon_sconfitti_p2', 'p1_max_speed_offense_product', 'nr_pokemon_sconfitti_p1', 'p2_hp_std', 'p1_max_offense_boost_diff', 'battle_duration', 'std_base_atk_diff_timeline', 'diff_def', 'diff_spd', 'p1_type_weakness', 'diff_atk', 'p1_max_offensive_stat', 'p1_pct_final_hp', 'hp_loss_rate', 'status_change_diff', 'early_hp_mean_diff', 'late_hp_mean_diff', 'p1_mean_sp', 'p1_mean_def', 'p1_mean_atk']
# """
# C=3
# results = random_bucket_feature_search_robust(
#     X, y,
#     n_buckets=n_buckets,
#     bucket_size=len(features),
#     cv=5,
#     seed_list=seed_list,#
#     C=C,
#     try_subsets=True,
#     verbose=True
# )

# print(f"Best features with C={C}:{results["best_features"]}")


# #pipe = train_regularization(X,y)
# #4 SUBMIT
# exit()

#finally 
"""
39/49 features: train (shuffle seed 42) 84.44% (+/- 0.43%), train(shuffle seed 1234)84.51% (+/- 1.04%); test: 83.46 
49/49 features: train (shuffle seed 42) 84.25% (+/- 0.33%), train(shuffle seed 1234)84.30% (+/- 1.17%); test: 83.26
""" 
selected = features#['p1_max_offensive_stat', 'p1_mean_sp', 'p1_cumulative_major_status_turns_pct', 'diff_def', 'mean_base_atk_diff_timeline', 'p1_n_pokemon_use', 'p1_hp_std', 'diff_atk', 'p2_cumulative_major_status_turns_pct', 'p1_bad_status_advantage', 'battle_duration', 'p1_mean_def', 'p1_avg_speed_stat_battaglia', 'mean_base_spa_diff_timeline', 'late_hp_mean_diff', 'nr_pokemon_sconfitti_p1', 'p1_type_weakness', 'p1_avg_high_speed_stat_battaglia', 'p1_pct_final_hp', 'p1_hp_advantage_mean', 'std_base_spa_diff_timeline', 'hp_diff_mean', 'std_base_spe_diff_timeline', 'diff_hp', 'p2_n_pokemon_use', 'p1_mean_atk', 'p2_hp_std', 'p2_major_status_infliction_rate', 'mean_base_spe_diff_timeline', 'hp_advantage_trend', 'p1_type_resistance', 'nr_pokemon_sconfitti_p2', 'priority_diff', 'diff_final_schieramento', 'p1_type_diversity', 'p1_mean_spe', 'p1_major_status_infliction_rate', 'expected_damage_ratio_turn_1', 'hp_loss_rate']
#now that I know which features are best I fit my model to these and submit
#selected = ['net_major_status_infliction', 'battle_duration', 'p1_max_offense_boost_diff', 'diff_final_hp', 'std_base_spa_diff_timeline', 'nr_pokemon_sconfitti_p1', 'p1_avg_high_speed_stat_battaglia', 'diff_spe', 'expected_damage_ratio_turn_1', 'hp_loss_rate', 'p2_cumulative_major_status_turns_pct', 'p1_mean_spe', 'p1_type_resistance', 'p1_max_speed_offense_product', 'nr_pokemon_sconfitti_p2', 'p2_hp_std', 'net_major_status_suffering', 'p2_n_pokemon_use', 'p1_type_weakness', 'diff_mean_stab', 'p1_mean_sp', 'p1_cumulative_major_status_turns_pct', 'p1_mean_def', 'hp_delta_std', 'mean_base_spa_diff_timeline', 'p1_n_pokemon_use', 'p1_mean_stab', 'diff_spd', 'p2_mean_stab', 'diff_hp', 'mean_base_atk_diff_timeline', 'p1_type_diversity', 'status_change_diff', 'diff_type_advantage', 'priority_diff', 'std_base_spe_diff_timeline', 'diff_final_schieramento', 'p1_mean_hp', 'p1_pct_final_hp', 'p2_pct_final_hp', 'p2_major_status_infliction_rate', 'p1_hp_advantage_mean', 'p1_max_speed_stat', 'diff_atk', 'nr_pokemon_sconfitti_diff', 'hp_advantage_trend', 'std_base_atk_diff_timeline', 'hp_delta_trend', 'late_hp_mean_diff', 'p1_bad_status_advantage', 'p1_major_status_infliction_rate']
"""
10/11/2025 9.26
Best params: {'logreg__C': 1, 'logreg__l1_ratio': 0.9, 'logreg__penalty': 'elasticnet', 'logreg__solver': 'saga'}
Best CV mean: 0.8441 ± 0.0045
Seed 1039284721: 0.8437 ± 0.0067
Seed 398172634: 0.8436 ± 0.0077
Seed 2750193806: 0.8426 ± 0.0092
Seed 198234176: 0.8431 ± 0.0029
Seed 4129837512: 0.8434 ± 0.0076
Seed 1298374650: 0.8441 ± 0.0094
Seed 3029487619: 0.8431 ± 0.0092
Seed 718236451: 0.8453 ± 0.0026
Seed 2543197682: 0.8432 ± 0.0106
Seed 1765432987: 0.8440 ± 0.0091
Seed 389124765: 0.8442 ± 0.0079
Seed 612984372: 0.8427 ± 0.0062
Seed 2983716540: 0.8432 ± 0.0062
Seed 830174562: 0.8438 ± 0.0071
Seed 1229837465: 0.8448 ± 0.0071
Seed 4198372651: 0.8446 ± 0.0056
Seed 2378164529: 0.8430 ± 0.0041
Seed 3487612098: 0.8443 ± 0.0069
Seed 954613287: 0.8445 ± 0.0031
Seed 1864293754: 0.8438 ± 0.0049
Generating predictions on the test set...

9.29
Best params: {'logreg__C': 0.1, 'logreg__l1_ratio': 0.5, 'logreg__penalty': 'elasticnet', 'logreg__solver': 'saga'}
Best CV mean: 0.8427 ± 0.0043
Seed 1039284721: 0.8406 ± 0.0091
Seed 398172634: 0.8413 ± 0.0073
Seed 2750193806: 0.8413 ± 0.0096
Seed 198234176: 0.8419 ± 0.0015
Seed 4129837512: 0.8398 ± 0.0075
Seed 1298374650: 0.8426 ± 0.0086
Seed 3029487619: 0.8409 ± 0.0081
Seed 718236451: 0.8428 ± 0.0039
Seed 2543197682: 0.8424 ± 0.0102
Seed 1765432987: 0.8424 ± 0.0094
Seed 389124765: 0.8419 ± 0.0070
Seed 612984372: 0.8401 ± 0.0066
Seed 2983716540: 0.8406 ± 0.0053
Seed 830174562: 0.8413 ± 0.0080
Seed 1229837465: 0.8423 ± 0.0064
Seed 4198372651: 0.8412 ± 0.0053
Seed 2378164529: 0.8418 ± 0.0042
Seed 3487612098: 0.8427 ± 0.0068
Seed 954613287: 0.8411 ± 0.0037
Seed 1864293754: 0.8415 ± 0.0044
"""
X_selected = X[selected]
print(f"selected shape={X_selected.shape}")
final_pipe = train_regularization(X_selected,y)

extracted_features_and_weights = extract_features_by_importance(final_pipe, selected)
print(f"extracted_features_and_weights under linearity assumption: {extracted_features_and_weights}")
# with open("extracted_features_and_weights.txt", "w") as f:
#     f.write(extracted_features_and_weights.to_string())
extracted_features_and_weights.to_csv("extracted_features_and_weights.csv", index=False)
#final_pipe = simple_train(X_selected,y)#creates and fits pipe
import json
with open("features_list_model83.33.json", "w") as f:
    json.dump(selected, f)
# Tune threshold on the training data
best_threshold = tune_threshold(final_pipe, X_selected, y)
predict_and_submit(test_df, features, final_pipe, threshold=best_threshold)



"""
on/start.py
Processing training data...
Extracting features:   0%|          | 0/10000 [00:00<?, ?it/s]

Processing test data...
Extracting features:   0%|          | 0/5000 [00:00<?, ?it/s]

Training features preview:
Bucket   1 → robust CV=0.8424 (44 features)
Bucket 1 took 159.92470407485962 time
Bucket   2 → robust CV=0.8413 (65 features)
Bucket 2 took 155.8063747882843 time
Bucket   3 → robust CV=0.8424 (57 features)
Bucket 3 took 160.08282089233398 time
Bucket   4 → robust CV=0.8418 (54 features)
Bucket 4 took 158.4005057811737 time
Bucket   5 → robust CV=0.8419 (63 features)
Bucket 5 took 160.62140703201294 time
Bucket   6 → robust CV=0.8418 (62 features)
Bucket 6 took 159.07306003570557 time
Bucket   7 → robust CV=0.8417 (50 features)
Bucket 7 took 152.51358199119568 time
Bucket   8 → robust CV=0.8416 (56 features)
Bucket 8 took 158.038419008255 time
Bucket   9 → robust CV=0.8427 (57 features)
Bucket 9 took 161.01064014434814 time
Bucket  10 → robust CV=0.8421 (53 features)
Bucket 10 took 158.64841318130493 time
Bucket  11 → robust CV=0.8418 (54 features)
Bucket 11 took 162.01017427444458 time
Bucket  12 → robust CV=0.8415 (59 features)
Bucket 12 took 159.26028108596802 time
Bucket  13 → robust CV=0.8421 (62 features)
Bucket 13 took 158.02277421951294 time
Bucket  14 → robust CV=0.8416 (63 features)
Bucket 14 took 155.18546295166016 time
Bucket  15 → robust CV=0.8418 (61 features)
Bucket 15 took 161.75664520263672 time
Bucket  16 → robust CV=0.8419 (59 features)
Bucket 16 took 155.82410502433777 time
Bucket  17 → robust CV=0.8418 (57 features)
Bucket 17 took 162.80598497390747 time
Bucket  18 → robust CV=0.8419 (64 features)
Bucket 18 took 157.67289400100708 time
Bucket  19 → robust CV=0.8417 (61 features)
Bucket 19 took 161.83914184570312 time
Bucket  20 → robust CV=0.8419 (65 features)
Bucket 20 took 165.3551049232483 time
Bucket  21 → robust CV=0.8416 (55 features)
Bucket 21 took 160.5589780807495 time
Bucket  22 → robust CV=0.8425 (51 features)
Bucket 22 took 159.75327110290527 time
Bucket  23 → robust CV=0.8417 (61 features)
Bucket 23 took 158.49593591690063 time
Bucket  24 → robust CV=0.8422 (55 features)
Bucket 24 took 150.73517680168152 time
Bucket  25 → robust CV=0.8421 (55 features)
Bucket 25 took 159.77960872650146 time
Bucket  26 → robust CV=0.8417 (47 features)
Bucket 26 took 153.92723298072815 time
Bucket  27 → robust CV=0.8415 (59 features)
Bucket 27 took 159.26315903663635 time
Bucket  28 → robust CV=0.8419 (51 features)
Bucket 28 took 158.41698002815247 time
Bucket  29 → robust CV=0.8419 (64 features)
Bucket 29 took 154.5263090133667 time
Bucket  30 → robust CV=0.8419 (51 features)
Bucket 30 took 157.52723097801208 time
Bucket  31 → robust CV=0.8418 (63 features)
Bucket 31 took 156.70402264595032 time
Bucket  32 → robust CV=0.8420 (63 features)
Bucket 32 took 158.83947372436523 time
Bucket  33 → robust CV=0.8418 (61 features)
Bucket 33 took 159.84325695037842 time
Bucket  34 → robust CV=0.8414 (62 features)
Bucket 34 took 160.4940071105957 time
Bucket  35 → robust CV=0.8419 (65 features)
Bucket 35 took 157.61908292770386 time
Bucket  36 → robust CV=0.8415 (64 features)
Bucket 36 took 156.8079080581665 time
Bucket  37 → robust CV=0.8420 (61 features)
Bucket 37 took 154.67036890983582 time
Bucket  38 → robust CV=0.8413 (65 features)
Bucket 38 took 162.95130705833435 time
Bucket  39 → robust CV=0.8420 (57 features)
Bucket 39 took 158.77612471580505 time
Bucket  40 → robust CV=0.8417 (62 features)
Bucket 40 took 158.10168194770813 time
Bucket  41 → robust CV=0.8420 (53 features)
Bucket 41 took 156.55175614356995 time
Bucket  42 → robust CV=0.8418 (64 features)
Bucket 42 took 165.29659295082092 time
Bucket  43 → robust CV=0.8420 (60 features)
Bucket 43 took 155.53614377975464 time
Bucket  44 → robust CV=0.8422 (56 features)
Bucket 44 took 155.84640073776245 time
Bucket  45 → robust CV=0.8421 (60 features)
Bucket 45 took 160.51579117774963 time
Bucket  46 → robust CV=0.8418 (58 features)
Bucket 46 took 160.20363306999207 time
Bucket  47 → robust CV=0.8420 (60 features)
Bucket 47 took 160.05242109298706 time
Bucket  48 → robust CV=0.8418 (61 features)
Bucket 48 took 160.95927906036377 time
Bucket  49 → robust CV=0.8419 (55 features)
Bucket 49 took 162.32286715507507 time
Bucket  50 → robust CV=0.8415 (45 features)
Bucket 50 took 156.51996874809265 time

🏆 Best bucket found:
Score: 0.8427 with 57 features
Top features: ['p1_hp_std', 'status_change_diff', 'hp_diff_mean', 'p1_avg_high_speed_stat_battaglia', 'p1_mean_spe', 'p2_hp_std', 'nr_pokemon_sconfitti_p1', 'p1_max_offensive_stat', 'p1_mean_stab', 'p1_mean_def', 'mean_base_spe_diff_timeline', 'diff_def', 'diff_spe', 'p1_mean_sp', 'std_base_atk_diff_timeline', 'diff_type_advantage', 'p1_hp_advantage_mean', 'p1_team_super_effective_moves', 'hp_loss_rate', 'p1_max_offense_boost_diff', 'std_base_spa_diff_timeline', 'p1_max_speed_stat', 'p1_n_pokemon_use', 'diff_spd', 'battle_duration', 'diff_hp', 'net_major_status_suffering', 'priority_rate_advantage', 'p1_bad_status_advantage', 'p2_pct_final_hp', 'p1_type_diversity', 'expected_damage_ratio_turn_1', 'p1_type_resistance', 'diff_mean_stab', 'p1_max_speed_offense_product', 'early_hp_mean_diff', 'diff_final_hp', 'p2_status_change', 'diff_final_schieramento', 'diff_atk', 'p1_type_weakness', 'std_base_spe_diff_timeline', 'p2_type_advantage', 'p1_major_status_infliction_rate', 'p1_avg_speed_stat_battaglia', 'p1_mean_hp', 'nr_pokemon_sconfitti_diff', 'hp_delta_trend', 'p1_type_advantage', 'mean_base_atk_diff_timeline', 'p1_mean_atk', 'net_major_status_infliction', 'p1_final_hp_per_ko', 'priority_diff', 'late_hp_mean_diff', 'p2_major_status_infliction_rate', 'hp_advantage_trend']
Best features with C=1:['p1_hp_std', 'status_change_diff', 'hp_diff_mean', 'p1_avg_high_speed_stat_battaglia', 'p1_mean_spe', 'p2_hp_std', 'nr_pokemon_sconfitti_p1', 'p1_max_offensive_stat', 'p1_mean_stab', 'p1_mean_def', 'mean_base_spe_diff_timeline', 'diff_def', 'diff_spe', 'p1_mean_sp', 'std_base_atk_diff_timeline', 'diff_type_advantage', 'p1_hp_advantage_mean', 'p1_team_super_effective_moves', 'hp_loss_rate', 'p1_max_offense_boost_diff', 'std_base_spa_diff_timeline', 'p1_max_speed_stat', 'p1_n_pokemon_use', 'diff_spd', 'battle_duration', 'diff_hp', 'net_major_status_suffering', 'priority_rate_advantage', 'p1_bad_status_advantage', 'p2_pct_final_hp', 'p1_type_diversity', 'expected_damage_ratio_turn_1', 'p1_type_resistance', 'diff_mean_stab', 'p1_max_speed_offense_product', 'early_hp_mean_diff', 'diff_final_hp', 'p2_status_change', 'diff_final_schieramento', 'diff_atk', 'p1_type_weakness', 'std_base_spe_diff_timeline', 'p2_type_advantage', 'p1_major_status_infliction_rate', 'p1_avg_speed_stat_battaglia', 'p1_mean_hp', 'nr_pokemon_sconfitti_diff', 'hp_delta_trend', 'p1_type_advantage', 'mean_base_atk_diff_timeline', 'p1_mean_atk', 'net_major_status_infliction', 'p1_final_hp_per_ko', 'priority_diff', 'late_hp_mean_diff', 'p2_major_status_infliction_rate', 'hp_advantage_trend']
Bucket   1 → robust CV=0.8421 (61 features)
Bucket 1 took 160.883061170578 time
Bucket   2 → robust CV=0.8419 (56 features)
Bucket 2 took 157.48382377624512 time
Bucket   3 → robust CV=0.8424 (63 features)
Bucket 3 took 158.44449996948242 time
Bucket   4 → robust CV=0.8419 (62 features)
Bucket 4 took 169.51906204223633 time
Bucket   5 → robust CV=0.8422 (55 features)
Bucket 5 took 162.50981402397156 time
Bucket   6 → robust CV=0.8427 (54 features)
Bucket 6 took 161.67177295684814 time
Bucket   7 → robust CV=0.8424 (56 features)
Bucket 7 took 167.01482009887695 time
Bucket   8 → robust CV=0.8418 (65 features)
Bucket 8 took 158.23124980926514 time
Bucket   9 → robust CV=0.8419 (64 features)
Bucket 9 took 159.6353840827942 time
Bucket  10 → robust CV=0.8420 (63 features)
Bucket 10 took 164.61752915382385 time
Bucket  11 → robust CV=0.8420 (61 features)
Bucket 11 took 161.99736428260803 time
Bucket  12 → robust CV=0.8423 (61 features)
Bucket 12 took 155.11795496940613 time
Bucket  13 → robust CV=0.8424 (61 features)
Bucket 13 took 163.80495405197144 time
 Bucket  14 → robust CV=0.8417 (66 features)
Bucket 14 took 165.03190660476685 time
Bucket  15 → robust CV=0.8426 (62 features)
Bucket 15 took 161.02220511436462 time
Bucket  16 → robust CV=0.8418 (49 features)
Bucket 16 took 168.58466506004333 time
Bucket  17 → robust CV=0.8421 (65 features)
Bucket 17 took 171.20154404640198 time
Bucket  18 → robust CV=0.8420 (57 features)
Bucket 18 took 170.60114908218384 time
Bucket  19 → robust CV=0.8422 (58 features)
Bucket 19 took 173.15379929542542 time
Bucket  20 → robust CV=0.8418 (49 features)
Bucket 20 took 167.95985913276672 time
Bucket  21 → robust CV=0.8424 (56 features)
Bucket 21 took 170.16775918006897 time
Bucket  22 → robust CV=0.8421 (63 features)
Bucket 22 took 168.14446091651917 time
Bucket  23 → robust CV=0.8421 (65 features)
Bucket 23 took 170.57118105888367 time
Bucket  24 → robust CV=0.8417 (66 features)
Bucket 24 took 169.88067770004272 time
Bucket  25 → robust CV=0.8422 (60 features)
Bucket 25 took 168.13657784461975 time
Bucket  26 → robust CV=0.8418 (60 features)
Bucket 26 took 175.831444978714 time
Bucket  27 → robust CV=0.8418 (65 features)
Bucket 27 took 163.4392249584198 time
Bucket  28 → robust CV=0.8426 (41 features)
Bucket 28 took 172.820387840271 time
Bucket  29 → robust CV=0.8418 (65 features)
Bucket 29 took 170.04541277885437 time
Bucket  30 → robust CV=0.8420 (55 features)
Bucket 30 took 165.16519021987915 time
Bucket  31 → robust CV=0.8422 (52 features)
Bucket 31 took 170.01101112365723 time
Bucket  32 → robust CV=0.8420 (55 features)
Bucket 32 took 167.45506501197815 time
Bucket  33 → robust CV=0.8420 (62 features)
Bucket 33 took 167.37723398208618 time
Bucket  34 → robust CV=0.8420 (51 features)
Bucket 34 took 161.42051696777344 time
Bucket  35 → robust CV=0.8419 (58 features)
Bucket 35 took 160.17681169509888 time
Bucket  36 → robust CV=0.8420 (65 features)
Bucket 36 took 168.85392808914185 time
Bucket  37 → robust CV=0.8420 (54 features)
Bucket 37 took 163.59460711479187 time
Bucket  38 → robust CV=0.8419 (65 features)
Bucket 38 took 157.56218004226685 time
Bucket  39 → robust CV=0.8421 (65 features)
Bucket 39 took 166.37104105949402 time
Bucket  40 → robust CV=0.8421 (61 features)
Bucket 40 took 160.9918930530548 time
Bucket  41 → robust CV=0.8421 (54 features)
Bucket 41 took 161.81665706634521 time
Bucket  42 → robust CV=0.8417 (66 features)
Bucket 42 took 159.0308620929718 time
Bucket  43 → robust CV=0.8423 (44 features)
Bucket 43 took 159.79427003860474 time
Bucket  44 → robust CV=0.8423 (60 features)
Bucket 44 took 162.14478588104248 time
Bucket  45 → robust CV=0.8425 (57 features)
Bucket 45 took 158.715096950531 time
Bucket  46 → robust CV=0.8420 (51 features)
Bucket 46 took 165.44257402420044 time
Bucket  47 → robust CV=0.8421 (58 features)
Bucket 47 took 158.14838290214539 time
Bucket  48 → robust CV=0.8419 (63 features)
Bucket 48 took 159.72496604919434 time
Bucket  49 → robust CV=0.8420 (65 features)
Bucket 49 took 159.94962787628174 time
Bucket  50 → robust CV=0.8422 (63 features)
Bucket 50 took 157.973806142807 time

🏆 Best bucket found:
Score: 0.8427 with 54 features
Top features: ['p1_max_offense_boost_diff', 'nr_pokemon_sconfitti_p2', 'p1_bad_status_advantage', 'net_major_status_suffering', 'p1_mean_stab', 'std_base_atk_diff_timeline', 'mean_base_spe_diff_timeline', 'priority_rate_advantage', 'diff_hp', 'p1_avg_high_speed_stat_battaglia', 'p2_status_change', 'p2_major_status_infliction_rate', 'p2_mean_stab', 'battle_duration', 'p1_status_change', 'hp_delta_trend', 'status_change_diff', 'hp_loss_rate', 'p1_max_speed_offense_product', 'diff_type_advantage', 'mean_base_atk_diff_timeline', 'p2_n_pokemon_use', 'diff_spd', 'p1_mean_def', 'p1_type_weakness', 'p1_type_resistance', 'hp_delta_std', 'p2_type_advantage', 'std_base_spa_diff_timeline', 'nr_pokemon_sconfitti_diff', 'mean_base_spa_diff_timeline', 'p2_cumulative_major_status_turns_pct', 'p1_mean_sp', 'diff_atk', 'p1_type_advantage', 'p1_mean_atk', 'p2_pct_final_hp', 'early_hp_mean_diff', 'priority_diff', 'diff_final_schieramento', 'p1_max_speed_stat', 'net_major_status_infliction', 'p1_type_diversity', 'p1_mean_spe', 'expected_damage_ratio_turn_1', 'p1_team_super_effective_moves', 'p1_max_offensive_stat', 'diff_def', 'p1_n_pokemon_use', 'p1_cumulative_major_status_turns_pct', 'p1_hp_advantage_mean', 'diff_spe', 'p2_hp_std', 'diff_final_hp']
Best features with C=30:['p1_max_offense_boost_diff', 'nr_pokemon_sconfitti_p2', 'p1_bad_status_advantage', 'net_major_status_suffering', 'p1_mean_stab', 'std_base_atk_diff_timeline', 'mean_base_spe_diff_timeline', 'priority_rate_advantage', 'diff_hp', 'p1_avg_high_speed_stat_battaglia', 'p2_status_change', 'p2_major_status_infliction_rate', 'p2_mean_stab', 'battle_duration', 'p1_status_change', 'hp_delta_trend', 'status_change_diff', 'hp_loss_rate', 'p1_max_speed_offense_product', 'diff_type_advantage', 'mean_base_atk_diff_timeline', 'p2_n_pokemon_use', 'diff_spd', 'p1_mean_def', 'p1_type_weakness', 'p1_type_resistance', 'hp_delta_std', 'p2_type_advantage', 'std_base_spa_diff_timeline', 'nr_pokemon_sconfitti_diff', 'mean_base_spa_diff_timeline', 'p2_cumulative_major_status_turns_pct', 'p1_mean_sp', 'diff_atk', 'p1_type_advantage', 'p1_mean_atk', 'p2_pct_final_hp', 'early_hp_mean_diff', 'priority_diff', 'diff_final_schieramento', 'p1_max_speed_stat', 'net_major_status_infliction', 'p1_type_diversity', 'p1_mean_spe', 'expected_damage_ratio_turn_1', 'p1_team_super_effective_moves', 'p1_max_offensive_stat', 'diff_def', 'p1_n_pokemon_use', 'p1_cumulative_major_status_turns_pct', 'p1_hp_advantage_mean', 'diff_spe', 'p2_hp_std', 'diff_final_hp']
Bucket   1 → robust CV=0.8420 (56 features)
Bucket 1 took 161.62166690826416 time
Bucket   2 → robust CV=0.8417 (62 features)
Bucket 2 took 162.73248505592346 time
Bucket   3 → robust CV=0.8422 (58 features)
Bucket 3 took 160.69101810455322 time
Bucket   4 → robust CV=0.8417 (65 features)
Bucket 4 took 157.51137900352478 time
Bucket   5 → robust CV=0.8428 (42 features)
Bucket 5 took 161.3370099067688 time
Bucket   6 → robust CV=0.8420 (60 features)
Bucket 6 took 157.64560294151306 time
Bucket   7 → robust CV=0.8424 (54 features)
Bucket 7 took 158.22573685646057 time
Bucket   8 → robust CV=0.8419 (60 features)
Bucket 8 took 158.8424150943756 time
Bucket   9 → robust CV=0.8422 (55 features)
Bucket 9 took 162.848317861557 time
Bucket  10 → robust CV=0.8416 (63 features)
Bucket 10 took 160.13517379760742 time
Bucket  11 → robust CV=0.8422 (62 features)
Bucket 11 took 166.95503497123718 time
Bucket  12 → robust CV=0.8420 (57 features)
Bucket 12 took 166.8625190258026 time
Bucket  13 → robust CV=0.8418 (62 features)
Bucket 13 took 157.4398889541626 time
Bucket  14 → robust CV=0.8419 (63 features)
Bucket 14 took 159.15538883209229 time
Bucket  15 → robust CV=0.8416 (53 features)
Bucket 15 took 158.02293872833252 time
Bucket  16 → robust CV=0.8423 (40 features)
Bucket 16 took 163.19240021705627 time
Bucket  17 → robust CV=0.8426 (40 features)
Bucket 17 took 157.55909514427185 time
Bucket  18 → robust CV=0.8416 (63 features)
Bucket 18 took 159.8713300228119 time
Bucket  19 → robust CV=0.8414 (65 features)
Bucket 19 took 159.83344316482544 time
Bucket  20 → robust CV=0.8417 (64 features)
Bucket 20 took 164.0521697998047 time
Bucket  21 → robust CV=0.8420 (56 features)
Bucket 21 took 158.18799686431885 time
Bucket  22 → robust CV=0.8422 (59 features)
Bucket 22 took 156.95113801956177 time
Bucket  23 → robust CV=0.8416 (59 features)
Bucket 23 took 161.91801190376282 time
Bucket  24 → robust CV=0.8420 (57 features)
Bucket 24 took 163.08732104301453 time
"""