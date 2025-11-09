import os

from start_utils import create_features, read_train_data, read_test_data, simple_train, build_pipe, train_regularization, greedy_feature_selection, random_bucket_feature_search, predict_and_submit

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
test_df = create_features(test_data)
    
print("\nTraining features preview:")
#display(train_df.head())
train_df.head().to_json("train_df_head.json", orient="records", indent=2)
#train_df.describe()

# Define features and target
#features = ['p1_type_diversity', 'p1_type_resistance', 'p1_type_weakness']
#features = ['diff_mean_stab']
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
print(f"Using {len(features)} features")
X = train_df[features]
y = train_df['player_won']

#3. TRAIN
#pipe = 
# simple_train(X,y)

# ###GREEDY
# pipe = build_pipe()
# selected, history = greedy_feature_selection(X, y, pipe, cv=5, min_delta=0.0005)

# print("\n✅ Selected features:")
# print(selected)
# print("\n📈 Accuracy progression:")
# print(history)

# # Run random bucket search
# """
# n_buckets 100
# bucket_size 25
# Best bucket found:
# Score: 0.8396 with 22 features
# Top features: ['p1_hp_std', 'mean_base_spa_diff_timeline', 'hp_delta_trend', 'hp_delta_std', 'p1_mean_def', 'diff_hp', 'mean_base_atk_diff_timeline', 'p2_pct_final_hp', 'p1_mean_spe', 'p1_mean_atk', 'diff_final_schieramento', 'p1_mean_hp', 'p1_hp_advantage_mean', 'p1_type_resistance', 'nr_pokemon_sconfitti_p1', 'hp_advantage_trend', 'p1_mean_sp', 'status_change_diff', 'p1_n_pokemon_use', 'p1_bad_status_advantage', 'p1_pct_final_hp', 'hp_loss_rate']
# """
# pipe = build_pipe()
# bucket_size = 25
# res = random_bucket_feature_search(X, y, pipe, n_buckets=100, bucket_size=bucket_size, try_subsets=True)

# print("\nBest score:", res["best_score"])
# print("Selected features:", res["best_features"])

# pipe = build_pipe()
# bucket_size = 35
# res = random_bucket_feature_search(X, y, pipe, n_buckets=200, bucket_size=bucket_size, try_subsets=True)

# print("\nBest score:", res["best_score"])
# print("Selected features:", res["best_features"])

# ###
# pipe = build_pipe()
# bucket_size = len(features)#25
# res = random_bucket_feature_search(X, y, pipe, n_buckets=300, bucket_size=bucket_size, try_subsets=True)

# print("\nBest score:", res["best_score"])
# print("Selected features:", res["best_features"])
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
X_selected = X[selected]
final_pipe = train_regularization(X_selected,y)
#final_pipe = simple_train(X_selected,y)#creates and fits pipe
predict_and_submit(test_df, selected, final_pipe)