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
train_df.head().to_json("train_df_head.json", orient="records", indent=2)

features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
# print(f"Using {len(features)} features")
X = train_df[features]
y = train_df['player_won']

#3. TRAIN

selected = features#['p1_max_offensive_stat', 'p1_mean_sp', 'p1_cumulative_major_status_turns_pct', 'diff_def', 'mean_base_atk_diff_timeline', 'p1_n_pokemon_use', 'p1_hp_std', 'diff_atk', 'p2_cumulative_major_status_turns_pct', 'p1_bad_status_advantage', 'battle_duration', 'p1_mean_def', 'p1_avg_speed_stat_battaglia', 'mean_base_spa_diff_timeline', 'late_hp_mean_diff', 'nr_pokemon_sconfitti_p1', 'p1_type_weakness', 'p1_avg_high_speed_stat_battaglia', 'p1_pct_final_hp', 'p1_hp_advantage_mean', 'std_base_spa_diff_timeline', 'hp_diff_mean', 'std_base_spe_diff_timeline', 'diff_hp', 'p2_n_pokemon_use', 'p1_mean_atk', 'p2_hp_std', 'p2_major_status_infliction_rate', 'mean_base_spe_diff_timeline', 'hp_advantage_trend', 'p1_type_resistance', 'nr_pokemon_sconfitti_p2', 'priority_diff', 'diff_final_schieramento', 'p1_type_diversity', 'p1_mean_spe', 'p1_major_status_infliction_rate', 'expected_damage_ratio_turn_1', 'hp_loss_rate']
#now that I know which features are best I fit my model to these and submit
#selected = ['net_major_status_infliction', 'battle_duration', 'p1_max_offense_boost_diff', 'diff_final_hp', 'std_base_spa_diff_timeline', 'nr_pokemon_sconfitti_p1', 'p1_avg_high_speed_stat_battaglia', 'diff_spe', 'expected_damage_ratio_turn_1', 'hp_loss_rate', 'p2_cumulative_major_status_turns_pct', 'p1_mean_spe', 'p1_type_resistance', 'p1_max_speed_offense_product', 'nr_pokemon_sconfitti_p2', 'p2_hp_std', 'net_major_status_suffering', 'p2_n_pokemon_use', 'p1_type_weakness', 'diff_mean_stab', 'p1_mean_sp', 'p1_cumulative_major_status_turns_pct', 'p1_mean_def', 'hp_delta_std', 'mean_base_spa_diff_timeline', 'p1_n_pokemon_use', 'p1_mean_stab', 'diff_spd', 'p2_mean_stab', 'diff_hp', 'mean_base_atk_diff_timeline', 'p1_type_diversity', 'status_change_diff', 'diff_type_advantage', 'priority_diff', 'std_base_spe_diff_timeline', 'diff_final_schieramento', 'p1_mean_hp', 'p1_pct_final_hp', 'p2_pct_final_hp', 'p2_major_status_infliction_rate', 'p1_hp_advantage_mean', 'p1_max_speed_stat', 'diff_atk', 'nr_pokemon_sconfitti_diff', 'hp_advantage_trend', 'std_base_atk_diff_timeline', 'hp_delta_trend', 'late_hp_mean_diff', 'p1_bad_status_advantage', 'p1_major_status_infliction_rate']

X_selected = X[selected]
print(f"selected shape={X_selected.shape}")
final_pipe = train_regularization(X_selected,y)

extracted_features_and_weights = extract_features_by_importance(final_pipe, selected)
print(f"{len(selected)} features - extracted_features_and_weights under linearity assumption: {extracted_features_and_weights}")
# with open("extracted_features_and_weights.txt", "w") as f:
#     f.write(extracted_features_and_weights.to_string())
extracted_features_and_weights.to_csv("extracted_features_and_weights.csv", index=False)
#final_pipe = simple_train(X_selected,y)#creates and fits pipe
import json
with open("features_list_model83.33.json", "w") as f:
    json.dump(selected, f)
# Tune threshold on the training data
predict_and_submit(test_df, features, final_pipe)