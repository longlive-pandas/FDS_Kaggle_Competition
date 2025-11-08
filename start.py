import os

from start_utils import create_features, read_train_data, read_test_data, train, train_regularization, predict_and_submit

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
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
print(f"Using {len(features)} features")
X = train_df[features]
y = train_df['player_won']

#3. TRAIN
#pipe = train(X,y)
pipe = train_regularization(X,y)
#4 SUBMIT
predict_and_submit(test_df, features, pipe)