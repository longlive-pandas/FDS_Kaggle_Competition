import os
import time
import logging
from reading import (
    read_train_data,
    read_test_data
)
from feature_engineering import create_features
from train import (
    train_regularization,
    train_with_feature_selection,
    correlation_pruning,
    fit_predict_submit,
    build_stacking_model
)


if __name__ == '__main__':
    ###############  INITIAL VARIABLES AND CONFIGS
    logging.basicConfig(
        filename='classification.log',
        filemode='w',            
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    start_time = time.time()
    
    ###############  DATA LOADING
    COMPETITION_NAME = "fds-pokemon-battles-prediction-2025"
    DATA_PATH = os.path.join("input", COMPETITION_NAME)
    train_file_path = os.path.join(DATA_PATH, "train.jsonl")
    test_file_path = os.path.join(DATA_PATH, "test.jsonl")
    train_data = read_train_data(train_file_path)
    test_data = read_test_data(test_file_path)
    ###############  FEATURES CREATION
    train_df = create_features(train_data)
    test_df = create_features(test_data)
    features = [col for col in train_df.columns if col not in ["battle_id", "player_won"]]
    X = train_df[features]
    y = train_df["player_won"]
    ###############  TRAIN
    middle_time = time.time()
    #LOGISTIC:
    print("LOGISTIC MODEL")
    selected = features#alternatively choose a subset of the features
    X_selected = X[selected]
    model = train_regularization(X_selected,y)
    fit_predict_submit(model, features, X, y, start_time, middle_time, test_df, prefix="logistic")

    #VOTING:
    print("VOTING MODEL")
    model, features, importances_table = train_with_feature_selection(
        X, y, k=80
    )
    X_reduced = X[features]
    features = correlation_pruning(X_reduced, threshold=0.92)
    print("\nModello finale pronto!")
    fit_predict_submit(model, features, X, y, start_time, middle_time, test_df, prefix= "voting")
    
    #STACKING
    print("STACKING MODEL")
    model = build_stacking_model()
    fit_predict_submit(model, features, X, y, start_time, middle_time, test_df, prefix= "stacking")