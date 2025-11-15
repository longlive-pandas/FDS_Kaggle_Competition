
import json

def read_train_data(train_file_path):
    train_data = []
    try:
        with open(train_file_path, "r") as f:
            for line in f:
                train_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Could not find the training file at {train_file_path}.")
        print("Please make sure you have added the competition data to this notebook.")
    finally:
        return train_data
def read_test_data(test_file_path):
    test_data = []
    with open(test_file_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data
