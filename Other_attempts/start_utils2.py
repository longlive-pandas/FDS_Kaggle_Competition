import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import linregress
import json
#train
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score

type_chart = {
    "normal":     {"rock":0.5, "ghost":0.0, "steel":0.5},
    "fire":       {"fire":0.5, "water":0.5, "grass":2.0, "ice":2.0, "bug":2.0, "rock":0.5, "dragon":0.5, "steel":2.0},
    "water":      {"fire":2.0, "water":0.5, "grass":0.5, "ground":2.0, "rock":2.0, "dragon":0.5},
    "electric":   {"water":2.0, "electric":0.5, "grass":0.5, "ground":0.0, "flying":2.0, "dragon":0.5},
    "grass":      {"fire":0.5, "water":2.0, "grass":0.5, "poison":0.5, "ground":2.0, "flying":0.5, "bug":0.5, "rock":2.0, "dragon":0.5, "steel":0.5},
    "ice":        {"fire":0.5, "water":0.5, "grass":2.0, "ice": 0.5, "ground":2.0, "flying":2.0, "dragon":2.0, "steel":0.5},
    "fighting":   {"normal":2.0, "ice":2.0, "poison":0.5, "flying":0.5, "psychic":0.5, "bug":0.5, "rock":2.0, "ghost":0.0, "dark":2.0, "steel":2.0, "fairy":0.5},
    "poison":     {"grass":2.0, "poison":0.5, "ground":0.5, "rock":0.5, "ghost":0.5, "steel":0.0, "fairy":2.0},
    "ground":     {"fire":2.0, "electric":2.0, "grass":0.5, "poison":2.0, "flying":0.0, "bug":0.5, "rock":2.0, "steel":2.0},
    "flying":     {"electric":0.5, "grass":2.0, "fighting":2.0, "bug":2.0, "rock":0.5, "steel":0.5},
    "psychic":    {"fighting":2.0, "poison":2.0, "psychic":0.5, "dark":0.0, "steel":0.5},
    "bug":        {"fire":0.5, "grass":2.0, "fighting":0.5, "poison":0.5, "flying":0.5, "psychic":2.0, "ghost":0.5, "dark":2.0, "steel":0.5, "fairy":0.5},
    "rock":       {"fire":2.0, "ice":2.0, "fighting":0.5, "ground":0.5, "flying":2.0, "bug":2.0, "steel":0.5},
    "ghost":      {"normal":0.0, "psychic":2.0, "ghost":2.0, "dark":0.5},
    "dragon":     {"dragon":2.0, "steel":0.5, "fairy":0.0},
    "dark":       {"fighting":0.5, "psychic":2.0, "ghost":2.0, "dark": 0.5, "fairy":0.5},
    "steel":      {"fire":0.5, "water":0.5, "electric":0.5, "ice":2.0, "rock":2.0, "steel":0.5, "fairy":2.0},
    "fairy":      {"fire":0.5, "fighting":2.0, "poison":0.5, "dragon":2.0, "dark":2.0, "steel":0.5}
}
def read_train_data(train_file_path):
    train_data = []
    # Read the file line by line
    try:
        with open(train_file_path, 'r') as f:
            for line in f:
                # json.loads() parses one line (one JSON object) into a Python dictionary
                train_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Could not find the training file at '{train_file_path}'.")
        print("Please make sure you have added the competition data to this notebook.")
    finally:
        return train_data

def read_test_data(test_file_path):
    test_data = []
    with open(test_file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data

def compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart, is_test=False, battle_id=''):
    if not timeline:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }

    p1_advantages = []
    p2_advantages = []
    for turn in timeline:
        p1_name = turn.get("p1_pokemon_state", {}).get("name")
        p2_name = turn.get("p2_pokemon_state", {}).get("name")

        if not p1_name or not p2_name:
            continue

        # Get types from dictionary
        p1_types = pokemon_dict.get(p1_name.lower(), [])
        p2_types = pokemon_dict.get(p2_name.lower(), [])
        #print(len(p1_types),len(p2_types))
        if not p1_types or not p2_types:
            continue
        # --- P1 attacking P2 ---
        p1_mult = []
        for atk_type in p1_types:
            mult = 1.0
            if atk_type.upper() not in type_chart or def_type.upper() not in type_chart.get(atk_type.upper(), {}):
                print("error")
                return
            for def_type in p2_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p1_mult.append(mult)
        #turn summary
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0

        # --- P2 attacking P1 ---
        p2_mult = []
        for atk_type in p2_types:
            mult = 1.0
            if atk_type.upper() not in type_chart or def_type.upper() not in type_chart.get(atk_type.upper(), {}):
                print("error")
                return
            for def_type in p1_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p2_mult.append(mult)
        #turn summary
        p2_adv = np.mean(p2_mult) if p2_mult else 1.0

        p1_advantages.append(p1_adv)
        p2_advantages.append(p2_adv)

    if not p1_advantages or not p2_advantages:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }

    #timeline summary
    p1_avg = np.mean(p1_advantages)
    p2_avg = np.mean(p2_advantages)
    # if is_test and battle_id == 109:
    #     print(f"first team:{debug_dict_p1}, second team:{debug_dict_p2}")
    #     #,p1_avg,p2_avg,p1_avg - p2_avg
    #     exit()
    return {
        "p1_type_advantage": p1_avg,
        "p2_type_advantage": p2_avg,
        "diff_type_advantage": p1_avg - p2_avg
    }
"""
{
    "p1_team_details(6)": [
        {
            "name": "starmie",
            "level": 100,
            "types": [
                "psychic",
                "water|notype"
            ],
            "base_hp": 60,
            "base_atk": 75,
            "base_def": 85,
            "base_spa": 100,
            "base_spd": 100,
            "base_spe": 115
        }
    ],
    "p2_lead_details": {
        "name": "starmie",
        "level": 100,
        "types": [
            "psychic",
            "water"
        ],
        "base_hp": 60,
        "base_atk": 75,
        "base_def": 85,
        "base_spa": 100,
        "base_spd": 100,
        "base_spe": 115
    },
    "battle_timeline(30)": [
        {
            "turn": 1,
            "p1_pokemon_state|p2_pokemon_state": {
                "name": "starmie",
                "hp_pct": 1.0,
                "status": "nostatus|frz",
                "effects": [
                    "noeffect"
                ],
                "boosts": {
                    "atk": 0,
                    "def": 0,
                    "spa": 0,
                    "spd": 0,
                    "spe": 0
                }
            },
            "p1_move_details|p2_move_details": {
                "name": "icebeam",
                "type": "ICE",
                "category": "SPECIAL",
                "base_power": 95,
                "accuracy": 1.0,
                "priority": 0
            }
        }
    ]
}

"""
def create_features(data: list[dict], is_test=False) -> pd.DataFrame:
    feature_list = []
    #p1_bad_status_advantage = []
    status_change_diff = []
    #restyle build dictionary pokemon => types
    pokemon_dict = {}
    for battle in data:
        p1_team = battle.get('p1_team_details', [])
        for p in p1_team:
            name = p.get("name")
            types = [t for t in p.get("types", []) if t != "notype"]
            if name:
                if name not in pokemon_dict:
                    pokemon_dict[name] = set()
                pokemon_dict[name].update(types)
        p2_lead = battle.get('p2_lead_details')
        if p2_lead:
            name = p2_lead.get("name")
            types = [t for t in p2_lead.get("types", []) if t != "notype"]
            if name:
                if name not in pokemon_dict:
                    pokemon_dict[name] = set()
                pokemon_dict[name].update(types)
    for battle in tqdm(data, desc="Extracting features"):
        battle_id = battle.get("battle_id")
        features = {}
        
        features['battle_id'] = battle_id
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)
    return pd.DataFrame(feature_list).fillna(0)




COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('input', COMPETITION_NAME)

train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')

train_data = read_train_data(train_file_path)
test_data = read_test_data(test_file_path)

train_df = create_features(train_data)
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,  
    random_state=1234,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,      
    learning_rate=0.03,
    max_depth=3, 
    random_state=1234
)

stacked_model = StackingClassifier(
    estimators=[
        ('rf', rf),         
        ('gb', gb)          
    ],
    final_estimator=LogisticRegression(
        max_iter=2000, 
        C=0.05, 
        random_state=1234
    ), 
    passthrough=False, 
    n_jobs=-1
)


"""
['status_change_diff', 'diff_final_hp'],0.8276, 0.90783996, 0.8161 ± 0.0088, 0.8902 ± 0.0083
['status_change_diff', 'diff_final_hp', 'nr_pokemon_sconfitti_diff'],0.8289, 0.90820228, 0.8174 ± 0.0108, 0.8921 ± 0.0085
['status_change_diff', 'diff_final_hp', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean'],0.8369, 0.91563674, 0.8195 ± 0.0112, 0.8934 ± 0.0083
['status_change_diff', 'diff_final_hp', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 'p1_n_pokemon_use'],0.836, 0.9148296400000001, 0.8182 ± 0.0111, 0.8935 ± 0.0084

['status_change_diff', 'diff_final_hp', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 'p1_n_pokemon_use'],0.8268, 0.9028299199999998, 0.8188 ± 0.0096, 0.8928 ± 0.0081
['status_change_diff', 'diff_final_hp', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 'p1_n_pokemon_use', 'p1_type_advantage'],0.8278, 0.90318476, 0.8186 ± 0.0102, 0.8929 ± 0.0083

['diff_final_hp', 'status_change_diff', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 'p1_n_pokemon_use', 'p1_type_advantage'],0.8277, 0.90266646, 0.8174 ± 0.0105, 0.8922 ± 0.0083
['diff_final_hp', 'status_change_diff', 'nr_pokemon_sconfitti_diff', 'hp_diff_mean', 'p1_n_pokemon_use', 'p1_type_advantage', 'net_stat_boost_advantage'],0.8255, 0.9022922800000001, 0.8171 ± 0.0112, 0.8918 ± 0.0082
"""
selected = features
X_selected = X[selected]
stacked_model.fit(X_selected, y)
final_pipe = stacked_model
#EVALUATE

y_train_pred = final_pipe.predict(X_selected)
y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]

#CHECK OVERFITTING
acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='accuracy')
auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='roc_auc')
print(f"{[f for f in selected]},{accuracy_score(y, y_train_pred)}, {roc_auc_score(y, y_train_proba)}, {acc.mean():.4f} ± {acc.std():.4f}, {auc.mean():.4f} ± {auc.std():.4f}")

#predict_and_submit(test_df, selected, final_pipe)