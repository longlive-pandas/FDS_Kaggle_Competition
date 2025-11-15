import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import linregress
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    RandomForestClassifier, 
    StackingClassifier,
    VotingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from typing import Dict, Iterable
import random
from xgboost import XGBClassifier
import time
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import itertools
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import linregress
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
import os
import random
import time
from typing import Dict, Any
start_time = time.time()
STAT_FIELDS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
type_chart = {
    "NORMAL":     {"ROCK":0.5, "GHOST":0.0, "STEEL":0.5},
    "FIRE":       {"FIRE":0.5, "WATER":0.5, "GRASS":2.0, "ICE":2.0, "BUG":2.0, "ROCK":0.5, "DRAGON":0.5, "STEEL":2.0},
    "WATER":      {"FIRE":2.0, "WATER":0.5, "GRASS":0.5, "GROUND":2.0, "ROCK":2.0, "DRAGON":0.5},
    "ELECTRIC":   {"WATER":2.0, "ELECTRIC":0.5, "GRASS":0.5, "GROUND":0.0, "FLYING":2.0, "DRAGON":0.5},
    "GRASS":      {"FIRE":0.5, "WATER":2.0, "GRASS":0.5, "POISON":0.5, "GROUND":2.0, "FLYING":0.5, "BUG":0.5, "ROCK":2.0, "DRAGON":0.5, "STEEL":0.5},
    "ICE":        {"FIRE":0.5, "WATER":0.5, "GRASS":2.0, "ICE": 0.5, "GROUND":2.0, "FLYING":2.0, "DRAGON":2.0, "STEEL":0.5},
    "FIGHTING":   {"NORMAL":2.0, "ICE":2.0, "POISON":0.5, "FLYING":0.5, "PSYCHIC":0.5, "BUG":0.5, "ROCK":2.0, "GHOST":0.0, "DARK":2.0, "STEEL":2.0, "FAIRY":0.5},
    "POISON":     {"GRASS":2.0, "POISON":0.5, "GROUND":0.5, "ROCK":0.5, "GHOST":0.5, "STEEL":0.0, "FAIRY":2.0},
    "GROUND":     {"FIRE":2.0, "ELECTRIC":2.0, "GRASS":0.5, "POISON":2.0, "FLYING":0.0, "BUG":0.5, "ROCK":2.0, "STEEL":2.0},
    "FLYING":     {"ELECTRIC":0.5, "GRASS":2.0, "FIGHTING":2.0, "BUG":2.0, "ROCK":0.5, "STEEL":0.5},
    "PSYCHIC":    {"FIGHTING":2.0, "POISON":2.0, "PSYCHIC":0.5, "DARK":0.0, "STEEL":0.5},
    "BUG":        {"FIRE":0.5, "GRASS":2.0, "FIGHTING":0.5, "POISON":0.5, "FLYING":0.5, "PSYCHIC":2.0, "GHOST":0.5, "DARK":2.0, "STEEL":0.5, "FAIRY":0.5},
    "ROCK":       {"FIRE":2.0, "ICE":2.0, "FIGHTING":0.5, "GROUND":0.5, "FLYING":2.0, "BUG":2.0, "STEEL":0.5},
    "GHOST":      {"NORMAL":0.0, "PSYCHIC":2.0, "GHOST":2.0, "DARK":0.5},
    "DRAGON":     {"DRAGON":2.0, "STEEL":0.5, "FAIRY":0.0},
    "DARK":       {"FIGHTING":0.5, "PSYCHIC":2.0, "GHOST":2.0, "DARK": 0.5, "FAIRY":0.5},
    "STEEL":      {"FIRE":0.5, "WATER":0.5, "ELECTRIC":0.5, "ICE":2.0, "ROCK":2.0, "STEEL":0.5, "FAIRY":2.0},
    "FAIRY":      {"FIRE":0.5, "FIGHTING":2.0, "POISON":0.5, "DRAGON":2.0, "DARK":2.0, "STEEL":0.5}
}
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
COMPETITION_NAME = "fds-pokemon-battles-prediction-2025"
DATA_PATH = os.path.join("input", COMPETITION_NAME)
train_file_path = os.path.join(DATA_PATH, "train.jsonl")
test_file_path = os.path.join(DATA_PATH, "test.jsonl")
train_data = read_train_data(train_file_path)
test_data = read_test_data(test_file_path)
######FEATURES UTILITIES/HELPERS
def calculate_interaction_features(features: Dict[str, float]) -> Dict[str, float]:
    # Offensive Control Differential (Net Infliction)
    p1_inflict = features.get("p1_major_status_infliction_rate", 0.0)
    p2_inflict = features.get("p2_major_status_infliction_rate", 0.0)
    features["net_major_status_infliction"] = p1_inflict - p2_inflict
    # Suffering Differential (Net Time Crippled)
    p1_suffered = features.get("p1_cumulative_major_status_turns_pct", 0.0)
    p2_suffered = features.get("p2_cumulative_major_status_turns_pct", 0.0)
    features["net_major_status_suffering"] = p2_suffered - p1_suffered
    # Speed/Damage Interaction (Fast Sweeper Potential)
    p1_max_spe = features.get("p1_max_speed_stat", 0.0)
    p1_max_off = features.get("p1_max_offensive_stat", 0.0)
    features["p1_max_speed_offense_product"] = p1_max_spe * p1_max_off
    # Final HP per KO Ratio (Adding +1 to avoid division by zero)
    p1_final_hp = features.get("p1_pct_final_hp", 0.0)
    p1_ko_count = features.get("nr_pokemon_sconfitti_p1", 0)
    features["p1_final_hp_per_ko"] = p1_final_hp / (p1_ko_count + 1)
    return features
def calculate_status_efficacy_features(battle: Dict[str, Any]) -> Dict[str, float]:
    features = {}
    timeline = battle.get("battle_timeline", [])
    if not timeline:
        return {
            "p1_major_status_infliction_rate": 0.0,
            "p1_cumulative_major_status_turns_pct": 0.0
        }
    MAJOR_STATUSES = {"slp", "frz"} 
    MAJOR_STATUS_MOVES = {"sleeppowder", "spore", "lovely kiss", "sing"}
    p1_major_status_attempts = 0
    p1_major_status_successes = 0
    p1_major_status_turns_suffered = 0
    total_turns = len(timeline)
    for turn in timeline:
        #Infliction Rate (P1 trying to hit P2)
        p1_move = turn.get("p1_move_details")
        p2_state = turn.get("p2_pokemon_state", {})
        p2_current_status = p2_state.get("status", "nostatus")
        if p1_move:
            move_name = p1_move.get("name", "").lower()
            # Check for direct major status moves
            if move_name in MAJOR_STATUS_MOVES:
                p1_major_status_attempts += 1
                # Check if P2 ended the turn with a major status
                if p2_current_status in MAJOR_STATUSES:
                    p1_major_status_successes += 1
        #Cumulative Status Turns Suffered (P1 suffering)
        p1_state = turn.get("p1_pokemon_state", {})
        p1_current_status = p1_state.get("status", "nostatus")
        if p1_current_status in MAJOR_STATUSES:
            p1_major_status_turns_suffered += 1
    p1_major_status_infliction_rate = 0.0
    if p1_major_status_attempts > 0:
        p1_major_status_infliction_rate = p1_major_status_successes / p1_major_status_attempts
    features["p1_major_status_infliction_rate"] = p1_major_status_infliction_rate
    p1_cumulative_major_status_turns_pct = 0.0
    if total_turns > 0:
        p1_cumulative_major_status_turns_pct = p1_major_status_turns_suffered / total_turns
    features["p1_cumulative_major_status_turns_pct"] = p1_cumulative_major_status_turns_pct
        
    return features
def calculate_p2_status_control_features(battle: Dict[str, Any]) -> Dict[str, float]:
    features = {}
    timeline = battle.get("battle_timeline", [])
    if not timeline:
        return {
            "p2_major_status_infliction_rate": 0.0,
            "p2_cumulative_major_status_turns_pct": 0.0
        }
    MAJOR_STATUSES = {"slp", "frz"} 
    # Common Gen 1 status moves that inflict major status
    MAJOR_STATUS_MOVES = {"sleeppowder", "spore", "lovely kiss", "sing"}
    p2_major_status_attempts = 0
    p2_major_status_successes = 0
    p2_major_status_turns_suffered = 0
    total_turns = len(timeline)
    for turn in timeline:
        # Infliction Rate (P2 trying to hit P1)
        p2_move = turn.get("p2_move_details")
        p1_state = turn.get("p1_pokemon_state", {})
        p1_current_status = p1_state.get("status", "nostatus")
        if p2_move:
            move_name = p2_move.get("name", "").lower()
            if move_name in MAJOR_STATUS_MOVES:
                p2_major_status_attempts += 1
                # Check if P1 ended the turn with a major status
                if p1_current_status in MAJOR_STATUSES:
                    p2_major_status_successes += 1
                    
        # Cumulative Status Turns Suffered (P2 suffering)
        p2_state = turn.get("p2_pokemon_state", {})
        p2_current_status = p2_state.get("status", "nostatus")
        
        if p2_current_status in MAJOR_STATUSES:
            p2_major_status_turns_suffered += 1
    
    if p2_major_status_attempts > 0:
        features["p2_major_status_infliction_rate"] = p2_major_status_successes / p2_major_status_attempts
    else:
        features["p2_major_status_infliction_rate"] = 0.0
    #P2 Cumulative Status Turns Percentage
    if total_turns > 0:
        features["p2_cumulative_major_status_turns_pct"] = p2_major_status_turns_suffered / total_turns
    else:
        features["p2_cumulative_major_status_turns_pct"] = 0.0
        
    return features
def calculate_dynamic_boost_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates features tracking the dynamic stat boost advantage (e.g., Swords Dance, Amnesia)
    gained by P1 relative to P2 across the timeline.
    """
    features = {}
    timeline = battle.get("battle_timeline", [])
    
    if not timeline:
        return {
            "p1_net_boost_sum": 0.0,
            "p1_max_offense_boost_diff": 0.0,
            "p1_max_speed_boost_diff": 0.0
        }
    net_boost_list = []
    offense_boost_diff_list = []
    speed_boost_diff_list = []
    for turn in timeline:
        p1_boosts = turn.get("p1_pokemon_state", {}).get("boosts", {})
        p2_boosts = turn.get("p2_pokemon_state", {}).get("boosts", {})
        
        # Boost levels can be -6 to +6. Boosts in the data are typically stat stage changes.
        
        # P1 Total Boost Sum vs P2 Total Boost Sum
        p1_total_boost = sum(p1_boosts.values())
        p2_total_boost = sum(p2_boosts.values())
        net_boost_list.append(p1_total_boost - p2_total_boost)
        
        # P1 Offensive Boost (atk + spa) vs P2 Offensive Boost
        p1_offense_boost = p1_boosts.get("atk", 0) + p1_boosts.get("spa", 0)
        p2_offense_boost = p2_boosts.get("atk", 0) + p2_boosts.get("spa", 0)
        offense_boost_diff_list.append(p1_offense_boost - p2_offense_boost)
        # P1 Speed Boost vs P2 Speed Boost
        p1_speed_boost = p1_boosts.get("spe", 0)
        p2_speed_boost = p2_boosts.get("spe", 0)
        speed_boost_diff_list.append(p1_speed_boost - p2_speed_boost)
    # 1. Net Cumulative Boost Sum
    # A positive sum indicates P1 spent more turns with a higher boost level than P2.
    features["p1_net_boost_sum"] = np.sum(net_boost_list)
    # 2. Maximum Offensive Boost Differential
    # Captures the peak offensive setup advantage.
    features["p1_max_offense_boost_diff"] = np.max(offense_boost_diff_list) if offense_boost_diff_list else 0.0
    # 3. Maximum Speed Boost Differential
    # Captures the peak speed advantage gained via boost moves.
    features["p1_max_speed_boost_diff"] = np.max(speed_boost_diff_list) if speed_boost_diff_list else 0.0
    
    return features
def get_type_multiplier(move_type: str, defender_types: list, type_chart: dict) -> float:
    """Calculates the combined type effectiveness multiplier."""
    if not defender_types or move_type.upper() == "NOTYPE":
        return 1.0
    
    multiplier = 1.0
    for def_type in defender_types:
        try:
            # Look up multiplier: TypeChart[Attacking Type][Defending Type]
            effectiveness = type_chart.get(move_type.upper(), {}).get(def_type.upper(), 1.0)
            multiplier *= effectiveness
        except:
            continue
            
    return multiplier
def calculate_team_coverage_features(battle: Dict[str, Any], type_chart: Dict) -> Dict[str, float]:
    """
    Calculates P1"s offensive coverage against P2"s lead Pokémon.
    """
    features = {}
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details", {})
    
    if not p1_team or not p2_lead:
        return {"p1_team_super_effective_moves": 0.0}
    p2_defender_types = [t for t in p2_lead.get("types", []) if t != "notype"]
    super_effective_count = 0
    
    # We only check P1"s Pokémon types, assuming they carry moves of their own type (STAB).
    # This is a strong proxy for offensive coverage.
    for p1_poke in p1_team:
        p1_poke_types = [t for t in p1_poke.get("types", []) if t != "notype"]
        has_super_effective_type = False
        for p1_type in p1_poke_types:
            # Check if this P1 type is Super Effective against any of P2"s lead types
            type_mult = get_type_multiplier(p1_type, p2_defender_types, type_chart)
            if type_mult >= 2.0:
                has_super_effective_type = True
                break
        if has_super_effective_type:
            super_effective_count += 1
    features["p1_team_super_effective_moves"] = float(super_effective_count)
    return features
def calculate_action_efficiency_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates P1"s rate of using non-damaging (Status) moves.
    """
    features = {}
    timeline = battle.get("battle_timeline", [])
    
    if not timeline:
        return {"p1_status_move_rate": 0.0}
    p1_status_move_count = 0
    p1_total_moves = 0
    
    for turn in timeline:
        p1_move = turn.get("p1_move_details")
        
        if p1_move and p1_move.get("category"):
            p1_total_moves += 1
            if p1_move["category"].upper() == "STATUS":
                p1_status_move_count += 1
    
    if p1_total_moves > 0:
        features["p1_status_move_rate"] = p1_status_move_count / p1_total_moves
    else:
        features["p1_status_move_rate"] = 0.0
        
    return features
def compute_team_resistance(team, type_chart):
    """
    Calcola un punteggio medio di resistenza per un team di Pokémon.
    È il reciproco della media delle debolezze (più alto = più resistente).
    Se un Pokémon ha poche debolezze (cioè molti tipi che fanno <1x danno),
    il team ottiene un punteggio di resistenza più alto.
    """
    if not team:
        return 1.0  # neutro
    all_attack_types = list(type_chart.keys())
    weakness_counts = []
    for p in team:
        p_types = [t for t in p.get("types", []) if t != "notype"]
        if not p_types:
            continue
        weak_to = []
        for atk_type in all_attack_types:
            # Calcolo del moltiplicatore di efficacia combinato
            multiplier = 1.0
            for d_type in p_types:
                multiplier *= type_chart.get(atk_type.upper(), {}).get(d_type.upper(), 1.0)
            if multiplier > 1.0:  # debolezza
                weak_to.append(atk_type)
        weakness_counts.append(len(weak_to))
    # Media delle debolezze
    mean_weakness = np.mean(weakness_counts) if weakness_counts else 0.0
    # Resistenza = reciproco della vulnerabilità media
    if mean_weakness > 0:
        return 1/mean_weakness#83.89% (+/- 0.53%)=>83.90% (+/- 0.55%)
    else:
        return 1.0  # team senza debolezze → massima resistenza
def compute_team_weakness(team, type_chart):
    if not team:
        return 0.0  # neutral (no team)
    all_attack_types = list(type_chart.keys())
    weakness_counts = []
    for p in team:
        p_types = [t for t in p.get("types", []) if t != "notype"]
        if not p_types:
            continue
        weak_to = []
        for atk_type in all_attack_types:
            # Combined multiplier for multitype Pokémon
            multiplier = 1.0
            for d_type in p_types:
                multiplier *= type_chart.get(atk_type.upper(), {}).get(d_type.upper(), 1.0)
            if multiplier > 1.0:  # super effective = weakness
                weak_to.append(atk_type)
        weakness_counts.append(len(weak_to))
    # Mean number of weaknesses per Pokémon
    mean_weakness = np.mean(weakness_counts) if weakness_counts else 0.0
    # Normalize to 0–1 range for consistency (optional)
    max_possible = len(all_attack_types)
    weakness_score = mean_weakness / max_possible if max_possible > 0 else mean_weakness
    return weakness_score
def calculate_expected_damage_ratio_turn_1(battle: dict, type_chart: dict) -> float:
    timeline = battle.get("battle_timeline", [])
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details", {})
    if not timeline or not p1_team or not p2_lead:
        return 0.0 
    turn_1 = timeline[0]
    p1_move = turn_1.get("p1_move_details")
    p2_move = turn_1.get("p2_move_details")
    p1_lead_stats = p1_team[0] 
    p2_lead_stats = p2_lead 
    p1_defender_types = [t for t in p1_lead_stats.get("types", []) if t != "notype"]
    p2_defender_types = [t for t in p2_lead_stats.get("types", []) if t != "notype"]
    p1_expected_damage = 0.0
    p2_expected_damage = 0.0
    
    # Calculate P1 Damage Potential on P2
    if p1_move and p1_move.get("category") in ["SPECIAL", "PHYSICAL"]:
        base_power = p1_move.get("base_power", 0)
        move_type = p1_move.get("type", "").upper()
        category = p1_move.get("category", "").upper()
        if category == "SPECIAL":
            att_stat = p1_lead_stats.get("base_spa", 1)
            def_stat = p2_lead_stats.get("base_spd", 1)
        else: # PHYSICAL
            att_stat = p1_lead_stats.get("base_atk", 1)
            def_stat = p2_lead_stats.get("base_def", 1)
        type_mult = get_type_multiplier(move_type, p2_defender_types, type_chart)
        p1_expected_damage = base_power * (att_stat / def_stat) * type_mult
    # Calculate P2 Damage Potential on P1 ---
    if p2_move and p2_move.get("category") in ["SPECIAL", "PHYSICAL"]:
        base_power = p2_move.get("base_power", 0)
        move_type = p2_move.get("type", "").upper()
        category = p2_move.get("category", "").upper()
        if category == "SPECIAL":
            att_stat = p2_lead_stats.get("base_spa", 1)
            def_stat = p1_lead_stats.get("base_spd", 1)
        else: # PHYSICAL
            att_stat = p2_lead_stats.get("base_atk", 1)
            def_stat = p1_lead_stats.get("base_def", 1)
        type_mult = get_type_multiplier(move_type, p1_defender_types, type_chart)
        p2_expected_damage = base_power * (att_stat / def_stat) * type_mult
    # 3. Return Log-Transformed Advantage ---
    # Using the log difference: log(A+1) - log(B+1) = log((A+1) / (B+1))
    # This stabilizes the feature, handles zero damage, and converts ratios to a scale 
    # centered around 0.
    
    # Add a small smoothing constant (1.0) to prevent log(0) and division issues.
    p1_smoothed_damage = p1_expected_damage + 1.0
    p2_smoothed_damage = p2_expected_damage + 1.0
    
    log_advantage = np.log(p1_smoothed_damage) - np.log(p2_smoothed_damage)
    
    return log_advantage
def get_pokemon_stats(team, name):
    for p in team:
        if p.get("name") == name:
            return {
                "base_hp": p.get("base_hp", 0),
                "base_atk": p.get("base_atk", 0),
                "base_def": p.get("base_def", 0),
                "base_spa": p.get("base_spa", 0),
                "base_spd": p.get("base_spd", 0),
                "base_spe": p.get("base_spe", 0)
            }
    return None
def battle_duration(battle):
    return len([t for t in battle["battle_timeline"] if t["p1_pokemon_state"]["hp_pct"] > 0 and
                                                     t["p2_pokemon_state"]["hp_pct"] > 0])
def compute_mean_stab_moves(timeline, pokemon_dict):
    if not timeline:
        return {
            "p1_mean_stab": 0.0,
            "p2_mean_stab": 0.0,
            "diff_mean_stab": 0.0
        }
    p1_stab_counts, p2_stab_counts = [], []
    for turn in timeline:
        # --- Player 1 ---
        p1_state = turn.get("p1_pokemon_state", {})
        p1_move = turn.get("p1_move_details", {})
        if p1_state and p1_move:
            p1_name = p1_state.get("name", "").lower()
            move_type = p1_move.get("type", "").upper()
            p1_types = pokemon_dict.get(p1_name, [])
            # Check STAB
            if move_type in [t.upper() for t in p1_types]:
                p1_stab_counts.append(1)
            else:
                p1_stab_counts.append(0)
        # --- Player 2 ---
        p2_state = turn.get("p2_pokemon_state", {})
        p2_move = turn.get("p2_move_details", {})
        if p2_state and p2_move:
            p2_name = p2_state.get("name", "").lower()
            move_type = p2_move.get("type", "").upper()
            p2_types = pokemon_dict.get(p2_name, [])
            if move_type in [t.upper() for t in p2_types]:
                p2_stab_counts.append(1)
            else:
                p2_stab_counts.append(0)
    # Compute means
    p1_mean_stab = np.sum(p1_stab_counts) if p1_stab_counts else 0.0
    p2_mean_stab = np.sum(p2_stab_counts) if p2_stab_counts else 0.0
    return {
        "p1_mean_stab": p1_mean_stab,
        "p2_mean_stab": p2_mean_stab,
        "diff_mean_stab": p1_mean_stab - p2_mean_stab
    }
def compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart, is_test=False, battle_id=""):
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
        #Get types from dictionary
        p1_types = pokemon_dict.get(p1_name.lower(), [])
        p2_types = pokemon_dict.get(p2_name.lower(), [])
        #print(len(p1_types),len(p2_types))
        if not p1_types or not p2_types:
            continue
        #P1 vs P2
        p1_mult = []
        for atk_type in p1_types:
            mult = 1.0
            for def_type in p2_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p1_mult.append(mult)
        #turn summary
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0
        #P2 vs P1
        p2_mult = []
        for atk_type in p2_types:
            mult = 1.0
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
    return {
        "p1_type_advantage": p1_avg,
        "p2_type_advantage": p2_avg,
        "diff_type_advantage": p1_avg - p2_avg
    }
def compute_avg_offensive_potential(timeline, pokemon_dict, type_chart, is_test=False, battle_id=""):
    #... (initial checks are the same)
    if not timeline:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }
    p1_advantages = []
    p2_advantages = []
    
    #Get all possible move types from the type chart keys
    all_move_types = list(type_chart.keys())
    for turn in timeline:
        #... (get names and types, same as before)
        p1_name = turn.get("p1_pokemon_state", {}).get("name")
        p2_name = turn.get("p2_pokemon_state", {}).get("name")
        if not p1_name or not p2_name:
            continue
        p1_types = pokemon_dict.get(p1_name.lower(), [])
        p2_types = pokemon_dict.get(p2_name.lower(), [])
        if not p1_types or not p2_types:
            continue
        #P1 vs P2: Calculate average effectiveness of ALL move types
        p1_mult = []
        for atk_type in all_move_types:
            mult = 1.0
            for def_type in p2_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p1_mult.append(mult)
        #turn summary
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0
        #P2 vs P1: Calculate average effectiveness of ALL move types
        p2_mult = []
        for atk_type in all_move_types:
            mult = 1.0
            for def_type in p1_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p2_mult.append(mult)
        #turn summary
        p2_adv = np.mean(p2_mult) if p2_mult else 1.0
        p1_advantages.append(p1_adv)
        p2_advantages.append(p2_adv)
    #... (final calculation and return are the same)
    if not p1_advantages or not p2_advantages:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }
    p1_avg = np.mean(p1_advantages)
    p2_avg = np.mean(p2_advantages)
    
    return {
        "p1_type_advantage": p1_avg,
        "p2_type_advantage": p2_avg,
        "diff_type_advantage": p1_avg - p2_avg
    }
def conta_status_anomali(timeline):
    conteggio_p1 = 0
    conteggio_p2 = 0
    major_status_p1 = 0
    major_status_p2 = 0
    MAJOR_STATUSES = {"slp", "frz"} 
    for turno in timeline:
        stato_dettagli_p1 = turno.get("p1_pokemon_state", {})
        status_string = stato_dettagli_p1.get("status", "")
        if status_string.lower() != "nostatus":
            conteggio_p1 += 1
            if status_string.lower() in MAJOR_STATUSES:#bloccanti
                major_status_p1 += 1
        stato_dettagli_p2 = turno.get("p2_pokemon_state", {})
        status_string = stato_dettagli_p2.get("status", "")
        if status_string:
            if status_string.lower() != "nostatus":
                conteggio_p2 += 1
                if status_string.lower() in MAJOR_STATUSES:#bloccanti
                    major_status_p2 += 1
    differenza = conteggio_p1 - conteggio_p2 #P1 meno P2
    
    return {
        "status_p1": conteggio_p1,
        "status_p2": conteggio_p2,
        "diff_status": differenza,
        "major_status_p1": major_status_p1,
        "major_status_p2": major_status_p2,
        "major_status_diff": major_status_p1-major_status_p2,
        
    }
    
def get_p1_base_speed(pokemon_name, p1_team_details):
    search_name = pokemon_name.lower().strip()
    for pokemon in p1_team_details:
        if pokemon.get("name", "").lower().strip() == search_name:
            return pokemon.get("base_spe", 0)
    return 0    
def compute_statistics(values: Iterable[float], prefix: str) -> Dict[str, float]:
    seq = list(values)
    if not seq:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            #less informative: min/max statistics removed
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    arr = np.asarray(seq, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        #less informative: min/max statistics removed
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }
def calculate_battle_stats(battle):
    #Leader P1: p1_team_details[0]
    p1_team = battle["p1_team_details"]
    p1_lead = p1_team[0]
    p2_lead = battle["p2_lead_details"]
    p1_lead_spe = p1_lead["base_spe"]
    #Leader P2: p2_lead_details
    p2_lead_spe = p2_lead["base_spe"]
    diff_speed_first = p1_lead_spe - p2_lead_spe
    timeline_speeds_diffs = []
    p1_team_details = battle.get("p1_team_details", [])
    p2_lead_details = battle.get("p2_lead_details", {})
    p2_lead_name = p2_lead_details.get("name", "").lower().strip()
    p2_base_spe = p2_lead_details.get("base_spe", 0)
    
    #Itera attraverso la timeline
    for turn in battle.get("battle_timeline", []):
        p1_state_data = turn.get("p1_pokemon_state")
        p1_name = p1_state_data.get("name", "") if p1_state_data else ""
        p2_state_data = turn.get("p2_pokemon_state")
        
        p2_name = p2_state_data.get("name", "") if p2_state_data else ""
        p1_speed = get_p1_base_speed(p1_name, p1_team_details)
        p2_speed = 0
        if p2_name.lower().strip() == p2_lead_name:
            p2_speed = p2_base_spe
        #Se P2 non ha il lead in campo (se il JSON fosse completo), il dato non sarebbe disponibile con i vincoli dati.
        
        #Calcola e aggiungi la differenza (P1 - P2)
        if p1_speed != 0 or p2_speed != 0:
            timeline_speeds_diffs.append(p1_speed - p2_speed)
    #Calcola la media delle differenze di velocità
    if timeline_speeds_diffs:
        diff_speed_timeline = np.mean(timeline_speeds_diffs)
    else:
        diff_speed_timeline = diff_speed_first
    p1_stats_sums = {key: 0 for key in STAT_FIELDS}
    
    for pokemon in battle["p1_team_details"]:
        for key in STAT_FIELDS:
            p1_stats_sums[key] += pokemon[key]
    
    #lead p1
    sum_stat_lead_p1 = 0
    sum_stat_lead_p2 = 0
    for key in STAT_FIELDS:
        sum_stat_lead_p1 += battle["p1_team_details"][0][key]
        sum_stat_lead_p2 += battle["p2_lead_details"][key]
        
    num_p1_pokemon = len(battle["p1_team_details"])
    p1_avg_stats = {key: p1_stats_sums[key] / num_p1_pokemon for key in STAT_FIELDS}
    #Statistiche del Lead del Team 2
    p2_lead_stats = {key: battle["p2_lead_details"][key] for key in STAT_FIELDS}
    #Calcolo della differenza media delle statistiche
    #Differenza: Media(Team 1) - Statistiche(Lead Team 2)
    stat_diffs = []
    for key in STAT_FIELDS:
        diff = p1_avg_stats[key] - p2_lead_stats[key]
        stat_diffs.append(diff)
    #La feature è la media di queste 6 differenze di statistica
    diff_stat = np.mean(stat_diffs) if stat_diffs else 0
    res = {
        #"diff_speed_first": diff_speed_first,
        "diff_speed_timeline": diff_speed_timeline,
        "diff_stat_mean": diff_stat,
        "sum_stat_lead_p1": sum_stat_lead_p1,
        "sum_stat_lead_p2": sum_stat_lead_p2,
        "diff_stat_lead": sum_stat_lead_p1 - sum_stat_lead_p2,
    }
    for stat in STAT_FIELDS:
        values = [member.get(stat, 0) or 0 for member in p1_team]
        res.update(compute_statistics(values, f"p1_team_{stat}"))
        
        p1_lead_stat = p1_lead.get(stat, 0)
        p2_lead_stat = p2_lead.get(stat, 0)
        res.update({f"p2_lead_{stat}":p2_lead_stat})
        
        res.update({f"diff_lead_{stat}":p1_lead_stat - p2_lead_stat})
    #features["p1_mean_spe"] = np.mean([p.get("base_spe", 0) for p in p1_team])
    return res
def hp_advantage_trend(battle):
    hp_adv = []
    for turn in battle["battle_timeline"]:
        p1_hp = turn["p1_pokemon_state"]["hp_pct"]
        p2_hp = turn["p2_pokemon_state"]["hp_pct"]
        hp_adv.append(p1_hp - p2_hp)
    x = np.arange(len(hp_adv))
    slope, _, _, _, _ = linregress(x, hp_adv)
    return slope
def extract_hp_features(battle):
    #Dizionari per tenere traccia del massimo hp_pct raggiunto da ciascun Pokémon distinto
    #La chiave è il nome del Pokémon, il valore è il massimo hp_pct visto.
    p1_set_hp_pct = {}
    p2_set_hp_pct = {}
    timeline = battle.get("battle_timeline", [])
    #2. Scorre la timeline della battaglia
    for turn in timeline:#
        #Estrai lo stato del Pokémon 1
        p1_pokemon_state = turn.get("p1_pokemon_state", None)
        if p1_pokemon_state:
            p1_name = p1_pokemon_state.get("name")
            p1_hp_pct = p1_pokemon_state.get("hp_pct")
            p1_set_hp_pct[p1_name] = p1_hp_pct
        #Estrai lo stato del Pokémon 2
        p2_pokemon_state = turn.get("p2_pokemon_state", None)
        if p2_pokemon_state:
            p2_name = p2_pokemon_state.get("name")
            p2_hp_pct = p2_pokemon_state.get("hp_pct")
            p2_set_hp_pct[p2_name] = p2_hp_pct
    team_member_count =  len(battle.get("p1_team_details"))-len(p1_set_hp_pct.keys())
    p1_hp_pct_sum = sum(p1_set_hp_pct.values()) + (team_member_count-len(p1_set_hp_pct.keys()))
    p2_hp_pct_sum = sum(p2_set_hp_pct.values()) + (team_member_count-len(p2_set_hp_pct.keys()))
    diff_hp_pct = p1_hp_pct_sum - p2_hp_pct_sum
    #print(p1_hp_pct_sum,p2_hp_pct_sum,diff_hp_pct)
    return {
        "p1_hp_pct_sum": p1_hp_pct_sum,
        "p2_hp_pct_sum": p2_hp_pct_sum,
        "diff_hp_pct": diff_hp_pct
    }
#MOVES
def get_move_features(timeline):
    p1_move_power_weighted = []
    p1_number_attacks = 0
    p1_number_status = 0
    
    p1_sum_negative_priority = 0
    p2_sum_negative_priority = 0
    
    
    p2_move_power_weighted = []
    p2_number_attacks = 0
    p2_number_status = 0
    for turn in timeline:
        #Assumiamo che la mossa sia sotto "p1_move_details|p2_move_details"
        move_details_key = "p1_move_details"#|p2_move_details
        if turn.get(move_details_key) != None:
            move = turn[move_details_key]
            #1. Feature: move_power_weighted
            #Un "danno atteso" che combina potenza e accuratezza
            accuracy = move.get("accuracy", 1.0) #Default a 1.0 se mancante
            base_power = move.get("base_power", 0)
            priority = move.get("priority", 0)
            #Se la precisione è 0, assumiamo che sia una mossa a 100% di precisione 
            #se non è specificato (come "noaccuracy"), altrimenti usiamo il valore fornito.
            if accuracy == 0:
                 weighted_power = base_power
            else:
                weighted_power = base_power * accuracy
            
            p1_move_power_weighted.append(round(weighted_power, 3))
            #2. Feature: is_physical_or_special
            #Codifica della categoria di attacco (1 per attacco, 0 per status)
            category = move.get("category", "STATUS").upper()
            if category in ["PHYSICAL", "SPECIAL"]:
                p1_number_attacks+=1
            elif category == "STATUS":
                p1_number_status+=1
                
            if(priority == -1):
                p1_sum_negative_priority +=1
        move_details_key = "p2_move_details"#|p2_move_details
        if turn.get(move_details_key) != None:
            move = turn[move_details_key]
            
            #1. Feature: move_power_weighted
            #Un "danno atteso" che combina potenza e accuratezza
            accuracy = move.get("accuracy", 1.0) #Default a 1.0 se mancante
            base_power = move.get("base_power", 0)
            priority = move.get("priority", 0)
            #Se la precisione è 0, assumiamo che sia una mossa a 100% di precisione 
            #se non è specificato (come "noaccuracy"), altrimenti usiamo il valore fornito.
            if accuracy == 0:
                 weighted_power = base_power
            else:
                weighted_power = base_power * accuracy
            
            p2_move_power_weighted.append(round(weighted_power, 3))
            #2. Feature: is_physical_or_special
            #Codifica della categoria di attacco (1 per attacco, 0 per status)
            category = move.get("category", "STATUS").upper()
            if category in ["PHYSICAL", "SPECIAL"]:
                p2_number_attacks+=1
            elif category == "STATUS":
                p2_number_status+=1
                
            if(priority == -1):
                p2_sum_negative_priority +=1
            
    return {
        "p1_move_power_weighted": np.sum(p1_move_power_weighted),
        "p1_number_attacks": p1_number_attacks,
        "p1_number_status": p1_number_status,
        
        "p2_move_power_weighted": np.sum(p2_move_power_weighted),
        "p2_number_attacks": p2_number_attacks,
        "p2_number_status": p2_number_status,
        
        #non ancora usate
        "diff_number_attack": p1_number_attacks - p2_number_attacks,
        "diff_number_status": p1_number_status - p2_number_status,
        
        "p1_sum_negative_priority": p1_sum_negative_priority,
        "p2_sum_negative_priority": p2_sum_negative_priority,
        "diff_negative_priority": p1_sum_negative_priority-p2_sum_negative_priority,
        
    }
BOOST_MULT = {
    -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
     0: 1.0,
     1: 1.5,  2: 2.0,  3: 2.5,  4: 3.0,  5: 3.5,  6: 4.0
}
def extract_boost_features(battle_timeline, base_stats_p1, base_stats_p2):
    p1_boost_count = 0
    p2_boost_count = 0
    p1_first_boost_turn = None
    p2_first_boost_turn = None
    last_boost_p1 = None
    last_boost_p2 = None
    for entry in battle_timeline:
        turn = entry.get("turn", None)
        p1_boosts = entry.get("p1_pokemon_state", {}).get("boosts", {})
        p2_boosts = entry.get("p2_pokemon_state", {}).get("boosts", {})
        #Salva ultimo stato
        last_boost_p1 = p1_boosts
        last_boost_p2 = p2_boosts
        #Count boost
        if any(v != 0 for v in p1_boosts.values()):
            p1_boost_count += 1
            if p1_first_boost_turn is None:
                p1_first_boost_turn = turn
        if any(v != 0 for v in p2_boosts.values()):
            p2_boost_count += 1
            if p2_first_boost_turn is None:
                p2_first_boost_turn = turn
    #se nessun boost mai visto
    if last_boost_p1 is None:
        last_boost_p1 = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}
    if last_boost_p2 is None:
        last_boost_p2 = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}
    #Differenze dell’ultimo turno
    diff_boost_atk = last_boost_p1["atk"] - last_boost_p2["atk"]
    diff_boost_def = last_boost_p1["def"] - last_boost_p2["def"]
    diff_boost_spa = last_boost_p1["spa"] - last_boost_p2["spa"]
    diff_boost_spd = last_boost_p1["spd"] - last_boost_p2["spd"]
    diff_boost_spe = last_boost_p1["spe"] - last_boost_p2["spe"]
    boost_p1_total = sum(last_boost_p1.values())
    boost_p2_total = sum(last_boost_p2.values())
    diff_boost_total = boost_p1_total - boost_p2_total
    #Statistiche temporali -
    diff_boost_count = p1_boost_count - p2_boost_count
    diff_first_boost_turn = (p1_first_boost_turn or 31) - (p2_first_boost_turn or 31)
    #Effective stat = base_stat × boost_multiplier -
    def eff(value, stage):
        return value * BOOST_MULT.get(stage, 1.0)
    effective_p1 = {
        stat: eff(base_stats_p1[f"base_{stat}"], last_boost_p1[stat])
        for stat in ["atk", "def", "spa", "spd", "spe"]
    }
    effective_p2 = {
        stat: eff(base_stats_p2[f"base_{stat}"], last_boost_p2[stat])
        for stat in ["atk", "def", "spa", "spd", "spe"]
    }
    #differenze reali
    diff_effective = {
        f"diff_effective_{stat}": effective_p1[stat] - effective_p2[stat]
        for stat in effective_p1
    }
    #power offensivo/difensivo
    p1_eff_off = effective_p1["atk"] + effective_p1["spa"]
    p1_eff_def = effective_p1["def"] + effective_p1["spd"]
    p2_eff_off = effective_p2["atk"] + effective_p2["spa"]
    p2_eff_def = effective_p2["def"] + effective_p2["spd"]
    diff_eff_off = p1_eff_off - p2_eff_off
    diff_eff_def = p1_eff_def - p2_eff_def
    #speed advantage reale (boolean)
    p1_is_faster = int(effective_p1["spe"] > effective_p2["spe"])
    res = {
        "diff_boost_last_turn": diff_boost_total,
        "diff_boost_atk_last_turn": diff_boost_atk,
        "diff_boost_def_last_turn": diff_boost_def,
        "diff_boost_spa_last_turn": diff_boost_spa,
        "diff_boost_spd_last_turn": diff_boost_spd,
        "diff_boost_spe_last_turn": diff_boost_spe,
        "diff_boost_count_turni": diff_boost_count,
        "diff_turn_first_boost": diff_first_boost_turn,
        #**diff_effective,
        "diff_effective_offense": diff_eff_off,
        "diff_effective_defense": diff_eff_def,
        "p1_is_faster_effective": p1_is_faster,
    }
    return res
def calcola_feature_boost(timeline):
    totale_boost_p1 = 0
    totale_boost_p2 = 0
    for turno in timeline:
        #Funzione helper per calcolare la somma dei boost di un singolo Pokémon
        def somma_boosts(pokemon_state):
            somma = 0
            if pokemon_state and "boosts" in pokemon_state:
                #Somma tutti i valori di boost (atk, def, spa, spd, spe)
                somma = sum(pokemon_state["boosts"].values())
            return somma
        #Prova ad accedere allo stato del P1 (presumendo che sia una chiave nel JSON)
        if "p1_pokemon_state" in turno:
            boost_corrente_p1 = somma_boosts(turno["p1_pokemon_state"])
            totale_boost_p1 += boost_corrente_p1
        #Prova ad accedere allo stato del P2 (presumendo che sia una chiave nel JSON)
        if "p2_pokemon_state" in turno:
            boost_corrente_p2 = somma_boosts(turno["p2_pokemon_state"])
            totale_boost_p2 += boost_corrente_p2
    #Calcola la differenza
    #diff_boost = totale_boost_p1 - totale_boost_p2
    #Restituisce le feature
    return {
        "boost_p1": totale_boost_p1,
        "boost_p2": totale_boost_p2,
        #"diff_boost": diff_boost
    }
#12
important_effects = [
    "substitute", "reflect", "light_screen",
    "leech_seed", "bind", "wrap", "clamp",
    "confusion", "toxic", "poison", "burn", "paralysis"
]
def extract_effect_features_from_timeline(timeline, prefix):
    #inizializza strutture
    freq = {eff: 0 for eff in important_effects}
    first_turn = {eff: None for eff in important_effects}
    for entry in timeline:
        turn = entry.get("turn", None)
        #prendi lo stato del pokemon corretto (p1 o p2)
        state_key = "p1_pokemon_state" if prefix == "p1" else "p2_pokemon_state"
        state = entry.get(state_key, {})
        effects = state.get("effects", [])
        #aggiorna freq e first_turn
        for eff in important_effects:
            if eff in effects:
                freq[eff] += 1
                if first_turn[eff] is None:
                    first_turn[eff] = turn
    #converte None → 31 per indicare "mai apparso"
    for eff in important_effects:
        if first_turn[eff] is None:
            first_turn[eff] = 31
    #impacchetta le feature
    features = {}
    for eff in important_effects:
        features[f"{prefix}_{eff}_freq"] = freq[eff]
        features[f"{prefix}_{eff}_first_turn"] = first_turn[eff]
    return features
#Output: {"p1_hp_pct_sum": 2.0, "p2_hp_pct_sum": 1.9, "diff_hp_pct": 0.1}
import random

def shuffle_dict(d, seed=1234):
    rnd = random.Random(seed)       # generatore deterministico
    items = list(d.items())
    rnd.shuffle(items)              # shuffle riproducibile
    return dict(items)

def create_pokemon_dict(data):
    pokemon_dict = {}
    for battle in data:
        p1_team = battle.get("p1_team_details", [])
        for p in p1_team:
            name = p.get("name")
            types = [t for t in p.get("types", []) if t != "notype"]
            if name:
                if name not in pokemon_dict:
                    pokemon_dict[name] = set()
                pokemon_dict[name].update(types)
        p2_lead = battle.get("p2_lead_details")
        if p2_lead:
            name = p2_lead.get("name")
            types = [t for t in p2_lead.get("types", []) if t != "notype"]
            if name:
                if name not in pokemon_dict:
                    pokemon_dict[name] = set()
                pokemon_dict[name].update(types)
    return pokemon_dict
def create_features(data: list[dict], is_test=False) -> pd.DataFrame:
    feature_list = []
    pokemon_dict = create_pokemon_dict(data)
    #definiamo le features
    for battle in tqdm(data, desc="Extracting features"):
        battle_id = battle.get("battle_id")
        features = {}
        status_change_diff = []
        p1_team = battle.get("p1_team_details", [])
        #features.update(create_feature_instance_final(battle, pokemon_dict, []))
        """inizio create_feature_instance_final"""
        features = {}
        # --- Player 1 Team Features ---
        p1_mean_hp = p1_mean_spe = p1_mean_atk = p1_mean_def = p1_mean_spd = p1_mean_spa = 0.0
        p1_lead_hp = p1_lead_spe = p1_lead_atk = p1_lead_def = p1_lead_spd = p1_lead_spa = 0.0
        p1_team = battle.get("p1_team_details", [])
        status_features = calculate_status_efficacy_features(battle)
        features["p1_major_status_infliction_rate"] = status_features["p1_major_status_infliction_rate"]
        features["p1_cumulative_major_status_turns_pct"] = status_features["p1_cumulative_major_status_turns_pct"]
        p2_status_features = calculate_p2_status_control_features(battle)
        features["p2_major_status_infliction_rate"] = p2_status_features["p2_major_status_infliction_rate"]
        features["p2_cumulative_major_status_turns_pct"] = p2_status_features["p2_cumulative_major_status_turns_pct"]
        dynamic_boost_features = calculate_dynamic_boost_features(battle)
        #features["p1_net_boost_sum"] = dynamic_boost_features["p1_net_boost_sum"]
        features["p1_max_offense_boost_diff"] = dynamic_boost_features["p1_max_offense_boost_diff"]
        #features["p1_max_speed_boost_diff"] = dynamic_boost_features["p1_max_speed_boost_diff"]
        p1_team_super_effective_moves = calculate_team_coverage_features(battle, type_chart)
        features["p1_team_super_effective_moves"] = p1_team_super_effective_moves["p1_team_super_effective_moves"]
        
        p1_status_move_rate = calculate_action_efficiency_features(battle)
        #features["p1_status_move_rate"] = p1_status_move_rate["p1_status_move_rate"]
        expected_damage_ratio_turn_1 = 0.0
        try:
            expected_damage_ratio_turn_1 = calculate_expected_damage_ratio_turn_1(battle, type_chart)
            features["expected_damage_ratio_turn_1"] = expected_damage_ratio_turn_1
        except Exception:
            features["expected_damage_ratio_turn_1"] = 0.0
        ##
        if p1_team:
            bst_values = []
            max_offense = 0
            max_speed = 0
            for p in p1_team:
                p_hp = p.get("base_hp", 0)
                p_atk = p.get("base_atk", 0)
                p_def = p.get("base_def", 0)
                p_spa = p.get("base_spa", 0)
                p_spd = p.get("base_spd", 0)
                p_spe = p.get("base_spe", 0)
                
                current_offense = max(p_atk, p_spa)
                max_offense = max(max_offense, current_offense)
                
                max_speed = max(max_speed, p_spe)
                
                bst_values.append(p_hp + p_atk + p_def + p_spa + p_spd + p_spe)
            features["p1_max_offensive_stat"] = max_offense#83.97% (+/- 0.52%)=>83.97% (+/- 0.40%)
            features["p1_max_speed_stat"] = max_speed#83.97% (+/- 0.52%)=>83.95% (+/- 0.54%)
            
            p1_mean_hp = np.mean([p.get("base_hp", 0) for p in p1_team])
            p1_mean_spe = np.mean([p.get("base_spe", 0) for p in p1_team])
            p1_mean_atk = np.mean([p.get("base_atk", 1) for p in p1_team])
            p1_mean_def = np.mean([p.get("base_def", 0) for p in p1_team])
            p1_mean_spd = np.mean([p.get("base_spd", 0) for p in p1_team])

            features["p1_mean_hp"] = p1_mean_hp
            features["p1_mean_spe"] = p1_mean_spe
            features["p1_mean_atk"] = p1_mean_atk
            features["p1_mean_def"] = p1_mean_def
            features["p1_mean_sp"] = p1_mean_spd
            p1_lead_hp =  p1_team[0].get("base_hp", 0)
            p1_lead_spe = p1_team[0].get("base_spe", 0)
            p1_lead_atk = p1_team[0].get("base_atk", 0)
            p1_lead_def = p1_team[0].get("base_def", 0)
            p1_lead_spd =  p1_team[0].get("base_spd", 0)


        #Player 2 Lead
        p2_hp = p2_spe = p2_atk = p2_def = p2_spd = 0.0
        p2_lead = battle.get("p2_lead_details")
        if p2_lead:
            p2_hp = p2_lead.get("base_hp", 0)
            p2_spe = p2_lead.get("base_spe", 0)
            p2_atk = p2_lead.get("base_atk", 0)
            p2_def = p2_lead.get("base_def", 0)
            p2_spd = p2_lead.get("base_spd", 0)

        features["diff_hp"]  = p1_lead_hp  - p2_hp
        features["diff_spe"] = p1_lead_spe - p2_spe
        features["diff_atk"] = p1_lead_atk - p2_atk
        features["diff_def"] = p1_lead_def - p2_def
        features["diff_spd"] =  p1_lead_spd - p2_spd
        
        timeline = battle.get("battle_timeline", [])
        if timeline:
            p1_atk_boosts, p2_atk_boosts, p2_def_boosts, p1_def_boosts = [], [], [], []
            for turn in timeline:
                p1_boosts = turn.get("p1_pokemon_state", {}).get("boosts", {})
                p2_boosts = turn.get("p2_pokemon_state", {}).get("boosts", {})
                p1_atk_boosts.append(p1_boosts.get("atk", 0))
                p2_atk_boosts.append(p2_boosts.get("atk", 0))
                p1_def_boosts.append(p1_boosts.get("def", 0))
                p2_def_boosts.append(p2_boosts.get("def", 0))

            p1_atk_boosts = np.array(p1_atk_boosts)
            p2_atk_boosts = np.array(p2_atk_boosts)
            p1_def_boosts = np.array(p1_def_boosts)
            p2_def_boosts = np.array(p2_def_boosts)

            #priorità delle mosse
            p1_priorities = []
            p2_priorities = []

            for turn in timeline:
                move1 = turn.get("p1_move_details")
                move2 = turn.get("p2_move_details")

                if isinstance(move1, dict) and move1.get("priority") is not None:
                    p1_priorities.append(move1["priority"])
                if isinstance(move2, dict) and move2.get("priority") is not None:
                    p2_priorities.append(move2["priority"])

            #priorità media per squadra
            p1_avg_move_priority = np.mean(p1_priorities) if p1_priorities else 0.0
            p2_avg_move_priority = np.mean(p2_priorities) if p2_priorities else 0.0

            #vantaggio relativo
            features["priority_diff"] = p1_avg_move_priority - p2_avg_move_priority

            #frazione dei turni in cui p1 ha più priorità
            if p1_priorities and p2_priorities:
                min_len = min(len(p1_priorities), len(p2_priorities))
                higher_priority_turns = sum(p1_priorities[i] > p2_priorities[i] for i in range(min_len))
                features["priority_rate_advantage"] = higher_priority_turns / max(1, min_len)
            else:
                features["priority_rate_advantage"] = 0.0

            stat_diffs = {
                "base_atk": [],
                "base_spa": [],
                "base_spe": []
            }

            for t in timeline:
                p1_state = t.get("p1_pokemon_state", {})
                p2_state = t.get("p2_pokemon_state", {})

                p1_name = p1_state.get("name")
                p2_name = p2_state.get("name")

                p1_stats = get_pokemon_stats(p1_team, p1_name) if p1_name else None
                p2_stats = None

                p2_lead = battle.get("p2_lead_details", {})
                if p2_name and p2_lead and p2_lead.get("name") == p2_name:
                    p2_stats = {
                            "base_hp": p2_lead.get("base_hp", 0),
                        "base_atk": p2_lead.get("base_atk", 0),
                            "base_def": p2_lead.get("base_def", 0),
                        "base_spa": p2_lead.get("base_spa", 0),
                            "base_spd": p2_lead.get("base_spd", 0),
                        "base_spe": p2_lead.get("base_spe", 0)
                    }

                #salto i turni se non ho le stati di uno dei 2 pokemon che si affrontano
                if not p1_stats or not p2_stats:
                    continue

                #diff
                for stat in stat_diffs.keys():
                    diff = p1_stats[stat] - p2_stats[stat]
                    stat_diffs[stat].append(diff)

            #aggrego le differenze di statistiche nella timeline
            for stat, diffs in stat_diffs.items():
                if diffs:
                    features[f"mean_{stat}_diff_timeline"] = np.mean(diffs)#83.74=>83.87
                    features[f"std_{stat}_diff_timeline"] = np.std(diffs)#83.78=>83.87
                else:
                    features[f"mean_{stat}_diff_timeline"] = 0.0
                    features[f"std_{stat}_diff_timeline"] = 0.0
            #SALUTE
            p1_hp = [t["p1_pokemon_state"]["hp_pct"] for t in timeline if t.get("p1_pokemon_state")]
            p2_hp = [t["p2_pokemon_state"]["hp_pct"] for t in timeline if t.get("p2_pokemon_state")]
            features["hp_diff_mean"] = np.mean(np.array(p1_hp) - np.array(p2_hp))
            features["p1_hp_advantage_mean"] = np.mean(np.array(p1_hp) > np.array(p2_hp))#GRAN BELLA OPZIONE DI CLASSIFICAZIONE POSSIBILE APPLICAZIONE DI EFFETTI DI ETEROGENEITA

            p1_hp_final ={}
            p2_hp_final ={}
            for t in timeline:
                if t.get("p1_pokemon_state"):
                    p1_hp_final[t["p1_pokemon_state"]["name"]]=t["p1_pokemon_state"]["hp_pct"]
                if t.get("p2_pokemon_state"):
                    p2_hp_final[t["p2_pokemon_state"]["name"]]=t["p2_pokemon_state"]["hp_pct"]
            #numero di pokemon usati dal giocatore nei primi 30 turni
            features["p1_n_pokemon_use"] =len(p1_hp_final.keys())
            features["p2_n_pokemon_use"] =len(p2_hp_final.keys())
            #differenza nello schieramento pockemon dopo 30 turni
            features["diff_final_schieramento"]=features["p1_n_pokemon_use"]-features["p2_n_pokemon_use"]
            nr_pokemon_sconfitti_p1 = np.sum([1 for e in list(p1_hp_final.values()) if e==0])
            nr_pokemon_sconfitti_p2 = np.sum([1 for e in list(p2_hp_final.values()) if e==0])
            features["nr_pokemon_sconfitti_p1"] = nr_pokemon_sconfitti_p1
            features["nr_pokemon_sconfitti_p2"] = nr_pokemon_sconfitti_p2
            #CHECK 84.31% (+/- 1.09%) => 84.35% (+/- 1.07%)
            features["nr_pokemon_sconfitti_diff"] = nr_pokemon_sconfitti_p1-nr_pokemon_sconfitti_p2
            #DOVREBBERO ESSERE BOMBA VITA DELLE DUE SQUADRE DOPO I 30 TURNI
            features["p1_pct_final_hp"] = np.sum(list(p1_hp_final.values()))+(6-len(p1_hp_final.keys()))
            features["p2_pct_final_hp"] = np.sum(list(p2_hp_final.values()))+(6-len(p2_hp_final.keys()))
            #SAREBBE CLAMOROSO NORMALIZZARLA ANCHE IN BASE ALLA DIFFERENZA DI VITA ASSOLUTA DEI POCKEMON LEADER DEI 2 PLAYER
            
            diff_final_hp = features["p1_pct_final_hp"]-features["p2_pct_final_hp"]
            #83.81% (+/- 0.52%) => 83.89% (+/- 0.55%)
            features["diff_final_hp"] = diff_final_hp

            #durata battaglia e tasso di perdita degli HP
            try:
                dur = battle_duration(battle)
            except Exception:
                dur = 0
            #83.66% (+/- 0.52%) => 83.89% (+/- 0.58%)
            features["battle_duration"] = dur
            #83.82% (+/- 0.49%) => 83.89% (+/- 0.55%)
            features["hp_loss_rate"] = diff_final_hp / dur if dur > 0 else 0.0

            #vedo anche come la salute media evolve nel tempo
            phases = 3 #early, mid, late game
            nr_turns = 30 #numero turni
            slice_idx = nr_turns // phases #slice index must be integer
            #print("slice_idx: ",slice_idx, "len p1_hp: ",len(p1_hp))
            features["early_hp_mean_diff"] = np.mean(np.array(p1_hp[:slice_idx]) - np.array(p2_hp[:slice_idx]))
            #83.94% (+/- 0.46%) => 83.98% (+/- 0.47%)
            features["late_hp_mean_diff"] = np.mean(np.array(p1_hp[-slice_idx:]) - np.array(p2_hp[-slice_idx:]))

            hp_delta = np.array(p1_hp) - np.array(p2_hp)
            features["hp_delta_trend"] = np.polyfit(range(len(hp_delta)), hp_delta, 1)[0]
            #83.87% (+/- 0.60%) => 83.89% (+/- 0.58%)
            features["hp_advantage_trend"] = hp_advantage_trend(battle)
            #fluttuazioni negli hp (andamento della partita: stabile o molto caotica)
            #restyle RIMOSSO p1_hp_std, p2_hp_std 83.89% (+/- 0.58%) => 83.89% (+/- 0.55%)
            features["p1_hp_std"] = np.std(p1_hp)
            features["p2_hp_std"] = np.std(p2_hp)
            features["hp_delta_std"] = np.std(hp_delta)

            ##STATUS (default nostatus, gli altri sono considerati negativi - i boost sono positivi)
            p1_status = [t["p1_pokemon_state"].get("status", "nostatus") for t in timeline if t.get("p1_pokemon_state")]
            p2_status = [t["p2_pokemon_state"].get("status", "nostatus") for t in timeline if t.get("p2_pokemon_state")]
            total_status = set(p1_status + p2_status)
            no_effect_status = {"nostatus", "noeffect"}
            negative_status = {s for s in total_status if s not in no_effect_status}
            #mean of negative status
            p1_negative_status_mean = np.mean([s in negative_status for s in p1_status])
            p2_negative_status_mean = np.mean([s in negative_status for s in p2_status])
            #status advantage if p1 applied more status to p2 (differenza delle medie dei negativi)
            features["p1_bad_status_advantage"] = p2_negative_status_mean-p1_negative_status_mean
            p1_status_change = np.sum(np.array(p1_status[1:]) != np.array(p1_status[:-1]))
            p2_status_change = np.sum(np.array(p2_status[1:]) != np.array(p2_status[:-1]))
            features["p1_status_change"] = p1_status_change
            features["p2_status_change"] = p2_status_change
            features["status_change_diff"] = p1_status_change - p2_status_change
            status_change_diff.append(features["status_change_diff"])

            p1_types = [t for p in p1_team for t in p.get("types", []) if t != "notype"]
            features["p1_type_diversity"] = len(set(p1_types))
            p1_type_resistance = compute_team_resistance(p1_team, type_chart)
            features["p1_type_resistance"] = p1_type_resistance
            p1_type_weakness = compute_team_weakness(p1_team, type_chart)
            features["p1_type_weakness"] = 1/p1_type_weakness
            res = compute_mean_stab_moves(timeline, pokemon_dict)
            features["p1_mean_stab"] = res["p1_mean_stab"]
            features["p2_mean_stab"] = res["p2_mean_stab"]
            features["diff_mean_stab"] = res["diff_mean_stab"]
            result = compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart)
            features["p1_type_advantage"] = result["p1_type_advantage"]
            features["p2_type_advantage"] = result["p2_type_advantage"]
            features["diff_type_advantage"] = result["diff_type_advantage"]
            MEDIUM_SPEED_THRESHOLD = 90 #medium-speed pokemon
            HIGH_SPEED_THRESHOLD = 100 #fast pokemon
            speeds = np.array([p.get("base_spe", 0) for p in p1_team])
            #restyle RIMOSSO 83.98% (+/- 0.47%) => 84.02% (+/- 0.50%)
            features["p1_avg_speed_stat_battaglia"] = np.mean(np.array(speeds) > MEDIUM_SPEED_THRESHOLD)
            #restyle RIMOSSO 84.02% (+/- 0.50%) => 84.03% (+/- 0.57%)
            features["p1_avg_high_speed_stat_battaglia"] = np.mean(np.array(speeds) > HIGH_SPEED_THRESHOLD)
        ##interaction features
        features.update(calculate_interaction_features(features))
        """fine create_feature_instance_final"""
        features["battle_id"] = battle_id
        if "player_won" in battle:
            features["player_won"] = int(battle["player_won"])
        battle_stats_results = calculate_battle_stats(battle)
        features.update(battle_stats_results)
        #features["diff_speed_first"] = battle_stats_results["diff_speed_first"]
        #features["diff_stat"] = battle_stats_results["diff_stat"]
        #features["avg_stat_p1"] = battle_stats_results["avg_stat_p1"]
        features["sum_stat_lead_p1"] = battle_stats_results["sum_stat_lead_p1"]
        features["sum_stat_lead_p2"] = battle_stats_results["sum_stat_lead_p2"]
        features["diff_stat_lead"] = battle_stats_results["diff_stat_lead"]
        hp_result = extract_hp_features(battle)
        features["p1_hp_pct_sum"] = hp_result["p1_hp_pct_sum"]
        features["p2_hp_pct_sum"] = hp_result["p2_hp_pct_sum"]
        features["diff_hp_pct"] = hp_result["diff_hp_pct"]
        features["hp_advantage_trend"] = hp_advantage_trend(battle)
        timeline = battle.get("battle_timeline", [])
        if timeline:
            """inizio create_feature_instance_final"""
            """fine create_feature_instance_final"""
            off_potential_result = compute_avg_offensive_potential(timeline, pokemon_dict, type_chart, is_test, battle_id)
            #type
            features["p1_type_advantage"] = off_potential_result["p1_type_advantage"]
            features["p2_type_advantage"] = off_potential_result["p2_type_advantage"]
            features["diff_type_advantage"] = off_potential_result["diff_type_advantage"]
            #status
            status_anomali_result = conta_status_anomali(timeline)
            features["status_p1"] = status_anomali_result["status_p1"]
            features["status_p2"] = status_anomali_result["status_p2"]
            features["diff_status"] = status_anomali_result["diff_status"]
            features["major_status_p1"] = status_anomali_result["major_status_p1"]
            features["major_status_p2"] = status_anomali_result["major_status_p2"]
            features["major_status_diff"] = status_anomali_result["major_status_diff"]
            #moves
            moves_result = get_move_features(timeline)
            features["p1_move_power_weighted"] = moves_result["p1_move_power_weighted"]
            features["p1_number_attacks"] = moves_result["p1_number_attacks"]
            features["p1_number_status"] = moves_result["p1_number_status"]
            features["p2_move_power_weighted"] = moves_result["p2_move_power_weighted"]
            features["p2_number_attacks"] = moves_result["p2_number_attacks"]
            features["p2_number_status"] = moves_result["p2_number_status"]
            #priority
            features["diff_number_attack"] = moves_result["diff_number_attack"]
            features["diff_number_status"] = moves_result["diff_number_status"]
            features["p1_sum_negative_priority"] = moves_result["p1_sum_negative_priority"]
            features["p2_sum_negative_priority"] = moves_result["p2_sum_negative_priority"]
            features["diff_negative_priority"] = moves_result["diff_negative_priority"]
            #boosts
            features.update(calcola_feature_boost(timeline))
            features.update(extract_boost_features(timeline,battle["p1_team_details"][0],battle["p2_lead_details"]))
            #effects (frequency and first occurrence)
            features.update(extract_effect_features_from_timeline(timeline, "p1"))
            features.update(extract_effect_features_from_timeline(timeline, "p2"))
        #features = dict(sorted(features.items()))
        features = shuffle_dict(features, seed=1234)
        feature_list.append(features)
    return pd.DataFrame(feature_list).fillna(0)
train_df = create_features(train_data)
test_df = create_features(test_data)
features = [col for col in train_df.columns if col not in ["battle_id", "player_won"]]
X = train_df[features]
y = train_df["player_won"]
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
    print(f"\nCalcolo permutation importances (Top {k})...")
    t0 = time.time()
    model.fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=10,
        random_state=1234,
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
    print(importance_df.head(20))
    print(f"[Permutation Importance completato in {time.time()-t0:.2f}s]")
    return list(top_features), importance_df
#COSTRUZIONE DEL VOTING MODEL (XGB + RF + LR)
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
        weights=[4, 1, 1],   #XGB più influente
        n_jobs=-1
    )
    return model
"""
TRAINING COMPLETO (feat selection => retrain =>  evaluate)
"""
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
def correlation_pruning(X, threshold=0.90):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropped {len(to_drop)} correlated features: {to_drop} (>{threshold}).")
    return [f for f in X.columns if f not in to_drop]
def final(prefix=""):
    selected = features
    X_selected = X[selected]
    model.fit(X_selected, y)
    final_pipe = model
    y_train_pred = final_pipe.predict(X_selected)
    y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]
    acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring="accuracy")
    auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring="roc_auc")
    end_time = time.time()
    print("featureArray,accuracy_score_training,roc_auc_score,accuracy_cross_val_score,roc_auc_cross_val_score")
    print(f"{[f for f in selected]},\n[{int(end_time-middle_time)}sec-{len(selected)}feat]\n{accuracy_score(y, y_train_pred)}->{acc.mean():.4f} ± {acc.std():.4f}, {roc_auc_score(y, y_train_proba)}->{auc.mean():.4f} ± {auc.std():.4f}")
    complete_prefix = prefix+str(int(10000*accuracy_score(y, y_train_pred)))+"_"+str(int(10000*acc.mean()))
    predict_and_submit(test_df, selected, final_pipe, prefix=complete_prefix)
    print(f"Total execution time: {int(end_time-start_time)} seconds")
#BASE
middle_time = time.time()
VOTING = True#False#True
NEW_VOTING = True#False#True
BASE = True#False#True
FINAL_VOTING = True#False#True
LOGISTIC = False#False#True
if LOGISTIC:
    #provo a usare il voting per la selezione delle feature e poi le passo alla logistic
    model, features, importance_table = train_with_feature_selection(
        X, y, k=80
    )
    X_reduced = X[features]
    features = correlation_pruning(X_reduced, threshold=0.92)
    selected = features
    X_selected = X[selected]
    final_pipe = train_regularization(X_selected,y)
    y_train_pred = final_pipe.predict(X_selected)
    y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]
    #CHECK OVERFITTING
    acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring="accuracy")
    auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring="roc_auc")
    end_time = time.time()
    print("featureArray,accuracy_score_training,roc_auc_score,accuracy_cross_val_score,roc_auc_cross_val_score")
    print(f"{[f for f in selected]},\n[{int(end_time-middle_time)}sec-{len(selected)}feat]\n{accuracy_score(y, y_train_pred)}->{acc.mean():.4f} ± {acc.std():.4f}, {roc_auc_score(y, y_train_proba)}->{auc.mean():.4f} ± {auc.std():.4f}")
    prefix = str(int(10000*accuracy_score(y, y_train_pred)))+"_"+str(int(10000*acc.mean()))
    
    predict_and_submit(test_df, features, final_pipe, prefix="LOGISTIC_FEATU_VOTING_")
elif FINAL_VOTING:
    model, features, importance_table = train_with_feature_selection(
        X, y, k=80
    )
    X_reduced = X[features]
    
    features = correlation_pruning(X_reduced, threshold=0.92)
    print("\nModello finale pronto!")
    final("FINAL_VOTING")
elif NEW_VOTING:
    #Logistic Regression (stabile per feature quasi lineari)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=1234
        ))
    ])
    #Random Forest (robusto, ottimo su high-dimensional noise)
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=1234
    )
    #XGBoost (il più forte)
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
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
        weights=[3, 1, 2],   #XGB più forte => peso maggiore
        n_jobs=-1
    )
    final()
elif VOTING:
    rf = RandomForestClassifier(
        n_estimators=1100,
        max_depth=5,                #più profondo
        min_samples_leaf=2,          #contro overfitting
        max_features=0.2,            #più decorrelazione
        bootstrap=True,
        class_weight="balanced",     #migliora roc_auc
        random_state=1234,
        n_jobs=-1
    )
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=500,
        learning_rate=0.5,
        random_state=1234
    )
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=5,
        random_state=1234
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        #("pca", PCA(n_components=0.95)), #keep 95% variance
        ("lr", LogisticRegression(
            C=1,
            max_iter=1000,
            solver="liblinear",
            #penalty="l1" 
        ))
    ])
    model = VotingClassifier(
        estimators=[
            ("rf", rf),
            #("ada", ada),
            ("gb", gb),
            ("lr", lr)
        ],
        voting="soft",
        weights=[1,1, 2],  #tune on performance
        n_jobs=-1
    )
    #scegli K = 40 (poi provo 30–60)
    features, importance_table = select_top_features(gb, X, y, k=90)
    #model = rf
    #model = gb
    #1112 850 918
    #2112 850 917
    #2113 852 917 rf,ada,gb,lr
    #213 tolto gb 847 916 rf,ada,lr
    #213 tolto ada 852 917 rf,gb,lr
    #113 850 917 rf,gb,lr
    #stacked_model = voting_model
    final()
elif BASE:
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
    model = StackingClassifier(
        estimators=[
            ("rf", rf),         
            ("gb", gb)          
        ],
        final_estimator=LogisticRegression(
            max_iter=2000, 
            C=0.05, 
            random_state=1234
        ), 
        passthrough=False, 
        n_jobs=-1
    )
    final()
else:
    rf = RandomForestClassifier(random_state=1234, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=1234)
    log_reg = LogisticRegression(random_state=1234, max_iter=2000)
   
    stacked_model_base = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=XGBClassifier(random_state=1234, n_estimators=100, learning_rate=0.05),
        passthrough=True,
        n_jobs=-1
    )
    param_grid = {
        #Random Forest 
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [None, 5, 10, 20],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        
        #Gradient Boosting 
        "gb__n_estimators": [100, 200, 300],
        "gb__learning_rate": [0.01, 0.05, 0.1],
        "gb__max_depth": [2, 3, 5],
        "gb__subsample": [0.8, 1.0],
        
        #XGBoost (meta-model) 
        "final_estimator__n_estimators": [100, 200, 300],
        "final_estimator__learning_rate": [0.01, 0.05, 0.1],
        "final_estimator__max_depth": [3, 5, 7],
        "final_estimator__subsample": [0.8, 1.0],
        "final_estimator__colsample_bytree": [0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    
    model = RandomizedSearchCV(
        estimator=stacked_model_base,
        param_distributions=param_grid,
        n_iter=50,  #prova anche con 100?
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        random_state=1234
    )
    final()