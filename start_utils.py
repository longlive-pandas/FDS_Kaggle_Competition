#restyle RIMOSSO indica le feature rimosse ora che faccio refactor del codice
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
def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)  # fallback
# superefficace 2
# non molto efficace 0.5
# non efficace 0
type_chart = {
    "Normal":     {"Rock":0.5, "Ghost":0.0, "Steel":0.5},
    "Fire":       {"Fire":0.5, "Water":0.5, "Grass":2.0, "Ice":2.0, "Bug":2.0, "Rock":0.5, "Dragon":0.5, "Steel":2.0},
    "Water":      {"Fire":2.0, "Water":0.5, "Grass":0.5, "Ground":2.0, "Rock":2.0, "Dragon":0.5},
    "Electric":   {"Water":2.0, "Electric":0.5, "Grass":0.5, "Ground":0.0, "Flying":2.0, "Dragon":0.5},
    "Grass":      {"Fire":0.5, "Water":2.0, "Grass":0.5, "Poison":0.5, "Ground":2.0, "Flying":0.5, "Bug":0.5, "Rock":2.0, "Dragon":0.5, "Steel":0.5},
    "Ice":        {"Fire":0.5, "Water":0.5, "Grass":2.0, "Ground":2.0, "Flying":2.0, "Dragon":2.0, "Steel":0.5},
    "Fighting":   {"Normal":2.0, "Ice":2.0, "Rock":2.0, "Dark":2.0, "Steel":2.0, "Poison":0.5, "Flying":0.5, "Psychic":0.5, "Bug":0.5, "Ghost":0.0, "Fairy":0.5},
    "Poison":     {"Grass":2.0, "Poison":0.5, "Ground":0.5, "Rock":0.5, "Ghost":0.5, "Steel":0.0, "Fairy":2.0},
    "Ground":     {"Fire":2.0, "Electric":2.0, "Grass":0.5, "Poison":2.0, "Flying":0.0, "Bug":0.5, "Rock":2.0, "Steel":2.0},
    "Flying":     {"Electric":0.5, "Grass":2.0, "Fighting":2.0, "Bug":2.0, "Rock":0.5, "Steel":0.5},
    "Psychic":    {"Fighting":2.0, "Poison":2.0, "Psychic":0.5, "Dark":0.0, "Steel":0.5},
    "Bug":        {"Fire":0.5, "Grass":2.0, "Fighting":0.5, "Poison":0.5, "Flying":0.5, "Psychic":2.0, "Ghost":0.5, "Dark":2.0, "Steel":0.5, "Fairy":0.5},
    "Rock":       {"Fire":2.0, "Ice":2.0, "Fighting":0.5, "Ground":0.5, "Flying":2.0, "Bug":2.0, "Steel":0.5},
    "Ghost":      {"Normal":0.0, "Psychic":2.0, "Ghost":2.0, "Dark":0.5},
    "Dragon":     {"Dragon":2.0, "Steel":0.5, "Fairy":0.0},
    "Dark":       {"Fighting":0.5, "Psychic":2.0, "Ghost":2.0, "Fairy":0.5},
    "Steel":      {"Fire":0.5, "Water":0.5, "Electric":0.5, "Ice":2.0, "Rock":2.0, "Fairy":2.0, "Steel":0.5},
    "Fairy":      {"Fire":0.5, "Fighting":2.0, "Poison":0.5, "Dragon":2.0, "Dark":2.0, "Steel":0.5}
}
def extract_features_by_importance(final_pipe, features):
    logreg = final_pipe.named_steps["logreg"]
    weights = logreg.coef_[0]   # shape (n_features,)
    intercept = logreg.intercept_[0]
    coef_df = pd.DataFrame({
        "feature": features,
        "weight": weights,
        "abs_weight": np.abs(weights)
    }).sort_values(by="abs_weight", ascending=False)
    return coef_df
#feature hp_advantage_trend
#83.87% (+/- 0.60%) => 83.89% (+/- 0.58%)
def hp_advantage_trend(battle):
    """Compute the linear slope of Player 1's HP advantage over 30 turns."""
    hp_adv = []
    for turn in battle['battle_timeline']:
        p1_hp = turn['p1_pokemon_state']['hp_pct']
        p2_hp = turn['p2_pokemon_state']['hp_pct']
        hp_adv.append(p1_hp - p2_hp)
    x = np.arange(len(hp_adv))
    slope, _, _, _, _ = linregress(x, hp_adv)
    return slope
def compute_mean_stab_moves(timeline, pokemon_dict):
    """
    Compute the mean number of STAB (Same Type Attack Bonus) moves used by P1 and P2
    over all turns of the battle timeline.

    Args:
        timeline (list): list of turn dictionaries
        pokemon_dict (dict): mapping {pokemon_name.lower(): [type1, type2]}

    Returns:
        dict: {
            "p1_mean_stab": float,
            "p2_mean_stab": float,
            "diff_mean_stab": float
        }
    """
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
            move_type = p1_move.get("type", "").title()
            p1_types = pokemon_dict.get(p1_name, [])
            # Check STAB
            if move_type in [t.title() for t in p1_types]:
                p1_stab_counts.append(1)
            else:
                p1_stab_counts.append(0)

        # --- Player 2 ---
        p2_state = turn.get("p2_pokemon_state", {})
        p2_move = turn.get("p2_move_details", {})
        if p2_state and p2_move:
            p2_name = p2_state.get("name", "").lower()
            move_type = p2_move.get("type", "").title()
            p2_types = pokemon_dict.get(p2_name, [])
            if move_type in [t.title() for t in p2_types]:
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

def compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart):
    if not timeline:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }

    p1_advantages = []
    p2_advantages = []
    debug_dict_p1 = {}
    debug_dict_p2 = {}
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
        debug_dict_p1[p1_name.lower()] = p1_types
        debug_dict_p2[p2_name.lower()] = p2_types
        # --- P1 attacking P2 ---
        # p1_mult = [
        #     type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0)
        #     for atk_type in p1_types for def_type in p2_types
        # ]
        # #print("p1_mult:",p1_mult)
        # p1_adv = np.mean(p1_mult) if p1_mult else 1.0
        p1_mult = []
        for atk_type in p1_types:
            mult = 1.0
            for def_type in p2_types:
                mult *= type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0)
            p1_mult.append(mult)
        #turn summary
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0

        # --- P2 attacking P1 ---
        # p2_mult = [
        #     type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0)
        #     for atk_type in p2_types for def_type in p1_types
        # ]
        # #print("p2_mult:",p2_mult)
        # p2_adv = np.mean(p2_mult) if p2_mult else 1.0
        # --- P2 attacking P1 ---
        p2_mult = []
        for atk_type in p2_types:
            mult = 1.0
            for def_type in p1_types:
                mult *= type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0)
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

# def compute_first_turn_type_advantage(timeline, pokemon_dict, type_chart):
#     if not timeline:
#         return {
#             "p1_type_advantage": 1.0,
#             "p2_type_advantage": 1.0,
#             "diff_type_advantage": 0.0
#         }

#     first_turn = timeline[0]
#     p1_name = first_turn.get("p1_pokemon_state", {}).get("name")
#     p2_name = first_turn.get("p2_pokemon_state", {}).get("name")

#     if not p1_name or not p2_name:
#         return {
#             "p1_type_advantage": 1.0,
#             "p2_type_advantage": 1.0,
#             "diff_type_advantage": 0.0
#         }

#     # Retrieve types from the dictionary
#     p1_types = pokemon_dict.get(p1_name.lower(), [])
#     p2_types = pokemon_dict.get(p2_name.lower(), [])

#     if not p1_types or not p2_types:
#         return {
#             "p1_type_advantage": 1.0,
#             "p2_type_advantage": 1.0,
#             "diff_type_advantage": 0.0
#         }

#     # --- Compute effectiveness of p1 attacking p2 ---
#     p1_mult = []
#     for atk_type in p1_types:
#         for def_type in p2_types:
#             p1_mult.append(type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0))
#     p1_adv = np.mean(p1_mult) if p1_mult else 1.0

#     # --- Compute effectiveness of p2 attacking p1 ---
#     p2_mult = []
#     for atk_type in p2_types:
#         for def_type in p1_types:
#             p2_mult.append(type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0))
#     p2_adv = np.mean(p2_mult) if p2_mult else 1.0
#     #exit()
#     return {
#         "p1_type_advantage": p1_adv,
#         "p2_type_advantage": p2_adv,
#         "diff_type_advantage": p1_adv - p2_adv
#     }

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
                multiplier *= type_chart.get(atk_type, {}).get(d_type.title(), 1.0)

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
    """
    Compute an average weakness score for a team of Pokémon.
    Higher = more weaknesses (more attack types that do >1x damage).

    If a Pokémon has many weaknesses (i.e. many types that deal >1x damage),
    the team obtains a higher weakness score.
    """
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
                multiplier *= type_chart.get(atk_type, {}).get(d_type.title(), 1.0)

            if multiplier > 1.0:  # super effective = weakness
                weak_to.append(atk_type)

        weakness_counts.append(len(weak_to))

    # Mean number of weaknesses per Pokémon
    mean_weakness = np.mean(weakness_counts) if weakness_counts else 0.0

    # Normalize to 0–1 range for consistency (optional)
    max_possible = len(all_attack_types)
    weakness_score = mean_weakness / max_possible if max_possible > 0 else mean_weakness

    return weakness_score

#83.66% (+/- 0.52%) => 83.89% (+/- 0.58%)
"""
battle duration before first faint happens
"""
def battle_duration(battle):
    return len([t for t in battle['battle_timeline'] if t['p1_pokemon_state']['hp_pct'] > 0 and
                                                     t['p2_pokemon_state']['hp_pct'] > 0])
def get_pokemon_stats(team, name):
    """Find Pokémon stats by name from a team list."""
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

def get_type_multiplier(move_type: str, defender_types: list, type_chart: dict) -> float:
    """Calculates the combined type effectiveness multiplier."""
    if not defender_types or move_type.upper() == 'NOTYPE':
        return 1.0
    
    multiplier = 1.0
    # Assumes type_chart is structured like: type_chart['ICE']['WATER'] = 0.5
    for def_type in defender_types:
        try:
            # Look up multiplier: TypeChart[Attacking Type][Defending Type]
            effectiveness = type_chart.get(move_type.upper(), {}).get(def_type.upper(), 1.0)
            multiplier *= effectiveness
        except:
            continue
            
    return multiplier

def calculate_expected_damage_ratio_turn_1(battle: dict, type_chart: dict) -> float:
    """
    Calculates the log-transformed expected damage advantage of P1 lead vs P2 lead in Turn 1.
    Positive values indicate P1 advantage; negative indicates P2 advantage.
    """
    timeline = battle.get('battle_timeline', [])
    p1_team = battle.get('p1_team_details', [])
    p2_lead = battle.get('p2_lead_details', {})
    
    # Check for minimum data required
    if not timeline or not p1_team or not p2_lead:
        # Log(1) = 0.0, representing neutral/no advantage
        return 0.0 

    turn_1 = timeline[0]
    p1_move = turn_1.get("p1_move_details")
    p2_move = turn_1.get("p2_move_details")

    p1_lead_stats = p1_team[0] 
    p2_lead_stats = p2_lead 

    p1_defender_types = [t for t in p1_lead_stats.get('types', []) if t != "notype"]
    p2_defender_types = [t for t in p2_lead_stats.get('types', []) if t != "notype"]

    p1_expected_damage = 0.0
    p2_expected_damage = 0.0
    
    # --- 1. Calculate P1 Damage Potential on P2 ---
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

    # --- 2. Calculate P2 Damage Potential on P1 ---
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

    # --- 3. Return Log-Transformed Advantage ---
    # Using the log difference: log(A+1) - log(B+1) = log((A+1) / (B+1))
    # This stabilizes the feature, handles zero damage, and converts ratios to a scale 
    # centered around 0.
    
    # Add a small smoothing constant (1.0) to prevent log(0) and division issues.
    p1_smoothed_damage = p1_expected_damage + 1.0
    p2_smoothed_damage = p2_expected_damage + 1.0
    
    log_advantage = np.log(p1_smoothed_damage) - np.log(p2_smoothed_damage)
    
    return log_advantage

from typing import Dict, List, Any

def calculate_status_efficacy_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates features related to Player 1's efficacy in applying and avoiding major status effects.
    """
    features = {}
    timeline = battle.get('battle_timeline', [])
    
    if not timeline:
        return {
            'p1_major_status_infliction_rate': 0.0,
            'p1_cumulative_major_status_turns_pct': 0.0
        }

    # Major Statuses that typically lead to missed turns (Gen 1 context: slp, frz)
    MAJOR_STATUSES = {'slp', 'frz'} 
    # check this they are statically defined Common Gen 1 status moves that inflict major status
    MAJOR_STATUS_MOVES = {'sleeppowder', 'spore', 'lovely kiss', 'sing'}
    
    p1_major_status_attempts = 0
    p1_major_status_successes = 0
    p1_major_status_turns_suffered = 0
    total_turns = len(timeline)

    for turn in timeline:
        
        # --- 1. Infliction Rate (P1 trying to hit P2) ---
        p1_move = turn.get("p1_move_details")
        p2_state = turn.get("p2_pokemon_state", {})
        p2_current_status = p2_state.get('status', 'nostatus')
        
        if p1_move:
            move_name = p1_move.get("name", "").lower()
            
            # Check for direct major status moves
            if move_name in MAJOR_STATUS_MOVES:
                p1_major_status_attempts += 1
                
                # Check if P2 ended the turn with a major status
                if p2_current_status in MAJOR_STATUSES:
                    p1_major_status_successes += 1
                    
        # --- 2. Cumulative Status Turns Suffered (P1 suffering) ---
        p1_state = turn.get("p1_pokemon_state", {})
        p1_current_status = p1_state.get('status', 'nostatus')
        
        if p1_current_status in MAJOR_STATUSES:
            p1_major_status_turns_suffered += 1

    # Calculate final features
    # first: => 84.01% (+/- 0.51%)
    # second: 83.80% (+/- 0.56%)
    # both: => 83.94% (+/- 0.52%)
    """
    0.1111111111111111
    0.0
    0.5
    1.0
    0.3333333333333333
    """
    p1_major_status_infliction_rate = 0.0
    if p1_major_status_attempts > 0:
        p1_major_status_infliction_rate = p1_major_status_successes / p1_major_status_attempts
    features['p1_major_status_infliction_rate'] = p1_major_status_infliction_rate
    #print(p1_major_status_infliction_rate)
    """
    0.0
    0.13333333333333333
    0.06666666666666667
    0.23333333333333334
    0.13333333333333333
    0.13333333333333333
    0.03333333333333333
    0.16666666666666666
    """
    p1_cumulative_major_status_turns_pct = 0.0
    if total_turns > 0:
        p1_cumulative_major_status_turns_pct = p1_major_status_turns_suffered / total_turns
    features['p1_cumulative_major_status_turns_pct'] = p1_cumulative_major_status_turns_pct
        
    return features

from typing import Dict, List, Any

def calculate_p2_status_control_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates features related to Player 2's efficacy in applying and suffering 
    major status effects (Sleep/Freeze) against Player 1.
    """
    features = {}
    timeline = battle.get('battle_timeline', [])
    
    if not timeline:
        return {
            'p2_major_status_infliction_rate': 0.0,
            'p2_cumulative_major_status_turns_pct': 0.0
        }

    MAJOR_STATUSES = {'slp', 'frz'} 
    # Common Gen 1 status moves that inflict major status
    MAJOR_STATUS_MOVES = {'sleeppowder', 'spore', 'lovely kiss', 'sing'}
    
    p2_major_status_attempts = 0
    p2_major_status_successes = 0
    p2_major_status_turns_suffered = 0
    total_turns = len(timeline)

    for turn in timeline:
        
        # --- 1. Infliction Rate (P2 trying to hit P1) ---
        p2_move = turn.get("p2_move_details")
        p1_state = turn.get("p1_pokemon_state", {})
        p1_current_status = p1_state.get('status', 'nostatus')
        
        if p2_move:
            move_name = p2_move.get("name", "").lower()
            
            if move_name in MAJOR_STATUS_MOVES:
                p2_major_status_attempts += 1
                
                # Check if P1 ended the turn with a major status
                if p1_current_status in MAJOR_STATUSES:
                    p2_major_status_successes += 1
                    
        # --- 2. Cumulative Status Turns Suffered (P2 suffering) ---
        p2_state = turn.get("p2_pokemon_state", {})
        p2_current_status = p2_state.get('status', 'nostatus')
        
        if p2_current_status in MAJOR_STATUSES:
            p2_major_status_turns_suffered += 1

    # Calculate final features
    # P2 Infliction Rate
    # without 84.01% (+/- 0.51%)
    # both => 84.25% (+/- 0.37%)
    # first => 84.08% (+/- 0.44%)
    # second => 84.11% (+/- 0.60%)
    
    if p2_major_status_attempts > 0:
        features['p2_major_status_infliction_rate'] = p2_major_status_successes / p2_major_status_attempts
    else:
        features['p2_major_status_infliction_rate'] = 0.0

    #P2 Cumulative Status Turns Percentage
    if total_turns > 0:
        features['p2_cumulative_major_status_turns_pct'] = p2_major_status_turns_suffered / total_turns
    else:
        features['p2_cumulative_major_status_turns_pct'] = 0.0
        
    return features
from typing import Dict, Any

def calculate_interaction_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates compound features based on existing metrics to capture net advantage.
    """
    #no one: 84.25% (+/- 0.33%)
    #1,2,3,4: (1)84.22% (+/- 0.38%)
    #1/2/3/4: (1)84.23% (+/- 0.33%), (2)84.25% (+/- 0.33%),
    #(3)84.25% (+/- 0.33%), (4)84.22% (+/- 0.39%)
    #1,2/1,3/1,4/2,3/2,4/3,4: (1,2)84.22% (+/- 0.31%), (2,3)84.25% (+/- 0.33%)
    #(2,4)84.24% (+/- 0.39%)
    #1,2,3/1,2,4/1,3,4/2,3,4: (1,2,3)84.22% (+/- 0.31%)
    # 1. Offensive Control Differential (Net Infliction)
    p1_inflict = features.get('p1_major_status_infliction_rate', 0.0)
    p2_inflict = features.get('p2_major_status_infliction_rate', 0.0)
    features['net_major_status_infliction'] = p1_inflict - p2_inflict

    # 2. Suffering Differential (Net Time Crippled)
    p1_suffered = features.get('p1_cumulative_major_status_turns_pct', 0.0)
    p2_suffered = features.get('p2_cumulative_major_status_turns_pct', 0.0)
    features['net_major_status_suffering'] = p2_suffered - p1_suffered
    #print(p2_suffered, p1_suffered, p2_suffered - p1_suffered)
    # # 3. Speed/Damage Interaction (Fast Sweeper Potential)
    p1_max_spe = features.get('p1_max_speed_stat', 0.0)
    p1_max_off = features.get('p1_max_offensive_stat', 0.0)
    features['p1_max_speed_offense_product'] = p1_max_spe * p1_max_off
    
    # 4. Final HP per KO Ratio (Adding +1 to avoid division by zero)
    p1_final_hp = features.get('p1_pct_final_hp', 0.0)
    p1_ko_count = features.get('nr_pokemon_sconfitti_p1', 0)
    features['p1_final_hp_per_ko'] = p1_final_hp / (p1_ko_count + 1)
    
    return features

from typing import Dict, Any, List
import numpy as np

def calculate_dynamic_boost_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates features tracking the dynamic stat boost advantage (e.g., Swords Dance, Amnesia)
    gained by P1 relative to P2 across the timeline.
    """
    features = {}
    timeline = battle.get('battle_timeline', [])
    
    if not timeline:
        return {
            'p1_net_boost_sum': 0.0,
            'p1_max_offense_boost_diff': 0.0,
            'p1_max_speed_boost_diff': 0.0
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
        p1_offense_boost = p1_boosts.get('atk', 0) + p1_boosts.get('spa', 0)
        p2_offense_boost = p2_boosts.get('atk', 0) + p2_boosts.get('spa', 0)
        offense_boost_diff_list.append(p1_offense_boost - p2_offense_boost)

        # P1 Speed Boost vs P2 Speed Boost
        p1_speed_boost = p1_boosts.get('spe', 0)
        p2_speed_boost = p2_boosts.get('spe', 0)
        speed_boost_diff_list.append(p1_speed_boost - p2_speed_boost)

    # 1. Net Cumulative Boost Sum
    # A positive sum indicates P1 spent more turns with a higher boost level than P2.
    features['p1_net_boost_sum'] = np.sum(net_boost_list)

    # 2. Maximum Offensive Boost Differential
    # Captures the peak offensive setup advantage.
    features['p1_max_offense_boost_diff'] = np.max(offense_boost_diff_list) if offense_boost_diff_list else 0.0

    # 3. Maximum Speed Boost Differential
    # Captures the peak speed advantage gained via boost moves.
    features['p1_max_speed_boost_diff'] = np.max(speed_boost_diff_list) if speed_boost_diff_list else 0.0
    
    return features

from typing import Dict, Any, List

def calculate_team_coverage_features(battle: Dict[str, Any], type_chart: Dict) -> Dict[str, float]:
    """
    Calculates P1's offensive coverage against P2's lead Pokémon.
    """
    features = {}
    p1_team = battle.get('p1_team_details', [])
    p2_lead = battle.get('p2_lead_details', {})
    
    if not p1_team or not p2_lead:
        return {'p1_team_super_effective_moves': 0.0}

    p2_defender_types = [t for t in p2_lead.get('types', []) if t != "notype"]
    super_effective_count = 0
    
    # We only check P1's *Pokémon types*, assuming they carry moves of their own type (STAB).
    # This is a strong proxy for offensive coverage.
    for p1_poke in p1_team:
        p1_poke_types = [t for t in p1_poke.get('types', []) if t != "notype"]
        has_super_effective_type = False
        
        for p1_type in p1_poke_types:
            # Check if this P1 type is Super Effective against any of P2's lead types
            type_mult = get_type_multiplier(p1_type, p2_defender_types, type_chart)
            if type_mult >= 2.0:
                has_super_effective_type = True
                break
        
        if has_super_effective_type:
            super_effective_count += 1
            
    features['p1_team_super_effective_moves'] = float(super_effective_count)
    return features

def calculate_action_efficiency_features(battle: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates P1's rate of using non-damaging (Status) moves.
    """
    features = {}
    timeline = battle.get('battle_timeline', [])
    
    if not timeline:
        return {'p1_status_move_rate': 0.0}

    p1_status_move_count = 0
    p1_total_moves = 0
    
    for turn in timeline:
        p1_move = turn.get("p1_move_details")
        
        if p1_move and p1_move.get("category"):
            p1_total_moves += 1
            if p1_move["category"].upper() == "STATUS":
                p1_status_move_count += 1
    
    if p1_total_moves > 0:
        features['p1_status_move_rate'] = p1_status_move_count / p1_total_moves
    else:
        features['p1_status_move_rate'] = 0.0
        
    return features

def create_feature_instance(battle, pokemon_dict, status_change_diff):
    battle_id = battle.get("battle_id")#debugging purposes
    features = {}
    # --- Player 1 Team Features ---
    p1_mean_hp = p1_mean_spe = p1_mean_atk = p1_mean_def = p1_mean_spd = p1_mean_spa = 0.0
    p1_lead_hp = p1_lead_spe = p1_lead_atk = p1_lead_def = p1_lead_spd = p1_lead_spa = 0.0
    p1_team = battle.get('p1_team_details', [])
    ####
    status_features = calculate_status_efficacy_features(battle)
    #83.97% (+/- 0.40%)=>84.01% (+/- 0.51%)
    features['p1_major_status_infliction_rate'] = status_features['p1_major_status_infliction_rate']
    #84.25% (+/- 0.37%) => 84.25% (+/- 0.33%)
    features['p1_cumulative_major_status_turns_pct'] = status_features['p1_cumulative_major_status_turns_pct']
    p2_status_features = calculate_p2_status_control_features(battle)
    features['p2_major_status_infliction_rate'] = p2_status_features['p2_major_status_infliction_rate']
    features['p2_cumulative_major_status_turns_pct'] = p2_status_features['p2_cumulative_major_status_turns_pct']
    
    ###
    dynamic_boost_features = calculate_dynamic_boost_features(battle)
    """
    Mancano 1,2; 1,3
    1,2,3 NO da 8394
    Fitting 5 folds for each of 34 candidates, totalling 170 fits
    Best params: {'logreg__C': 0.1, 'logreg__l1_ratio': 0.5, 'logreg__penalty': 'elasticnet', 'logreg__solver': 'saga'}
    Best CV mean: 0.8428 ± 0.0041 (da 8394 a 8429)
    It took 104.46463012695312 time

    1
    Fitting 5 folds for each of 34 candidates, totalling 170 fits
    Best params: {'logreg__C': 0.1, 'logreg__l1_ratio': 0.1, 'logreg__penalty': 'elasticnet', 'logreg__solver': 'saga'}
    Best CV mean: 0.8429 ± 0.0046 (da 8408 a 8430)
    It took 79.87716317176819 time
    """
    #features['p1_net_boost_sum'] = dynamic_boost_features['p1_net_boost_sum']
    """
    1,2 NO
    Best params: {'logreg__C': 0.1, 'logreg__l1_ratio': 0.5, 'logreg__penalty': 'elasticnet', 'logreg__solver': 'saga'}
    Best CV mean: 0.8428 ± 0.0043 (da 8397 a 8426)
    """
    """
    2,3 NO
    Best params: {'logreg__C': 10, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}
    Best CV mean: 0.8427 ± 0.0048 (da 8416 a 8451)
    It took 51.33575224876404 time
    """

    """
    2 molto buono
    Best params: {'logreg__C': 10, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}
    Best CV mean: 0.8427 ± 0.0042 (da 8419 a 8451)
    It took 52.9518678188324 time
    """
    features['p1_max_offense_boost_diff'] = dynamic_boost_features['p1_max_offense_boost_diff']
    """
    3
    Best CV mean: 0.8428 ± 0.0041 (da 8413 a 8444)
    It took 284.0617139339447 time
    """
    
    #features['p1_max_speed_boost_diff'] = dynamic_boost_features['p1_max_speed_boost_diff']
    

    ####
    """
    nessuno
    Best CV mean: 0.8442 ± 0.0041 (da 8419 a 8451)

    1,2
    Fitting 5 folds for each of 34 candidates, totalling 170 fits
    Best params: {'logreg__C': 30, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}
    Best CV mean: 0.8434 ± 0.0045 (da 8413 a 8434)
    """

    """
    1 CHOSEN (improved min, same max, decrease best)
    Fitting 5 folds for each of 34 candidates, totalling 170 fits
    Best params: {'logreg__C': 10, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}
    Best CV mean: 0.8427 ± 0.0042 (da 8420 a 8451)
    """

    """
    2
    Fitting 5 folds for each of 34 candidates, totalling 170 fits
    Best params: {'logreg__C': 30, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}
    Best CV mean: 0.8434 ± 0.0045 (da 8413 a 8435)
    """
    # Integrate Team Coverage (Step 4)
    #print("calculating p1_team_super_effective_moves")
    p1_team_super_effective_moves = calculate_team_coverage_features(battle, type_chart)
    features['p1_team_super_effective_moves'] = p1_team_super_effective_moves['p1_team_super_effective_moves']
    
    # Integrate Action Efficiency (Step 5)
    p1_status_move_rate = calculate_action_efficiency_features(battle)
    #features['p1_status_move_rate'] = p1_status_move_rate['p1_status_move_rate']
    ####
    ##
    expected_damage_ratio_turn_1 = 0.0
    try:
        """
        0.0
        4.61512051684126
        0.8873031950009029
        -3.7636451046866286
        """
        expected_damage_ratio_turn_1 = calculate_expected_damage_ratio_turn_1(battle, type_chart)
        #print(expected_damage_ratio_turn_1)
        #83.97% (+/- 0.52%)
        features['expected_damage_ratio_turn_1'] = expected_damage_ratio_turn_1
    except Exception:
        features['expected_damage_ratio_turn_1'] = 0.0
    ##
    if p1_team:
        ####
        # --- P1 Team Maximum Potential Features ---
        # Calculate Base Stat Total (BST) for each Pokémon
        bst_values = []
        
        # Collect max offense and max speed across the entire team
        max_offense = 0
        max_speed = 0

        for p in p1_team:
            p_hp = p.get('base_hp', 0)
            p_atk = p.get('base_atk', 0)
            p_def = p.get('base_def', 0)
            p_spa = p.get('base_spa', 0)
            p_spd = p.get('base_spd', 0)
            p_spe = p.get('base_spe', 0)
            
            # 1. Max Offensive Stat
            current_offense = max(p_atk, p_spa)
            max_offense = max(max_offense, current_offense)
            
            # 2. Max Speed Stat
            max_speed = max(max_speed, p_spe)
            
            # 3. BST for Variance Calculation
            bst_values.append(p_hp + p_atk + p_def + p_spa + p_spd + p_spe)
        #together 83.97% (+/- 0.52%)=>83.91% (+/- 0.44%)
        features['p1_max_offensive_stat'] = max_offense#83.97% (+/- 0.52%)=>83.97% (+/- 0.40%)
        #CHECK 84.30% (+/- 1.17%) => 84.31% (+/- 1.09%)
        features['p1_max_speed_stat'] = max_speed#83.97% (+/- 0.52%)=>83.95% (+/- 0.54%)
        
        #3. BST Variance (only calculated if there's more than one Pokémon)
        #var/std 83.97% (+/- 0.52%)=>83.91% (+/- 0.44%)/83.94% (+/- 0.44%)
        # if len(bst_values) > 1:
        #     features['p1_team_bst_variance'] = np.std(bst_values)
        # else:
        #     features['p1_team_bst_variance'] = 0.0 # Should be rare for a 6v6 battle
        ####
        p1_mean_hp = np.mean([p.get('base_hp', 0) for p in p1_team])
        p1_mean_spe = np.mean([p.get('base_spe', 0) for p in p1_team])
        p1_mean_atk = np.mean([p.get('base_atk', 1) for p in p1_team])
        p1_mean_def = np.mean([p.get('base_def', 0) for p in p1_team])
        p1_mean_spd = np.mean([p.get('base_spd', 0) for p in p1_team])

        features['p1_mean_hp'] = p1_mean_hp
        #restyle RIMOSSO 83.89% (+/- 0.55%) => 83.98% (+/- 0.47%)
        features['p1_mean_spe'] = p1_mean_spe
        features['p1_mean_atk'] = p1_mean_atk#83.88% (+/- 0.54%)
        features['p1_mean_def'] = p1_mean_def
        features['p1_mean_sp'] = p1_mean_spd

        #PER UN CONFRONTO EQUO UTILIZZIAMO SOLO DATI DEL LEADER ANCHE NELLA SQUADRA 1 PER LE DIFFERENZE
        p1_lead_hp =  p1_team[0].get('base_hp', 0)
        p1_lead_spe = p1_team[0].get('base_spe', 0)
        p1_lead_atk = p1_team[0].get('base_atk', 0)
        p1_lead_def = p1_team[0].get('base_def', 0)
        p1_lead_spd =  p1_team[0].get('base_spd', 0)


    # --- Player 2 Lead Features ---
    p2_hp = p2_spe = p2_atk = p2_def = p2_spd = 0.0
    p2_lead = battle.get('p2_lead_details')
    if p2_lead:
        
        # Player 2's lead Pokémon's stats
        p2_hp = p2_lead.get('base_hp', 0)
        p2_spe = p2_lead.get('base_spe', 0)
        p2_atk = p2_lead.get('base_atk', 0)
        p2_def = p2_lead.get('base_def', 0)
        p2_spd = p2_lead.get('base_spd', 0)

    # I ADD THE DIFFS/DELTAS
    features['diff_hp']  = p1_lead_hp  - p2_hp
    features['diff_spe'] = p1_lead_spe - p2_spe
    features['diff_atk'] = p1_lead_atk - p2_atk
    features['diff_def'] = p1_lead_def - p2_def#83.93% (+/- 0.53%) => 83.93% (+/- 0.52%)
    #CHECK 84.31% (+/- 1.09%) => 84.31% (+/- 1.09%)
    features['diff_spd'] =  p1_lead_spd - p2_spd#83.93% (+/- 0.53%) => 83.87% (+/- 0.57%)
    
    #informazioni dinamiche della battaglia
    #Chi mantiene più HP medi e conduce più turni spesso vince anche se la battaglia non è ancora finita
    timeline = battle.get('battle_timeline', [])
    if timeline:
        #boost+damage diff
        # --- Mean attack and defense boosts across timeline ---
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

        #new feature: confronta stat se disponibili
        #differenza di statistiche nei turni
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

            #base stats
            p1_stats = get_pokemon_stats(p1_team, p1_name) if p1_name else None
            p2_stats = None

            # p2: if same as lead, use lead; otherwise, None
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
        p1_hp = [t['p1_pokemon_state']['hp_pct'] for t in timeline if t.get('p1_pokemon_state')]
        p2_hp = [t['p2_pokemon_state']['hp_pct'] for t in timeline if t.get('p2_pokemon_state')]
        #vantaggio medio in salute (media della differenza tra la salute dei pokemon del primo giocatore e quella dei pokemon del secondo giocatore)
        features['hp_diff_mean'] = np.mean(np.array(p1_hp) - np.array(p2_hp))

        #percentuale di tempo in vantaggio (ovvero media dei booleani che indicano il vantaggio => proporzione del vantaggio)
        features['p1_hp_advantage_mean'] = np.mean(np.array(p1_hp) > np.array(p2_hp))#GRAN BELLA OPZIONE DI CLASSIFICAZIONE POSSIBILE APPLICAZIONE DI EFFETTI DI ETEROGENEITA

        #SUM OF FINAL HP PERCENTAGE OF EACH PLAYER
        p1_hp_final ={}
        p2_hp_final ={}
        for t in timeline:
            if t.get('p1_pokemon_state'):
                p1_hp_final[t['p1_pokemon_state']['name']]=t['p1_pokemon_state']['hp_pct']
            if t.get('p2_pokemon_state'):
                p2_hp_final[t['p2_pokemon_state']['name']]=t['p2_pokemon_state']['hp_pct']
        #numero di pokemon usati dal giocatore nei primi 30 turni
        features['p1_n_pokemon_use'] =len(p1_hp_final.keys())
        features['p2_n_pokemon_use'] =len(p2_hp_final.keys())
        #differenza nello schieramento pockemon dopo 30 turni
        features['diff_final_schieramento']=features['p1_n_pokemon_use']-features['p2_n_pokemon_use']
        nr_pokemon_sconfitti_p1 = np.sum([1 for e in list(p1_hp_final.values()) if e==0])
        nr_pokemon_sconfitti_p2 = np.sum([1 for e in list(p2_hp_final.values()) if e==0])
        features['nr_pokemon_sconfitti_p1'] = nr_pokemon_sconfitti_p1
        features['nr_pokemon_sconfitti_p2'] = nr_pokemon_sconfitti_p2
        #CHECK 84.31% (+/- 1.09%) => 84.35% (+/- 1.07%)
        features['nr_pokemon_sconfitti_diff'] = nr_pokemon_sconfitti_p1-nr_pokemon_sconfitti_p2
        #DOVREBBERO ESSERE BOMBA VITA DELLE DUE SQUADRE DOPO I 30 TURNI
        features['p1_pct_final_hp'] = np.sum(list(p1_hp_final.values()))+(6-len(p1_hp_final.keys()))
        features['p2_pct_final_hp'] = np.sum(list(p2_hp_final.values()))+(6-len(p2_hp_final.keys()))
        #SAREBBE CLAMOROSO NORMALIZZARLA ANCHE IN BASE ALLA DIFFERENZA DI VITA ASSOLUTA DEI POCKEMON LEADER DEI 2 PLAYER
        
        diff_final_hp = features['p1_pct_final_hp']-features['p2_pct_final_hp']
        #83.81% (+/- 0.52%) => 83.89% (+/- 0.55%)
        features['diff_final_hp'] = diff_final_hp

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
        features['early_hp_mean_diff'] = np.mean(np.array(p1_hp[:slice_idx]) - np.array(p2_hp[:slice_idx]))
        #83.94% (+/- 0.46%) => 83.98% (+/- 0.47%)
        features['late_hp_mean_diff'] = np.mean(np.array(p1_hp[-slice_idx:]) - np.array(p2_hp[-slice_idx:]))

        hp_delta = np.array(p1_hp) - np.array(p2_hp)
        features['hp_delta_trend'] = np.polyfit(range(len(hp_delta)), hp_delta, 1)[0]
        #83.87% (+/- 0.60%) => 83.89% (+/- 0.58%)
        features['hp_advantage_trend'] = hp_advantage_trend(battle)
        #fluttuazioni negli hp (andamento della partita: stabile o molto caotica)
        #restyle RIMOSSO p1_hp_std, p2_hp_std 83.89% (+/- 0.58%) => 83.89% (+/- 0.55%)
        features['p1_hp_std'] = np.std(p1_hp)
        features['p2_hp_std'] = np.std(p2_hp)
        features['hp_delta_std'] = np.std(hp_delta)

        ##STATUS (default nostatus, gli altri sono considerati negativi - i boost sono positivi)
        p1_status = [t['p1_pokemon_state'].get('status', 'nostatus') for t in timeline if t.get('p1_pokemon_state')]
        p2_status = [t['p2_pokemon_state'].get('status', 'nostatus') for t in timeline if t.get('p2_pokemon_state')]
        total_status = set(p1_status + p2_status)
        no_effect_status = {'nostatus', 'noeffect'}
        negative_status = {s for s in total_status if s not in no_effect_status}
        #mean of negative status
        p1_negative_status_mean = np.mean([s in negative_status for s in p1_status])
        p2_negative_status_mean = np.mean([s in negative_status for s in p2_status])
        #status advantage if p1 applied more status to p2 (differenza delle medie dei negativi)
        features['p1_bad_status_advantage'] = p2_negative_status_mean-p1_negative_status_mean
        #p1_bad_status_advantage.append(features['p1_bad_status_advantage'])
        #how many times status changed?
        # we have to check that first array shifted by 1 is
        # different from the same array excluding the last element
        # (so basically checking if status change in time)
        #somma il nr di volte in cui lo stato cambia, vedi se collineare
        p1_status_change = np.sum(np.array(p1_status[1:]) != np.array(p1_status[:-1]))
        p2_status_change = np.sum(np.array(p2_status[1:]) != np.array(p2_status[:-1]))
        #CHECK 84.31% (+/- 1.09%) =>  BOTH 84.34% (+/- 1.06%)
        #CHECK 84.31% (+/- 1.09%) =>  84.32% (+/- 1.06%)
        features['p1_status_change'] = p1_status_change
        #CHECK 84.31% (+/- 1.09%) => 84.33% (+/- 1.07%)
        features['p2_status_change'] = p2_status_change

        features['status_change_diff'] = p1_status_change - p2_status_change
        status_change_diff.append(features['status_change_diff'])

        #QUANTO IL TEAM è BILANCIATO (TIPI E VELOCITA)
        p1_types = [t for p in p1_team for t in p.get('types', []) if t != 'notype']
        #84.02% (+/- 0.58%) => 84.03% (+/- 0.57%)
        features['p1_type_diversity'] = len(set(p1_types))
        #!!
        p1_type_resistance = compute_team_resistance(p1_team, type_chart)
        #83.89% (+/- 0.58%)=>Mean CV accuracy: 83.90% (+/- 0.55%)
        features['p1_type_resistance'] = p1_type_resistance
        
        p1_type_weakness = compute_team_weakness(p1_team, type_chart)
        features['p1_type_weakness'] = 1/p1_type_weakness

        
        res = compute_mean_stab_moves(timeline, pokemon_dict)
        # print(res)
        # exit()
        #83.92% (+/- 0.53%) => 83.84% (+/- 0.66%)
        #CHECK 84.31% (+/- 1.09%) =>  84.34% (+/- 1.10%)
        features['p1_mean_stab'] = res["p1_mean_stab"]
        
        #83.92% (+/- 0.53%) => 83.83% (+/- 0.51%)
        #CHECK 84.34% (+/- 1.10%) =>  84.38% (+/- 1.11%)
        features['p2_mean_stab'] = res["p2_mean_stab"]

        #83.92% (+/- 0.53%) => 83.82% (+/- 0.55%)
        #CHECK 84.38% (+/- 1.11%) =>  84.40% (+/- 1.12%)
        features['diff_mean_stab'] = res["diff_mean_stab"]

        result = compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart)
        #CHECK 84.40% (+/- 1.12%) =>  84.35% (+/- 1.01%)
        features['p1_type_advantage'] = result['p1_type_advantage']
        #CHECK 84.35% (+/- 1.01%) =>  84.47% (+/- 1.01%)
        features['p2_type_advantage'] = result['p2_type_advantage']
        #CHECK 84.47% (+/- 1.01%) =>  84.45% (+/- 1.00%)
        features['diff_type_advantage'] = result['diff_type_advantage']

        #print("p1_type_vulnerability: ",p1_type_vulnerability)
        # 83.89% (+/- 0.58%) => 83.85% (+/- 0.50%)
        #features['p1_type_vulnerability'] = p1_type_vulnerability


        MEDIUM_SPEED_THRESHOLD = 90 #medium-speed pokemon
        HIGH_SPEED_THRESHOLD = 100 #fast pokemon
        speeds = np.array([p.get('base_spe', 0) for p in p1_team])
        #restyle RIMOSSO 83.98% (+/- 0.47%) => 84.02% (+/- 0.50%)
        features['p1_avg_speed_stat_battaglia'] = np.mean(np.array(speeds) > MEDIUM_SPEED_THRESHOLD)
        #restyle RIMOSSO 84.02% (+/- 0.50%) => 84.03% (+/- 0.57%)
        features['p1_avg_high_speed_stat_battaglia'] = np.mean(np.array(speeds) > HIGH_SPEED_THRESHOLD)


    ##interaction features
    #CHECK 84.45% (+/- 1.00%) => 84.44% (+/- 0.99%)
    features.update(calculate_interaction_features(features))
    # We also need the ID and the target variable (if it exists)
    
    # if is_test and battle_id == 109:
    #     with open(f"test_battle_{battle_id}_features.json", "w", encoding="utf-8") as f:
    #         json.dump(features, f, ensure_ascii=False, indent=4, default=default)
    #     exit()
    features['battle_id'] = battle_id
    if 'player_won' in battle:
        features['player_won'] = int(battle['player_won'])

    
    return features
def create_features(data: list[dict]) -> pd.DataFrame:
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
        features = create_feature_instance(battle, pokemon_dict, status_change_diff)
        feature_list.append(features)
    #print("pokemon_dict: ",pokemon_dict, len(pokemon_dict), "len data: ",len(data))
    #exit()
    return pd.DataFrame(feature_list).fillna(0)

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

def train_regularization(X, y, USE_PCA=False, POLY_ENABLED=False, seed=1234):
    # Build grid search pipeline
    print("build pipe")
    grid_search = build_pipe(USE_PCA=USE_PCA, POLY_ENABLED=POLY_ENABLED, seed=seed)
    print("pipe built")
    # Fit grid search
    grid_search.fit(X, y)

    # --- Show results ---
    print(f"Best params: {grid_search.best_params_}")
    mean_score = grid_search.best_score_
    std_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    print(f"Best CV mean: {mean_score:.4f} ± {std_score:.4f}")

    """
    49 features => 64 features
    Best params: {'logreg__C': 10, 'logreg__penalty': 'l2', 'logreg__solver': 'lbfgs'}
    Best CV score: 0.8450

    after stratified

    Best params: {'logreg__C': 10, 'logreg__penalty': 'l2', 'logreg__solver': 'lbfgs'}
    Best CV score: 0.8428
    """
    # --- Refit on all data automatically (refit=True) ---
    best_model = grid_search.best_estimator_
    """
    10/11/2025 9.36

    Best params: {'logreg__C': 10, 'logreg__penalty': 'l1', 'logreg__solver': 'liblinear'}
    Best CV mean: 0.9139 ± 0.0062
    Seed 1039284721: 0.8422 ± 0.0090
    Seed 398172634: 0.8431 ± 0.0069
    Seed 2750193806: 0.8431 ± 0.0079
    Seed 198234176: 0.8418 ± 0.0017
    Seed 4129837512: 0.8432 ± 0.0071
    Seed 1298374650: 0.8434 ± 0.0087
    Seed 3029487619: 0.8434 ± 0.0079
    Seed 718236451: 0.8449 ± 0.0026
    Seed 2543197682: 0.8430 ± 0.0093
    Seed 1765432987: 0.8423 ± 0.0066
    Seed 389124765: 0.8437 ± 0.0063
    Seed 612984372: 0.8417 ± 0.0057
    Seed 2983716540: 0.8430 ± 0.0059
    Seed 830174562: 0.8430 ± 0.0068
    Seed 1229837465: 0.8439 ± 0.0067
    Seed 4198372651: 0.8441 ± 0.0059
    Seed 2378164529: 0.8423 ± 0.0043
    Seed 3487612098: 0.8430 ± 0.0064
    Seed 954613287: 0.8428 ± 0.0049
    Seed 1864293754: 0.8424 ± 0.0045
    """
    return best_model
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

import random
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

def random_bucket_feature_search_robust(
    X, y,
    n_buckets=100,
    bucket_size=25,
    cv=5,
    seed_list=[42, 1234, 999, 2023],
    C=10,
    try_subsets=True,
    verbose=True
):
    """
    Leak-free robust random bucket feature selection.
    Score = min(CV mean accuracy across multiple seeds).
    """

    all_features = list(X.columns)
    bucket_records = []

    best_global_score = -np.inf
    best_global_features = None

    # --- Scoring function (fresh model every time, no leakage)
    def score_features(features):
        X_subset = X[features]
        seed_scores = []

        for seed in seed_list:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    C=C, penalty="l2", solver="lbfgs", max_iter=4000))
            ])

            scores = cross_val_score(
                pipe, X_subset, y, cv=skf, scoring='accuracy', n_jobs=-1
            )
            seed_scores.append(np.mean(scores))

        # robust metric
        return min(seed_scores)

    # --- Main bucket loop ---
    with open("bucket_results.txt", "w") as f:

        for i in range(1, n_buckets + 1):
            start_time = time.time()
            # Sample random features
            sampled = random.sample(all_features, min(bucket_size, len(all_features)))

            # score entire bucket
            bucket_score = score_features(sampled)
            best_bucket_score = bucket_score
            best_bucket_feats = sampled

            # Try internal subsets for local optimization
            if try_subsets:
                for k in range(bucket_size - 1, 5, -1):
                    candidate = random.sample(sampled, k)
                    candidate_score = score_features(candidate)

                    if candidate_score > best_bucket_score:
                        best_bucket_score = candidate_score
                        best_bucket_feats = candidate

            # Save results
            bucket_records.append({
                "bucket": i,
                "score": best_bucket_score,
                "n_features": len(best_bucket_feats),
                "features": best_bucket_feats,
            })

            # Logging
            if verbose:
                print(f"Bucket {i}/{n_buckets} → robust CV={best_bucket_score:.4f} "
                      f"({len(best_bucket_feats)} features)")
                f.write(f"Bucket {i}: {best_bucket_score:.4f} "
                        f"({len(best_bucket_feats)} features)\n")

            # update global best
            if best_bucket_score > best_global_score:
                best_global_score = best_bucket_score
                best_global_features = best_bucket_feats

            # timing
            elapsed = time.time() - start_time
            if verbose:
                print(f"Time: {elapsed:.2f}s")
            f.write(f"Time: {elapsed:.2f}s\n")
            f.flush()
            os.fsync(f.fileno())

    # final results
    df = pd.DataFrame(bucket_records).sort_values(
        by="score", ascending=False
    ).reset_index(drop=True)

    print("\n✅ Best subset found!")
    print(f"Score: {best_global_score:.4f} using {len(best_global_features)} features")
    print(best_global_features)

    return {
        "best_score": best_global_score,
        "best_features": best_global_features,
        "bucket_scores": df
    }

from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
import pandas as pd
import time

def greedy_feature_selection_dynamicC(
    X, y,
    cv=5,
    seed_list=[42, 1234, 999, 2023],
    C_grid=[0.1, 1, 3, 10, 30],
    min_delta=0.0005,
    verbose=True
):
    """
    Greedy forward feature selection with dynamic C tuning.
    Uses robustness metric: min(mean CV accuracy across seeds).
    
    Args:
        X: pd.DataFrame
        y: pd.Series
        cv: # folds
        seed_list: seeds for robust scoring
        C_grid: candidate C values for tuning
        min_delta: min improvement required to accept feature
        verbose: print progress
    
    Returns:
        selected_features, history_df
    """

    start_time = time.time()

    remaining = list(X.columns)
    selected = []
    best_score = 0.0
    history = []

    iteration = 0

    while remaining:
        iteration += 1
        scores_with_candidates = []

        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Remaining features: {len(remaining)} | Selected so far: {len(selected)}")

        for f in remaining:
            candidate_features = selected + [f]
            X_subset = X[candidate_features]

            # Evaluate all C values for all seeds
            C_scores = []  # will store robust score per C

            for C in C_grid:
                seed_scores = []

                for seed in seed_list:
                    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("logreg", LogisticRegression(
                            solver="lbfgs",
                            penalty="l2",
                            C=C,
                            max_iter=4000
                        ))
                    ])

                    cv_scores = cross_val_score(
                        pipe,
                        X_subset, y,
                        cv=kfold,
                        scoring='accuracy',
                        n_jobs=-1
                    )

                    seed_scores.append(np.mean(cv_scores))

                # robust score for this C = min mean across seeds
                C_scores.append(min(seed_scores))

            # best robust C score for this feature
            best_C_score = max(C_scores)
            scores_with_candidates.append((f, best_C_score))

        # Best feature this iteration
        best_candidate, best_candidate_score = max(scores_with_candidates, key=lambda x: x[1])
        delta = best_candidate_score - best_score

        if verbose:
            print(f"Best candidate: {best_candidate} | robust score = {best_candidate_score:.4f} | Δ = {delta:.4f}")

        if delta > min_delta:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            if verbose:
                print(f" ✅ Accepted feature '{best_candidate}'. New best robust score: {best_score:.4f}")
        else:
            if verbose:
                print(" ⏹️ No meaningful improvement. Stopping.")
            break

        history.append((iteration, len(selected), best_score))

    total_time = time.time() - start_time
    if verbose:
        print(f"\nTotal time: {total_time:.2f} seconds")

    return selected, pd.DataFrame(history, columns=["iteration", "n_features", "robust_cv_accuracy"])

def simple_train(X,y): 
    pipe = build_pipe()
    #kfold cross-validation 
    kfold = KFold(n_splits=5, shuffle=True, random_state=1234) 
    # 5-fold CV 
    print("Training Logistic Regression con 5-Fold Cross-Validation...\n") 
    scores = cross_val_score(pipe, X, y, cv=kfold, scoring='accuracy', n_jobs=-1) 
    print(f"Cross-validation accuracies: {np.round(scores, 4)}") 
    print(f"Mean CV accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)") 
    #Training finale 
    pipe.fit(X, y) 
    print("\nFinal model trained on all training data.") 
    return pipe

# def build_pipe(USE_PCA=False, POLY_ENABLED=False):
#     steps = []
#     if POLY_ENABLED:
#         steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
#     steps.append(("scaler", StandardScaler()))
#     if USE_PCA:
#         steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
#     steps.append(("logreg", LogisticRegression(max_iter=2000, random_state=1234)))

#     return Pipeline(steps)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np

"""
10/11/2025 9.16
Fitting 5 folds for each of 34 candidates, totalling 170 fits
Best params: {'logreg__C': 30, 'logreg__penalty': 'l1', 'logreg__solver': 'liblinear'}
Best CV mean: 0.8442 ± 0.0039
Seed 1039284721: 0.8429 ± 0.0070
Seed 398172634: 0.8436 ± 0.0075
Seed 2750193806: 0.8435 ± 0.0092
Seed 198234176: 0.8429 ± 0.0032
Seed 4129837512: 0.8436 ± 0.0078
Seed 1298374650: 0.8439 ± 0.0096
Seed 3029487619: 0.8433 ± 0.0094
Seed 718236451: 0.8456 ± 0.0028
Seed 2543197682: 0.8435 ± 0.0099
Seed 1765432987: 0.8443 ± 0.0093
Seed 389124765: 0.8440 ± 0.0077
Seed 612984372: 0.8424 ± 0.0062
Seed 2983716540: 0.8428 ± 0.0054
Seed 830174562: 0.8430 ± 0.0071
Seed 1229837465: 0.8445 ± 0.0061
Seed 4198372651: 0.8442 ± 0.0057
Seed 2378164529: 0.8433 ± 0.0040
Seed 3487612098: 0.8442 ± 0.0069
Seed 954613287: 0.8438 ± 0.0041
Seed 1864293754: 0.8438 ± 0.0037
"""
def build_pipe(USE_PCA=False, POLY_ENABLED=False, seed=1234):
    """
    Builds a logistic regression pipeline and runs grid search + stability checks.
    Returns: the best_model (fitted) + best_params + stability report
    """

    # --- Pipeline construction ---
    steps = []
    if POLY_ENABLED:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    steps.append(("scaler", StandardScaler()))

    if USE_PCA:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))

    # Base estimator placeholder (solver chosen during grid search)
    steps.append(("logreg", LogisticRegression(max_iter=4000, random_state=seed)))
    pipe = Pipeline(steps)
    param_grid = [
        {
            'logreg__solver': ['liblinear'],
            'logreg__penalty': ['l1', 'l2'],
            'logreg__C': [0.01, 0.1, 1, 10],
        },
        {
            'logreg__solver': ['lbfgs'],
            'logreg__penalty': ['l2'],
            'logreg__C': [0.01, 0.1, 1, 10],
        },
    ]
    # --- Grid search with stratified 5-fold CV ---
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # grid_search = GridSearchCV(
    #     estimator=pipe,
    #     param_grid=param_grid,
    #     scoring="accuracy",
    #     cv=kfold,
    #     n_jobs=-1,
    #     verbose=1,
    #     refit=True
    # )
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',#roc_auc#accuracy
        n_jobs=4,        # use 4 cores in parallel
        cv=kfold,            # 5-fold cross-validation, more on this later
        refit=True,      # retrain the best model on the full training set
        return_train_score=True
    )

    return grid_search  # not fitted yet — caller will call `fit(X, y)`
def predict_and_submit(test_df, features, pipe):
    # Make predictions on the real test data
    X_test = test_df[features]
    print("Generating predictions on the test set...")
    test_predictions = pipe.predict(X_test)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    # Save submission CSV
    submission_df.to_csv('submission.csv', index=False)
    print("\n'submission.csv' file created successfully!")
    #display(submission_df.head())
