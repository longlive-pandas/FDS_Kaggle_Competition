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

    for turn in timeline:
        p1_name = turn.get("p1_pokemon_state", {}).get("name")
        p2_name = turn.get("p2_pokemon_state", {}).get("name")

        if not p1_name or not p2_name:
            continue

        # Get types from dictionary
        p1_types = pokemon_dict.get(p1_name.lower(), [])
        p2_types = pokemon_dict.get(p2_name.lower(), [])

        if not p1_types or not p2_types:
            continue

        # --- P1 attacking P2 ---
        p1_mult = [
            type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0)
            for atk_type in p1_types for def_type in p2_types
        ]
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0

        # --- P2 attacking P1 ---
        p2_mult = [
            type_chart.get(atk_type.title(), {}).get(def_type.title(), 1.0)
            for atk_type in p2_types for def_type in p1_types
        ]
        p2_adv = np.mean(p2_mult) if p2_mult else 1.0

        p1_advantages.append(p1_adv)
        p2_advantages.append(p2_adv)

    if not p1_advantages or not p2_advantages:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }

    # Compute averages over all turns
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
            features['p1_pct_final_hp'] =np.sum(list(p1_hp_final.values()))+(6-len(p1_hp_final.keys()))
            features['p2_pct_final_hp'] =np.sum(list(p2_hp_final.values()))+(6-len(p1_hp_final.keys()))
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
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

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

def train_regularization(X, y):
    USE_PCA = False
    POLY_ENABLED = False
    seed = 1234
    # --- Build base pipeline ---
    steps = []
    if POLY_ENABLED:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    steps.append(("scaler", StandardScaler()))
    if USE_PCA:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))

    steps.append(("logreg", LogisticRegression(max_iter=2000, random_state=seed)))
    pipe = Pipeline(steps)

    # --- Define parameter grid for GridSearchCV ---
    param_grid = [
        # --- L1 and L2 with liblinear (good for sparse selection + small datasets) ---
        {
            'logreg__solver': ['liblinear'],
            'logreg__penalty': ['l1', 'l2'],
            'logreg__C': [0.01, 0.1, 1, 3, 10, 30],
        },

        # --- Pure L2 with lbfgs (fast, robust, handles many features well) ---
        {
            'logreg__solver': ['lbfgs'],
            'logreg__penalty': ['l2'],  # only L2 is valid with lbfgs
            'logreg__C': [0.01, 0.1, 1, 3, 10, 30, 100],
        },

        # --- ElasticNet with saga (good when features are noisy + correlated) ---
        {
            'logreg__solver': ['saga'],
            'logreg__penalty': ['elasticnet'],
            'logreg__l1_ratio': [0.1, 0.5, 0.9],
            'logreg__C': [0.01, 0.1, 1, 3, 10],
            # saga needs more iterations for convergence
        }
    ]



    # --- Create GridSearchCV wrapper ---
    #StratifiedKFold preserves class balance
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',    # or 'roc_auc' if binary classification
        cv=kfold,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    # --- Fit grid search ---
    #print("🔍 Performing Grid Search...")
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
    for s in [42, 1234, 999, 2023]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        scores = cross_val_score(best_model, X, y, cv=skf)
        print(f"Seed {s}: {scores.mean():.4f} ± {scores.std():.4f}")
    return best_model
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

import random
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

def random_bucket_feature_search(
    X, y, base_pipe, n_buckets=100, bucket_size=25, cv=5,
    try_subsets=True, verbose=True
):
    """
    Random feature subset (bucket) selection.
    Repeats random sampling of feature subsets and returns the best-performing one.

    Args:
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series): Target labels.
        base_pipe: sklearn pipeline (e.g., Logistic Regression pipeline).
        n_buckets (int): Number of random subsets to try.
        bucket_size (int): Number of features per random subset.
        cv (int): Cross-validation folds.
        try_subsets (bool): If True, try reducing each bucket to smaller subsets for better score.
        verbose (bool): Print progress info.

    Returns:
        dict: {
            "best_score": float,
            "best_features": list,
            "bucket_scores": pd.DataFrame
        }
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    all_features = list(X.columns)
    bucket_results = []

    best_score = -np.inf
    best_features = []

    for i in range(1, n_buckets + 1):
        # --- Step 1: pick random features ---
        sampled_features = random.sample(all_features, min(bucket_size, len(all_features)))
        X_subset = X[sampled_features]

        # --- Step 2: evaluate CV accuracy ---
        base_score = np.mean(cross_val_score(base_pipe, X_subset, y, cv=kfold, scoring='accuracy', n_jobs=-1))

        # --- Step 3: optional refinement (greedy reduction within the bucket) ---
        best_bucket_score = base_score
        best_bucket_features = sampled_features

        if try_subsets:
            for j in range(len(sampled_features) - 1, 5, -1):  # test smaller subsets
                reduced_features = random.sample(sampled_features, j)
                reduced_score = np.mean(cross_val_score(base_pipe, X[reduced_features], y, cv=kfold, scoring='accuracy', n_jobs=-1))
                if reduced_score > best_bucket_score:
                    best_bucket_score = reduced_score
                    best_bucket_features = reduced_features

        bucket_results.append({
            "bucket": i,
            "score": best_bucket_score,
            "n_features": len(best_bucket_features),
            "features": best_bucket_features
        })

        if verbose:
            print(f"Bucket {i:3d} → CV={best_bucket_score:.4f} ({len(best_bucket_features)} features)")

        # --- Step 4: update global best ---
        if best_bucket_score > best_score:
            best_score = best_bucket_score
            best_features = best_bucket_features

    results_df = pd.DataFrame(bucket_results).sort_values(by="score", ascending=False).reset_index(drop=True)

    if verbose:
        print("\n🏆 Best bucket found:")
        print(f"Score: {best_score:.4f} with {len(best_features)} features")
        print("Top features:", best_features)

    return {
        "best_score": best_score,
        "best_features": best_features,
        "bucket_scores": results_df
    }

def greedy_feature_selection(X, y, base_pipe, cv=5, min_delta=0.0005, verbose=True):
    """
    Greedy forward feature selection using cross-validation score.
    
    Args:
        X (pd.DataFrame): feature dataframe
        y (pd.Series): target
        base_pipe: sklearn pipeline (e.g. your Logistic Regression pipeline)
        cv (int): number of cross-validation folds
        min_delta (float): minimum improvement required to add a feature
        verbose (bool): print progress
        
    Returns:
        selected_features (list): features chosen by the greedy algorithm
        history (pd.DataFrame): accuracy progression per iteration
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    remaining = list(X.columns)
    selected = []#['p1_type_weakness']
    best_score = 0.0
    history = []

    iteration = 0
    while remaining:
        iteration += 1
        scores_with_candidates = []

        # Evaluate adding each remaining feature
        for f in remaining:
            candidate_features = selected + [f]
            X_subset = X[candidate_features]
            score = np.mean(cross_val_score(base_pipe, X_subset, y, cv=kfold, scoring='accuracy', n_jobs=-1))
            scores_with_candidates.append((f, score))

        # Pick the best feature this round
        best_candidate, best_candidate_score = max(scores_with_candidates, key=lambda x: x[1])
        delta = best_candidate_score - best_score

        if delta > min_delta:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            if verbose:
                print(f"Step {iteration}: ➕ Added '{best_candidate}' → mean CV={best_score:.4f} (+{delta:.4f})")
        else:
            if verbose:
                print(f"\n⏹️  No improvement at step {iteration}. Stopping selection.")
            break

        history.append((iteration, len(selected), best_score))

    return selected, pd.DataFrame(history, columns=["iteration", "n_features", "cv_accuracy"])

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
def build_pipe(USE_PCA=False, POLY_ENABLED=False):
    steps = []
    if POLY_ENABLED:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("scaler", StandardScaler()))
    if USE_PCA:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    steps.append(("logreg", LogisticRegression(max_iter=2000, random_state=1234)))

    return Pipeline(steps)

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
