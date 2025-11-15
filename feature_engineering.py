import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import linregress
from typing import Dict, Iterable
import random
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import linregress
import random
from typing import Dict, Any

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
BOOST_MULT = {
    -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
    0: 1.0,
    1: 1.5,  2: 2.0,  3: 2.5,  4: 3.0,  5: 3.5,  6: 4.0
}
important_effects = [
    "substitute", "reflect", "light_screen",
    "leech_seed", "bind", "wrap", "clamp",
    "confusion", "toxic", "poison", "burn", "paralysis"
]

def calculate_interaction_features(features):
    """
    Generate interaction features without modifying the original feature dictionary.
    """
    # Statuts-related interactions
    p1_inflict = features.get("p1_major_status_infliction_rate", 0.0)
    p2_inflict = features.get("p2_major_status_infliction_rate", 0.0)

    p1_suffered = features.get("p1_cumulative_major_status_turns_pct", 0.0)
    p2_suffered = features.get("p2_cumulative_major_status_turns_pct", 0.0)

    # Offensive/Speed interactions
    p1_max_spe = features.get("p1_max_speed_stat", 0.0)
    p1_max_off = features.get("p1_max_offensive_stat", 0.0)

    # Final HP / KO ratio
    p1_final_hp = features.get("p1_pct_final_hp", 0.0)
    p1_ko_count = features.get("nr_pokemon_sconfitti_p1", 0)

    return {
        "net_major_status_infliction": p1_inflict - p2_inflict,
        "net_major_status_suffering": p2_suffered - p1_suffered,
        "p1_max_speed_offense_product": p1_max_spe * p1_max_off,
        "p1_final_hp_per_ko": p1_final_hp / (p1_ko_count + 1)
    }
def compute_status_features(timeline):
    """
    Unifica:
      - conta_status_anomali
      - calculate_status_efficacy_features
      - calculate_p2_status_control_features

    Restituisce TUTTE le feature sullo status in un'unica passata.
    """

    # Default response for empty timeline
    if not timeline:
        return {
            # Counts
            "status_p1": 0,
            "status_p2": 0,
            "diff_status": 0,
            "major_status_p1": 0,
            "major_status_p2": 0,
            "major_status_diff": 0,

            # Status change
            "p1_status_change": 0,
            "p2_status_change": 0,
            "status_change_diff": 0,

            # P1 infliction
            "p1_major_status_infliction_rate": 0.0,
            "p1_cumulative_major_status_turns_pct": 0.0,

            # P2 infliction
            "p2_major_status_infliction_rate": 0.0,
            "p2_cumulative_major_status_turns_pct": 0.0,

            # Advantage (negative status mean)
            "p1_bad_status_advantage": 0.0,
        }

    MAJOR_STATUSES = {"slp", "frz"}
    MAJOR_STATUS_MOVES = {"sleeppowder", "spore", "lovely kiss", "sing"}
    NO_EFFECT = {"nostatus", "noeffect"}

    # Counts
    status_count_p1 = 0
    status_count_p2 = 0
    major_count_p1 = 0
    major_count_p2 = 0

    # Status lists (for change detection & negative mean)
    p1_status_list = []
    p2_status_list = []

    # Major status attempts and successes
    p1_attempt = p1_success = 0
    p2_attempt = p2_success = 0

    # Cumulative major status turns (suffered)
    p1_major_suffer = 0
    p2_major_suffer = 0

    for turn in timeline:
        # P1
        p1_state = turn.get("p1_pokemon_state", {})
        s1 = p1_state.get("status", "nostatus").lower()
        p1_status_list.append(s1)

        if s1 not in NO_EFFECT:
            status_count_p1 += 1
            if s1 in MAJOR_STATUSES:
                major_count_p1 += 1
                p1_major_suffer += 1

        # P2
        p2_state = turn.get("p2_pokemon_state", {})
        s2 = p2_state.get("status", "nostatus").lower()
        p2_status_list.append(s2)

        if s2 not in NO_EFFECT:
            status_count_p2 += 1
            if s2 in MAJOR_STATUSES:
                major_count_p2 += 1
                p2_major_suffer += 1

        # P1 Infliction Attempt (P1 hitting P2)
        m1 = turn.get("p1_move_details")
        if m1:
            name = m1.get("name", "").lower()
            if name in MAJOR_STATUS_MOVES:
                p1_attempt += 1
                if s2 in MAJOR_STATUSES:
                    p1_success += 1

        # P2 Infliction Attempt (P2 hitting P1)
        m2 = turn.get("p2_move_details")
        if m2:
            name = m2.get("name", "").lower()
            if name in MAJOR_STATUS_MOVES:
                p2_attempt += 1
                if s1 in MAJOR_STATUSES:
                    p2_success += 1

    total_turns = len(timeline)

    # ------------------------------------------------------
    # Compute derived features
    # ------------------------------------------------------

    # Status changes
    p1_status_change = int(np.sum(np.array(p1_status_list[1:]) != np.array(p1_status_list[:-1])))
    p2_status_change = int(np.sum(np.array(p2_status_list[1:]) != np.array(p2_status_list[:-1])))

    # Negative status means
    total_status_set = set(p1_status_list + p2_status_list)
    negative_status = {s for s in total_status_set if s not in NO_EFFECT}

    p1_negative_mean = np.mean([s in negative_status for s in p1_status_list])
    p2_negative_mean = np.mean([s in negative_status for s in p2_status_list])

    # Infliction rates
    p1_infliction_rate = (p1_success / p1_attempt) if p1_attempt > 0 else 0.0
    p2_infliction_rate = (p2_success / p2_attempt) if p2_attempt > 0 else 0.0

    # Cumulative suffers
    p1_cumulative_pct = p1_major_suffer / total_turns
    p2_cumulative_pct = p2_major_suffer / total_turns

    # ------------------------------------------------------
    # Return unified result
    # ------------------------------------------------------
    return {
        # Basic counts
        "status_p1": status_count_p1,
        "status_p2": status_count_p2,
        "diff_status": status_count_p1 - status_count_p2,

        "major_status_p1": major_count_p1,
        "major_status_p2": major_count_p2,
        "major_status_diff": major_count_p1 - major_count_p2,

        # Changes
        "p1_status_change": p1_status_change,
        "p2_status_change": p2_status_change,
        "status_change_diff": p1_status_change - p2_status_change,

        # Negative status advantage
        "p1_bad_status_advantage": p2_negative_mean - p1_negative_mean,

        # Infliction
        "p1_major_status_infliction_rate": p1_infliction_rate,
        "p1_cumulative_major_status_turns_pct": p1_cumulative_pct,
        "p2_major_status_infliction_rate": p2_infliction_rate,
        "p2_cumulative_major_status_turns_pct": p2_cumulative_pct,
    }
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
def calculate_team_coverage_features(battle, type_chart):
    features = {}
    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details", {})
    if not p1_team or not p2_lead:
        return {"p1_team_super_effective_moves": 0.0}
    p2_defender_types = [t for t in p2_lead.get("types", []) if t != "notype"]
    super_effective_count = 0
    for p1_poke in p1_team:
        p1_poke_types = [t for t in p1_poke.get("types", []) if t != "notype"]
        has_super_effective_type = False
        for p1_type in p1_poke_types:
            type_mult = get_type_multiplier(p1_type, p2_defender_types, type_chart)
            if type_mult >= 2.0:
                has_super_effective_type = True
                break
        if has_super_effective_type:
            super_effective_count += 1
    features["p1_team_super_effective_moves"] = float(super_effective_count)
    return features
def calculate_action_efficiency_features(battle: Dict[str, Any]) -> Dict[str, float]:
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
def compute_avg_offensive_potential(timeline, pokemon_dict):
    if not timeline:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }
    p1_advantages = []
    p2_advantages = []
    all_move_types = list(type_chart.keys())
    for turn in timeline:
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
def extract_full_hp_features(timeline, team_size=6):
    if not timeline:
        # fallback per battaglie vuote
        return {k: 0.0 for k in [
            "hp_diff_mean", "p1_hp_advantage_mean",
            "p1_n_pokemon_use", "p2_n_pokemon_use",
            "diff_final_schieramento", "nr_pokemon_sconfitti_p1",
            "nr_pokemon_sconfitti_p2", "nr_pokemon_sconfitti_diff",
            "p1_pct_final_hp", "p2_pct_final_hp", "diff_final_hp",
            "battle_duration", "hp_loss_rate",
            "early_hp_mean_diff", "late_hp_mean_diff",
            "hp_delta_trend", "p1_hp_std", "p2_hp_std", "hp_delta_std",
            "hp_volatility", "momentum_shift", "p1_momentum_phases"
        ]}

    hp_deltas = [t["p1_pokemon_state"]["hp_pct"] - t["p2_pokemon_state"]["hp_pct"] 
                 for t in timeline]
    
    # Conta i cambi di segno (shift di momentum)
    momentum_shifts = sum(1 for i in range(1, len(hp_deltas)) 
                         if hp_deltas[i] * hp_deltas[i-1] < 0)
    # Fasi di vantaggio P1
    p1_advantage_phases = sum(1 for delta in hp_deltas if delta > 10)
    p1_hp = [t["p1_pokemon_state"]["hp_pct"] for t in timeline if t.get("p1_pokemon_state")]
    p2_hp = [t["p2_pokemon_state"]["hp_pct"] for t in timeline if t.get("p2_pokemon_state")]

    p1_hp = np.array(p1_hp)
    p2_hp = np.array(p2_hp)
    hp_delta = p1_hp - p2_hp

    # Feature base sugli HP
    hp_diff_mean = float(np.mean(hp_delta))
    p1_hp_advantage_mean = float(np.mean(p1_hp > p2_hp))

    # Pokémon finali (ultimo HP noto)
    p1_hp_final = {}
    p2_hp_final = {}

    for t in timeline:
        if t.get("p1_pokemon_state"):
            p1_hp_final[t["p1_pokemon_state"]["name"]] = t["p1_pokemon_state"]["hp_pct"]
        if t.get("p2_pokemon_state"):
            p2_hp_final[t["p2_pokemon_state"]["name"]] = t["p2_pokemon_state"]["hp_pct"]

    p1_n_pokemon_use = len(p1_hp_final)
    p2_n_pokemon_use = len(p2_hp_final)
    diff_final_schieramento = p1_n_pokemon_use - p2_n_pokemon_use

    nr_pokemon_sconfitti_p1 = sum(v == 0 for v in p1_hp_final.values())
    nr_pokemon_sconfitti_p2 = sum(v == 0 for v in p2_hp_final.values())
    nr_pokemon_sconfitti_diff = nr_pokemon_sconfitti_p1 - nr_pokemon_sconfitti_p2

    # Final HP percent
    # (normalizzato anche per Pokémon non entrati)
    p1_pct_final_hp = sum(p1_hp_final.values()) + (team_size - len(p1_hp_final))
    p2_pct_final_hp = sum(p2_hp_final.values()) + (team_size - len(p2_hp_final))
    diff_final_hp = p1_pct_final_hp - p2_pct_final_hp

    # Durata della battaglia
    try:
        duration = sum(
            t["p1_pokemon_state"]["hp_pct"] > 0 and 
            t["p2_pokemon_state"]["hp_pct"] > 0
            for t in timeline
        )
    except:
        duration = len(timeline)

    hp_loss_rate = diff_final_hp / duration if duration > 0 else 0.0

    # Early / Late game HP differences
    phases = 3
    slice_idx = len(p1_hp) // phases

    early_hp_mean_diff = float(np.mean(hp_delta[:slice_idx])) if slice_idx > 0 else 0.0
    late_hp_mean_diff  = float(np.mean(hp_delta[-slice_idx:])) if slice_idx > 0 else 0.0

    # Trend HP (regressione)
    if len(hp_delta) > 1:
        slope, _, _, _, _ = linregress(np.arange(len(hp_delta)), hp_delta)
        hp_delta_trend = float(slope)
    else:
        hp_delta_trend = 0.0

    # Instabilità HP
    p1_hp_std = float(np.std(p1_hp))
    p2_hp_std = float(np.std(p2_hp))
    hp_delta_std = float(np.std(hp_delta))

    return {
        "hp_diff_mean": hp_diff_mean,
        "p1_hp_advantage_mean": p1_hp_advantage_mean,

        "p1_n_pokemon_use": p1_n_pokemon_use,
        "p2_n_pokemon_use": p2_n_pokemon_use,
        "diff_final_schieramento": diff_final_schieramento,

        "nr_pokemon_sconfitti_p1": nr_pokemon_sconfitti_p1,
        "nr_pokemon_sconfitti_p2": nr_pokemon_sconfitti_p2,
        "nr_pokemon_sconfitti_diff": nr_pokemon_sconfitti_diff,

        "p1_pct_final_hp": p1_pct_final_hp,
        "p2_pct_final_hp": p2_pct_final_hp,
        "diff_final_hp": diff_final_hp,

        "battle_duration": duration,
        "hp_loss_rate": hp_loss_rate,

        "early_hp_mean_diff": early_hp_mean_diff,
        "late_hp_mean_diff":  late_hp_mean_diff,

        "hp_delta_trend": hp_delta_trend,
        "p1_hp_std": p1_hp_std,
        "p2_hp_std": p2_hp_std,
        "hp_delta_std": hp_delta_std,
        "momentum_shift": momentum_shifts,
        "p1_momentum_phases": p1_advantage_phases,
        "hp_volatility": np.std(np.diff(hp_deltas)) if len(hp_deltas) > 1 else 0
    }
def get_full_move_features(timeline):
    p1_move_power_weighted = []
    p2_move_power_weighted = []

    p1_number_attacks = p2_number_attacks = 0
    p1_number_status = p2_number_status = 0

    p1_sum_negative_priority = 0
    p2_sum_negative_priority = 0

    # --- PRIORITY TRACKING ---
    p1_priorities = []
    p2_priorities = []

    for turn in timeline:
        # PLAYER 1
        move = turn.get("p1_move_details")
        if isinstance(move, dict):
            acc = move.get("accuracy", 1.0)
            base = move.get("base_power", 0)
            prio = move.get("priority", 0)

            weighted_power = base if acc == 0 else base * acc
            p1_move_power_weighted.append(weighted_power)

            category = move.get("category", "STATUS").upper()
            if category in ["PHYSICAL", "SPECIAL"]:
                p1_number_attacks += 1
            else:
                p1_number_status += 1

            if prio == -1:
                p1_sum_negative_priority += 1

            if prio is not None:
                p1_priorities.append(prio)

        # PLAYER 2
        move = turn.get("p2_move_details")
        if isinstance(move, dict):
            acc = move.get("accuracy", 1.0)
            base = move.get("base_power", 0)
            prio = move.get("priority", 0)

            weighted_power = base if acc == 0 else base * acc
            p2_move_power_weighted.append(weighted_power)

            category = move.get("category", "STATUS").upper()
            if category in ["PHYSICAL", "SPECIAL"]:
                p2_number_attacks += 1
            else:
                p2_number_status += 1

            if prio == -1:
                p2_sum_negative_priority += 1

            if prio is not None:
                p2_priorities.append(prio)

    # PRIORITY ADVANTAGE METRICS (UNIFIED)
    if p1_priorities and p2_priorities:
        avg_p1 = np.mean(p1_priorities)
        avg_p2 = np.mean(p2_priorities)

        priority_diff = avg_p1 - avg_p2

        # fraction of turns where P1 had higher priority
        min_len = min(len(p1_priorities), len(p2_priorities))
        higher = sum(p1_priorities[i] > p2_priorities[i] for i in range(min_len))
        priority_rate_advantage = higher / max(1, min_len)
    else:
        priority_diff = 0.0
        priority_rate_advantage = 0.0

    return {
        "p1_move_power_weighted": np.sum(p1_move_power_weighted),
        "p1_number_attacks": p1_number_attacks,
        "p1_number_status": p1_number_status,

        "p2_move_power_weighted": np.sum(p2_move_power_weighted),
        "p2_number_attacks": p2_number_attacks,
        "p2_number_status": p2_number_status,

        "diff_number_attack": p1_number_attacks - p2_number_attacks,
        "diff_number_status": p1_number_status - p2_number_status,

        "p1_sum_negative_priority": p1_sum_negative_priority,
        "p2_sum_negative_priority": p2_sum_negative_priority,
        "diff_negative_priority": p1_sum_negative_priority - p2_sum_negative_priority,

        # Integrated priority metrics
        "priority_diff": priority_diff,
        "priority_rate_advantage": priority_rate_advantage,
    }
def compute_switch_pressure(timeline):
    if not timeline:
        return {
            "p1_switch_count": 0,
            "p2_switch_count": 0,
            "diff_switch_count": 0,
            "p1_avg_active_duration": 0.0,
            "p2_avg_active_duration": 0.0,
            "diff_avg_active_duration": 0.0,
        }

    def get_stats_for_player(prefix: str):
        state_key = f"{prefix}_pokemon_state"
        last_name = None
        current_run = 0
        runs = []
        switches = 0

        for entry in timeline:
            state = entry.get(state_key, {}) or {}
            name = state.get("name")
            if not name:
                continue
            name = name.lower()

            if name == last_name:
                current_run += 1
                continue

            if last_name is not None:
                runs.append(current_run)
                switches += 1

            last_name = name
            current_run = 1

        if current_run:
            runs.append(current_run)

        avg_duration = float(np.mean(runs)) if runs else 0.0
        return switches, avg_duration

    p1_switches, p1_avg_duration = get_stats_for_player("p1")
    p2_switches, p2_avg_duration = get_stats_for_player("p2")

    return {
        "p1_avg_active_duration": p1_avg_duration,
        "p2_avg_active_duration": p2_avg_duration,
        "diff_avg_active_duration": p1_avg_duration - p2_avg_duration,
    }
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
def build_pokedex(data):
    """Aggregate base stats for every Pokemon observed in the dataset."""
    pokedex = {}

    def register_pokemon(pokemon):
        if not pokemon:
            return
        name = (pokemon.get("name") or "").lower()
        if not name:
            return
        pokedex[name] = {stat: pokemon.get(stat, 0) for stat in STAT_FIELDS}

    for battle in data:
        for member in battle.get("p1_team_details", []):
            register_pokemon(member)
        p2_lead = battle.get("p2_lead_details")
        register_pokemon(p2_lead)
    return pokedex
def compute_static_stats_features(battle):
    p1_mean_hp = p1_mean_spe = p1_mean_atk = p1_mean_def = p1_mean_spd = p1_mean_spa = 0.0
    p1_lead_hp = p1_lead_spe = p1_lead_atk = p1_lead_def = p1_lead_spd = p1_lead_spa = 0.0
    #feature statiche
    features = {}
    p1_team = battle.get("p1_team_details", [])
    if p1_team:
        stats = {
            "hp":  [p.get("base_hp", 0)  for p in p1_team],
            "spe": [p.get("base_spe", 0) for p in p1_team],
            "atk": [p.get("base_atk", 0) for p in p1_team],
            "def": [p.get("base_def", 0) for p in p1_team],
            "spd": [p.get("base_spd", 0) for p in p1_team],
            "spa": [p.get("base_spa", 0) for p in p1_team],
        }

        # max offense and speed
        features["p1_max_offensive_stat"] = max(
            max(a, s) for a, s in zip(stats["atk"], stats["spa"])
        )
        features["p1_max_speed_stat"] = max(stats["spe"])

        # means
        features["p1_mean_hp"]  = np.mean(stats["hp"])
        features["p1_mean_spe"] = np.mean(stats["spe"])
        features["p1_mean_atk"] = np.mean(stats["atk"])
        features["p1_mean_def"] = np.mean(stats["def"])
        features["p1_mean_sp"]  = np.mean(stats["spd"])

        # Lead stats (primo Pokémon)
        lead = p1_team[0]
        p1_lead_hp  = lead.get("base_hp", 0)
        p1_lead_spe = lead.get("base_spe", 0)
        p1_lead_atk = lead.get("base_atk", 0)
        p1_lead_def = lead.get("base_def", 0)
        p1_lead_spd = lead.get("base_spd", 0)


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
    return features
def compute_boost_features(timeline, base_stats_p1=None, base_stats_p2=None):
    if not timeline:
        return {
            # dynamic boost features
            "p1_max_offense_boost_diff": 0.0,
            # cumulative boost features
            "boost_p1": 0,
            "boost_p2": 0,
            # extract_boost_features summary
            "diff_boost_last_turn": 0,
            "diff_boost_atk_last_turn": 0,
            "diff_boost_def_last_turn": 0,
            "diff_boost_spa_last_turn": 0,
            "diff_boost_spd_last_turn": 0,
            "diff_boost_spe_last_turn": 0,
            "diff_boost_count_turni": 0,
            "diff_turn_first_boost": 0,
            "diff_effective_offense": 0,
            "diff_effective_defense": 0,
            "p1_is_faster_effective": 0,
        }

    # ----------------------------------------------------------
    # Init
    # ----------------------------------------------------------
    offense_boost_diff_list = []
    sum_boost_p1 = 0
    sum_boost_p2 = 0

    p1_boost_count = 0
    p2_boost_count = 0
    p1_first_turn = None
    p2_first_turn = None

    last_b1 = None
    last_b2 = None

    # ----------------------------------------------------------
    # SINGLE SCAN OF TIMELINE
    # ----------------------------------------------------------
    for entry in timeline:
        turn = entry.get("turn", None)
        b1 = entry.get("p1_pokemon_state", {}).get("boosts", {})
        b2 = entry.get("p2_pokemon_state", {}).get("boosts", {})

        # save last
        last_b1 = b1
        last_b2 = b2

        # cumulative
        sum_boost_p1 += sum(b1.values())
        sum_boost_p2 += sum(b2.values())

        # dynamic offense/speed diff
        p1_off = b1.get("atk", 0) + b1.get("spa", 0)
        p2_off = b2.get("atk", 0) + b2.get("spa", 0)
        offense_boost_diff_list.append(p1_off - p2_off)

        # counts
        if any(v != 0 for v in b1.values()):
            p1_boost_count += 1
            if p1_first_turn is None:
                p1_first_turn = turn

        if any(v != 0 for v in b2.values()):
            p2_boost_count += 1
            if p2_first_turn is None:
                p2_first_turn = turn

    # default last boosts if missing
    if last_b1 is None:
        last_b1 = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}
    if last_b2 is None:
        last_b2 = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}

    # ----------------------------------------------------------
    # LAST TURN DIFFS
    # ----------------------------------------------------------
    diff_atk = last_b1["atk"] - last_b2["atk"]
    diff_def = last_b1["def"] - last_b2["def"]
    diff_spa = last_b1["spa"] - last_b2["spa"]
    diff_spd = last_b1["spd"] - last_b2["spd"]
    diff_spe = last_b1["spe"] - last_b2["spe"]

    diff_total = sum(last_b1.values()) - sum(last_b2.values())

    # ----------------------------------------------------------
    # EFFECTIVE STATS (only if base stats provided)
    # ----------------------------------------------------------
    diff_eff_off = 0
    diff_eff_def = 0
    p1_is_faster_effective = 0

    if base_stats_p1 and base_stats_p2:
        eff = lambda base, stage: base * BOOST_MULT.get(stage, 1.0)

        eff1 = {
            s: eff(base_stats_p1[f"base_{s}"], last_b1[s]) 
            for s in ["atk","def","spa","spd","spe"]
        }
        eff2 = {
            s: eff(base_stats_p2[f"base_{s}"], last_b2[s]) 
            for s in ["atk","def","spa","spd","spe"]
        }

        diff_eff_off = (eff1["atk"] + eff1["spa"]) - (eff2["atk"] + eff2["spa"])
        diff_eff_def = (eff1["def"] + eff1["spd"]) - (eff2["def"] + eff2["spd"])
        p1_is_faster_effective = int(eff1["spe"] > eff2["spe"])

    return {
        # dynamic (max difference)
        "p1_max_offense_boost_diff": max(offense_boost_diff_list),

        # cumulative boosts
        "boost_p1": sum_boost_p1,
        "boost_p2": sum_boost_p2,

        # last-turn raw differences
        "diff_boost_last_turn": diff_total,
        "diff_boost_atk_last_turn": diff_atk,
        "diff_boost_def_last_turn": diff_def,
        "diff_boost_spa_last_turn": diff_spa,
        "diff_boost_spd_last_turn": diff_spd,
        "diff_boost_spe_last_turn": diff_spe,

        # temporal info
        "diff_boost_count_turni": p1_boost_count - p2_boost_count,
        "diff_turn_first_boost": (p1_first_turn or 31) - (p2_first_turn or 31),

        # effective stats
        "diff_effective_offense": diff_eff_off,
        "diff_effective_defense": diff_eff_def,
        "p1_is_faster_effective": p1_is_faster_effective,
    }
def compute_effect_features(timeline):
    freq = {
        "p1": {eff: 0 for eff in important_effects},
        "p2": {eff: 0 for eff in important_effects},
    }
    first_turn = {
        "p1": {eff: None for eff in important_effects},
        "p2": {eff: None for eff in important_effects},
    }

    # ---- SINGLE PASS ----
    for entry in timeline:
        turn = entry.get("turn", None)

        for prefix, state_key in (("p1", "p1_pokemon_state"), ("p2", "p2_pokemon_state")):
            state = entry.get(state_key, {})
            effects = state.get("effects", [])

            for eff in important_effects:
                if eff in effects:
                    freq[prefix][eff] += 1
                    if first_turn[prefix][eff] is None:
                        first_turn[prefix][eff] = turn

    # replace None with 31
    for prefix in ("p1", "p2"):
        for eff in important_effects:
            if first_turn[prefix][eff] is None:
                first_turn[prefix][eff] = 31

    # ---- flatten result in feature dict ----
    out = {}
    for prefix in ("p1", "p2"):
        for eff in important_effects:
            out[f"{prefix}_{eff}_freq"] = freq[prefix][eff]
            out[f"{prefix}_{eff}_first_turn"] = first_turn[prefix][eff]

    return out
def calculate_expected_damage_ratio_turn_1(battle, type_chart):
    try:
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

        # defender types
        p1_defender_types = [t for t in p1_lead_stats.get("types", []) if t != "notype"]
        p2_defender_types = [t for t in p2_lead_stats.get("types", []) if t != "notype"]

        p1_expected_damage = 0.0
        p2_expected_damage = 0.0

        # -----------------------------
        # P1 damage on P2
        # -----------------------------
        if p1_move and p1_move.get("category") in ["SPECIAL", "PHYSICAL"]:
            base_power = p1_move.get("base_power", 0)
            move_type = p1_move.get("type", "").upper()
            cat = p1_move.get("category", "").upper()

            if cat == "SPECIAL":
                att = p1_lead_stats.get("base_spa", 1)
                dfn = p2_lead_stats.get("base_spd", 1)
            else:
                att = p1_lead_stats.get("base_atk", 1)
                dfn = p2_lead_stats.get("base_def", 1)

            m = get_type_multiplier(move_type, p2_defender_types, type_chart)
            p1_expected_damage = base_power * (att / dfn) * m

        # -----------------------------
        # P2 damage on P1
        # -----------------------------
        if p2_move and p2_move.get("category") in ["SPECIAL", "PHYSICAL"]:
            base_power = p2_move.get("base_power", 0)
            move_type = p2_move.get("type", "").upper()
            cat = p2_move.get("category", "").upper()

            if cat == "SPECIAL":
                att = p2_lead_stats.get("base_spa", 1)
                dfn = p1_lead_stats.get("base_spd", 1)
            else:
                att = p2_lead_stats.get("base_atk", 1)
                dfn = p1_lead_stats.get("base_def", 1)

            m = get_type_multiplier(move_type, p1_defender_types, type_chart)
            p2_expected_damage = base_power * (att / dfn) * m

        # -----------------------------
        # Log-smoothed advantage
        # -----------------------------
        p1_smooth = p1_expected_damage + 1.0
        p2_smooth = p2_expected_damage + 1.0

        return float(np.log(p1_smooth) - np.log(p2_smooth))

    except Exception:
        # Qualsiasi errore → fallback sicuro
        return 0.0
def extract_dynamic_stat_diffs(timeline, p1_team, pokedex):
    MEDIUM_SPEED_THRESHOLD = 90   # medium-speed Pokémon
    HIGH_SPEED_THRESHOLD = 100    # high-speed Pokémon

    # TEAM SPEED-BASED FEATURES (NUOVE)
    speeds = np.array([p.get("base_spe", 0) for p in p1_team])

    p1_avg_speed_stat_battaglia = float(np.mean(speeds > MEDIUM_SPEED_THRESHOLD))
    p1_avg_high_speed_stat_battaglia = float(np.mean(speeds > HIGH_SPEED_THRESHOLD))

    # DYNAMIC STAT DIFFERENCES P1 ACTIVE VS P2
    stat_keys = ["base_atk", "base_spa", "base_spe"]
    stat_diffs = {k: [] for k in stat_keys}

    for t in timeline:
        p1_state = t.get("p1_pokemon_state", {})
        p2_state = t.get("p2_pokemon_state", {})

        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()

        p1_stats = pokedex.get(p1_name) if p1_name else None
        p2_stats = pokedex.get(p2_name) if p2_name else None

        if not p1_stats or not p2_stats:
            continue

        # Accumula differenze dinamiche
        for stat in stat_keys:
            p1_value = p1_stats.get(stat, 0)
            p2_value = p2_stats.get(stat, 0)
            stat_diffs[stat].append(p1_value - p2_value)

    # Aggregazioni finali
    results = {}

    for stat, values in stat_diffs.items():
        if values:
            results[f"mean_{stat}_diff_timeline"] = float(np.mean(values))
            results[f"std_{stat}_diff_timeline"]  = float(np.std(values))
        else:
            results[f"mean_{stat}_diff_timeline"] = 0.0
            results[f"std_{stat}_diff_timeline"]  = 0.0

    # SPEED FEATURES
    results["p1_avg_speed_stat_battaglia"] = p1_avg_speed_stat_battaglia
    results["p1_avg_high_speed_stat_battaglia"] = p1_avg_high_speed_stat_battaglia

    return results
def extract_type_advantage_features(battle, timeline, p1_team, pokemon_dict, type_chart):
    features = {}
    all_types = list(type_chart.keys())

    # TEAM TYPE DIVERSITY (P1)
    p1_types = [t for p in p1_team for t in p.get("types", []) if t != "notype"]
    features["p1_type_diversity"] = len(set(p1_types))

    # TEAM RESISTANCE + WEAKNESS (P1)
    def team_weakness(team):
        weakness_counts = []
        for p in team:
            types = [t for t in p.get("types", []) if t != "notype"]
            if not types:
                continue
            weak_to = 0
            for atk in all_types:
                mult = 1.0
                for d in types:
                    mult *= type_chart.get(atk.upper(), {}).get(d.upper(), 1.0)
                if mult > 1.0:
                    weak_to += 1
            weakness_counts.append(weak_to)
        if not weakness_counts:
            return 0.0
        mean_weak = np.mean(weakness_counts)
        return mean_weak / len(all_types)

    def team_resistance(team):
        weakness_counts = []
        for p in team:
            types = [t for t in p.get("types", []) if t != "notype"]
            if not types:
                continue
            weak_to = 0
            for atk in all_types:
                mult = 1.0
                for d in types:
                    mult *= type_chart.get(atk.upper(), {}).get(d.upper(), 1.0)
                if mult > 1.0:
                    weak_to += 1
            weakness_counts.append(weak_to)
        if not weakness_counts:
            return 1.0
        mean_weak = np.mean(weakness_counts)
        return 1.0 / mean_weak if mean_weak > 0 else 1.0

    features["p1_type_resistance"] = team_resistance(p1_team)
    features["p1_type_weakness"] = team_weakness(p1_team)

    # COVERAGE: P1 ha tipi super-effective vs P2 lead?
    p2_lead = battle.get("p2_lead_details", {})
    if p2_lead:
        p2_types = [t for t in p2_lead.get("types", []) if t != "notype"]
        count_supereffective = 0

        for p1 in p1_team:
            p1_types = [t for t in p1.get("types", []) if t != "notype"]
            found = False
            for atk in p1_types:
                if get_type_multiplier(atk, p2_types, type_chart) >= 2.0:
                    found = True
                    break
            if found:
                count_supereffective += 1

        features["p1_team_super_effective_moves"] = float(count_supereffective)
    else:
        features["p1_team_super_effective_moves"] = 0.0

    # TIMELINE TYPE ADVANTAGE (P1 vs P2)
    def avg_advantage_over_timeline():
        p1_adv_list = []
        p2_adv_list = []

        for turn in timeline:
            p1_name = turn.get("p1_pokemon_state", {}).get("name", "")
            p2_name = turn.get("p2_pokemon_state", {}).get("name", "")
            if not p1_name or not p2_name:
                continue

            p1_types = pokemon_dict.get(p1_name.lower(), [])
            p2_types = pokemon_dict.get(p2_name.lower(), [])
            if not p1_types or not p2_types:
                continue

            # P1 attacking P2
            p1_results = []
            for atk in all_types:
                mult = 1.0
                for d in p2_types:
                    mult *= type_chart.get(atk.upper(), {}).get(d.upper(), 1.0)
                p1_results.append(mult)

            # P2 attacking P1
            p2_results = []
            for atk in all_types:
                mult = 1.0
                for d in p1_types:
                    mult *= type_chart.get(atk.upper(), {}).get(d.upper(), 1.0)
                p2_results.append(mult)

            if p1_results:
                p1_adv_list.append(np.mean(p1_results))
            if p2_results:
                p2_adv_list.append(np.mean(p2_results))

        if not p1_adv_list or not p2_adv_list:
            return (1.0, 1.0, 0.0)

        p1_avg = np.mean(p1_adv_list)
        p2_avg = np.mean(p2_adv_list)
        return (p1_avg, p2_avg, p1_avg - p2_avg)

    p1_adv, p2_adv, diff_adv = avg_advantage_over_timeline()

    features["p1_type_advantage"] = p1_adv
    features["p2_type_advantage"] = p2_adv
    features["diff_type_advantage"] = diff_adv

    return features

def create_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    pokemon_dict = create_pokemon_dict(data)
    pokedex = build_pokedex(data)
    #definiamo le features
    for battle in tqdm(data, desc="Extracting features"):
        battle_id = battle.get("battle_id")
        features = {}
        #STATISTICHE
        features.update(compute_static_stats_features(battle))
        p1_team = battle["p1_team_details"]
        p1_lead = p1_team[0]
        p2_lead = battle["p2_lead_details"]
        timeline = battle.get("battle_timeline", [])
        if timeline:
            #HP
            features.update(extract_full_hp_features(timeline, team_size=len(p1_team)))
            #BOOST
            features.update(compute_boost_features(timeline, p1_lead, p2_lead))
            #TYPE
            features.update(
                extract_type_advantage_features(battle, timeline, p1_team, pokemon_dict, type_chart)
            )
            #MOVES and priority
            features.update(get_full_move_features(timeline))
            #SWITCH PRESSURE
            features.update(compute_switch_pressure(timeline))
            #STATUS
            features.update(compute_status_features(timeline))
            #EFFECTS
            features.update(compute_effect_features(timeline))
            #EXPECTED DAMAGE TURN1
            features["expected_damage_ratio_turn_1"] = calculate_expected_damage_ratio_turn_1(battle, type_chart)
            #DYNAMIC STATS
            features.update(extract_dynamic_stat_diffs(timeline, p1_team, pokedex))
            #STABS
            features.update(compute_mean_stab_moves(timeline, pokemon_dict))
            #interaction
            features.update(calculate_interaction_features(features))
        features["battle_id"] = battle_id
        if "player_won" in battle:
            features["player_won"] = int(battle["player_won"])
        #features = dict(sorted(features.items()))
        #features = shuffle_dict(features, seed=1234)
        feature_list.append(features)
    return pd.DataFrame(feature_list).fillna(0)
