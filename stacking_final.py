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

import pandas as pd
import json
from collections import Counter
import numpy as np
import pandas as pd
from collections import Counter
import numpy as np

import pandas as pd
from typing import List, Dict, Any, Counter
from typing import Any, Dict, Iterable, List, Tuple
from start_utils import predict_and_submit, create_feature_instance

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

COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('input', COMPETITION_NAME)

train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')

train_data = read_train_data(train_file_path)
test_data = read_test_data(test_file_path)

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
            for def_type in p2_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p1_mult.append(mult)
        #turn summary
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0

        # --- P2 attacking P1 ---
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
    # if is_test and battle_id == 109:
    #     print(f"first team:{debug_dict_p1}, second team:{debug_dict_p2}")
    #     #,p1_avg,p2_avg,p1_avg - p2_avg
    #     exit()
    return {
        "p1_type_advantage": p1_avg,
        "p2_type_advantage": p2_avg,
        "diff_type_advantage": p1_avg - p2_avg
    }

def compute_avg_offensive_potential(timeline, pokemon_dict, type_chart, is_test=False, battle_id=''):
    # ... (initial checks are the same)
    if not timeline:
        return {
            "p1_type_advantage": 1.0,
            "p2_type_advantage": 1.0,
            "diff_type_advantage": 0.0
        }

    p1_advantages = []
    p2_advantages = []
    
    # Get all possible move types from the type chart keys
    all_move_types = list(type_chart.keys())

    for turn in timeline:
        # ... (get names and types, same as before)
        p1_name = turn.get("p1_pokemon_state", {}).get("name")
        p2_name = turn.get("p2_pokemon_state", {}).get("name")

        if not p1_name or not p2_name:
            continue

        p1_types = pokemon_dict.get(p1_name.lower(), [])
        p2_types = pokemon_dict.get(p2_name.lower(), [])

        if not p1_types or not p2_types:
            continue
        # --- P1 attacking P2: Calculate average effectiveness of ALL move types ---
        p1_mult = []
        for atk_type in all_move_types: # <-- **CHANGE**: Iterate over ALL possible move types
            mult = 1.0
            for def_type in p2_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p1_mult.append(mult)
        # turn summary
        p1_adv = np.mean(p1_mult) if p1_mult else 1.0

        # --- P2 attacking P1: Calculate average effectiveness of ALL move types ---
        p2_mult = []
        for atk_type in all_move_types: # <-- **CHANGE**: Iterate over ALL possible move types
            mult = 1.0
            for def_type in p1_types:
                mult *= type_chart.get(atk_type.upper(), {}).get(def_type.upper(), 1.0)
            p2_mult.append(mult)
        # turn summary
        p2_adv = np.mean(p2_mult) if p2_mult else 1.0

        p1_advantages.append(p1_adv)
        p2_advantages.append(p2_adv)

    # ... (final calculation and return are the same)
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
    """
    Conta il numero di volte in cui i Pokémon di P1 e P2 hanno subito uno stato
    diverso da 'nostatus' in ogni turno della battaglia.

    Args:
        dati_battaglia (dict): Il dizionario contenente la struttura dei dati della battaglia.

    Returns:
        dict: Un dizionario con i conteggi totali per p1, p2 e la loro differenza.
    """
    conteggio_p1 = 0
    conteggio_p2 = 0
    major_status_p1 = 0
    major_status_p2 = 0
    MAJOR_STATUSES = {'slp', 'frz'} 
    for turno in timeline:
        # Assumiamo che la chiave contenga lo stato del Pokémon nel formato
        # "p1_pokemon_state|p2_pokemon_state" e che lo stato sia la seconda parte dopo la virgola.
        # È fondamentale sapere esattamente come sono formattati i dati.

        # Ipotizzando che la chiave sia presente e il suo valore sia un dizionario:
        stato_dettagli_p1 = turno.get("p1_pokemon_state", {})
        # Estraiamo la stringa dello stato
        # Esempio: "name": "starmie", "hp_pct": 1.0, "status": "'nostatus'|'slp', 'frz', 'brn', ...
        status_string = stato_dettagli_p1.get("status", "")
        if status_string:
            # La stringa è formattata come "stato_p1|stato_p2". La dividiamo.
            if status_string.lower() != 'nostatus':
                conteggio_p1 += 1
                if status_string.lower() in MAJOR_STATUSES:#bloccanti
                    major_status_p1 += 1
                
        stato_dettagli_p2 = turno.get("p2_pokemon_state", {})
        status_string = stato_dettagli_p2.get("status", "")
        if status_string:
            # La stringa è formattata come "stato_p1|stato_p2". La dividiamo.
            if status_string.lower() != 'nostatus':
                conteggio_p2 += 1
                if status_string.lower() in MAJOR_STATUSES:#bloccanti
                    major_status_p2 += 1


    differenza = conteggio_p1 - conteggio_p2 # P1 meno P2
    
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
            # less informative: min/max statistics removed
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    arr = np.asarray(seq, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        # less informative: min/max statistics removed
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }
def calculate_battle_stats(battle):
    # Leader P1: p1_team_details[0]
    p1_team = battle["p1_team_details"]
    p1_lead = p1_team[0]
    p2_lead = battle["p2_lead_details"]
    p1_lead_spe = p1_lead["base_spe"]
    # Leader P2: p2_lead_details
    p2_lead_spe = p2_lead["base_spe"]

    diff_speed_first = p1_lead_spe - p2_lead_spe

    timeline_speeds_diffs = []
    p1_team_details = battle.get("p1_team_details", [])
    p2_lead_details = battle.get("p2_lead_details", {})
    p2_lead_name = p2_lead_details.get("name", "").lower().strip()
    p2_base_spe = p2_lead_details.get("base_spe", 0)
    
    # Itera attraverso la timeline
    for turn in battle.get("battle_timeline", []):
        p1_state_data = turn.get("p1_pokemon_state")
        p1_name = p1_state_data.get("name", "") if p1_state_data else ""
        p2_state_data = turn.get("p2_pokemon_state")
        
        p2_name = p2_state_data.get("name", "") if p2_state_data else ""

        p1_speed = get_p1_base_speed(p1_name, p1_team_details)
        p2_speed = 0
        if p2_name.lower().strip() == p2_lead_name:
            p2_speed = p2_base_spe
        # Se P2 non ha il lead in campo (se il JSON fosse completo), il dato non sarebbe disponibile con i vincoli dati.
        
        # Calcola e aggiungi la differenza (P1 - P2)
        if p1_speed != 0 or p2_speed != 0:
            timeline_speeds_diffs.append(p1_speed - p2_speed)

    # Calcola la media delle differenze di velocità
    if timeline_speeds_diffs:
        diff_speed_timeline = np.mean(timeline_speeds_diffs)
    else:
        # Se la timeline è vuota o non si trovano dati validi, usa la differenza del lead
        diff_speed_timeline = diff_speed_first


    # --- 3. diff_stat: Differenza media delle statistiche rilevanti ---

    # Statistiche base da considerare

    # Calcolo della media delle statistiche per il Team 1 (6 Pokémon)
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

    # Statistiche del Lead del Team 2
    p2_lead_stats = {key: battle["p2_lead_details"][key] for key in STAT_FIELDS}

    # Calcolo della differenza media delle statistiche
    # Differenza: Media(Team 1) - Statistiche(Lead Team 2)
    stat_diffs = []
    for key in STAT_FIELDS:
        diff = p1_avg_stats[key] - p2_lead_stats[key]
        stat_diffs.append(diff)

    # La feature è la media di queste 6 differenze di statistica
    diff_stat = np.mean(stat_diffs) if stat_diffs else 0
    res = {
        #"diff_speed_first": diff_speed_first,
        "diff_speed_timeline": diff_speed_timeline,
        "diff_stat_mean": diff_stat,
        
        #non ancora usate
        "sum_stat_lead_p1": sum_stat_lead_p1,
        "sum_stat_lead_p2": sum_stat_lead_p2,
        "diff_stat_lead": sum_stat_lead_p1 - sum_stat_lead_p2,
        #"avg_stat_p1": avg_stat_p1,
        #"avg_stat_diff": avg_stat_p1 - 
        
    }
    ###
    # p1_stats = [
    #     p1_lead["base_hp"],
    #     p1_lead["base_atk"],
    #     p1_lead["base_def"],
    #     p1_lead["base_spa"],
    #     p1_lead["base_spd"],
    #     p1_lead["base_spd"]
    # ]
    for stat in STAT_FIELDS:
        values = [member.get(stat, 0) or 0 for member in p1_team]
        res.update(compute_statistics(values, f"p1_team_{stat}"))
        
        p1_lead_stat = p1_lead.get(stat, 0)
        p2_lead_stat = p2_lead.get(stat, 0)
        res.update({f"p2_lead_{stat}":p2_lead_stat})
        
        res.update({f"diff_lead_{stat}":p1_lead_stat - p2_lead_stat})
    #avg_stat_p1 = np.mean(p1_stats)
    #avg_stat_p2 = np.mean(p2_stats)
    #features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
    # --- Ritorno dei risultati ---
    return res
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
def extract_hp_features(battle):
    """
    Calcola tre feature basate sui dati di battaglia:
    1. p1_hp_pct_sum: Somma del massimo hp_pct per i Pokémon distinti del Team 1.
    2. p2_hp_pct_sum: Somma del massimo hp_pct per i Pokémon distinti del Team 2.
    3. diff_hp_pct: Differenza (p1_hp_pct_sum - p2_hp_pct_sum).

    Args:
        battle_data (dict): Dizionario contenente i dettagli della battaglia.

    Returns:
        dict: Un dizionario con le tre feature calcolate.
    """

    # Dizionari per tenere traccia del massimo hp_pct raggiunto da ciascun Pokémon distinto
    # La chiave è il nome del Pokémon, il valore è il massimo hp_pct visto.
    p1_set_hp_pct = {}
    p2_set_hp_pct = {}

    timeline = battle.get("battle_timeline", [])
    # 2. Scorre la timeline della battaglia
    for turn in timeline:#
        # Estrai lo stato del Pokémon 1
        p1_pokemon_state = turn.get("p1_pokemon_state", None)
        if p1_pokemon_state:
            p1_name = p1_pokemon_state.get("name")
            p1_hp_pct = p1_pokemon_state.get("hp_pct")
            p1_set_hp_pct[p1_name] = p1_hp_pct

        # Estrai lo stato del Pokémon 2
        p2_pokemon_state = turn.get("p2_pokemon_state", None)
        if p2_pokemon_state:
            p2_name = p2_pokemon_state.get("name")
            p2_hp_pct = p2_pokemon_state.get("hp_pct")
            p2_set_hp_pct[p2_name] = p2_hp_pct

    team_member_count =  len(battle.get('p1_team_details'))-len(p1_set_hp_pct.keys())
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
def create_move_features(timeline):
    
    # Mappa per la codifica numerica della categoria
    category_map = {
        "PHYSICAL": 1,
        "SPECIAL": 2,
        "STATUS": 0
    }
    
    # Lista per contenere i dati della timeline con le nuove feature
    extended_timeline = []
    p1_move_power_weighted = []
    p1_number_attacks = 0
    p1_number_status = 0
    
    p1_sum_negative_priority = 0
    p2_sum_negative_priority = 0
    
    
    p2_move_power_weighted = []
    p2_number_attacks = 0
    p2_number_status = 0
    for turn in timeline:
        # Assumiamo che la mossa sia sotto 'p1_move_details|p2_move_details'
        move_details_key = "p1_move_details"#|p2_move_details
        if turn.get(move_details_key) != None:
            move = turn[move_details_key]
            
            # 1. Feature: move_power_weighted
            # Un 'danno atteso' che combina potenza e accuratezza
            accuracy = move.get("accuracy", 1.0) # Default a 1.0 se mancante
            base_power = move.get("base_power", 0)
            priority = move.get("priority", 0)
            # Se la precisione è 0, assumiamo che sia una mossa a 100% di precisione 
            # se non è specificato (come "noaccuracy"), altrimenti usiamo il valore fornito.
            if accuracy == 0:
                 weighted_power = base_power
            else:
                weighted_power = base_power * accuracy
            
            p1_move_power_weighted.append(round(weighted_power, 3))

            # 2. Feature: is_physical_or_special
            # Codifica della categoria di attacco (1 per attacco, 0 per status)
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
            
            # 1. Feature: move_power_weighted
            # Un 'danno atteso' che combina potenza e accuratezza
            accuracy = move.get("accuracy", 1.0) # Default a 1.0 se mancante
            base_power = move.get("base_power", 0)
            priority = move.get("priority", 0)
            # Se la precisione è 0, assumiamo che sia una mossa a 100% di precisione 
            # se non è specificato (come "noaccuracy"), altrimenti usiamo il valore fornito.
            if accuracy == 0:
                 weighted_power = base_power
            else:
                weighted_power = base_power * accuracy
            
            p2_move_power_weighted.append(round(weighted_power, 3))

            # 2. Feature: is_physical_or_special
            # Codifica della categoria di attacco (1 per attacco, 0 per status)
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

    # ---- Tracking temporale ----
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

        # Salva ultimo stato
        last_boost_p1 = p1_boosts
        last_boost_p2 = p2_boosts

        # Count boost
        if any(v != 0 for v in p1_boosts.values()):
            p1_boost_count += 1
            if p1_first_boost_turn is None:
                p1_first_boost_turn = turn

        if any(v != 0 for v in p2_boosts.values()):
            p2_boost_count += 1
            if p2_first_boost_turn is None:
                p2_first_boost_turn = turn

    # se nessun boost mai visto
    if last_boost_p1 is None:
        last_boost_p1 = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}
    if last_boost_p2 is None:
        last_boost_p2 = {"atk":0,"def":0,"spa":0,"spd":0,"spe":0}

    # ---- Differenze dell’ultimo turno ----
    diff_boost_atk = last_boost_p1["atk"] - last_boost_p2["atk"]
    diff_boost_def = last_boost_p1["def"] - last_boost_p2["def"]
    diff_boost_spa = last_boost_p1["spa"] - last_boost_p2["spa"]
    diff_boost_spd = last_boost_p1["spd"] - last_boost_p2["spd"]
    diff_boost_spe = last_boost_p1["spe"] - last_boost_p2["spe"]

    boost_p1_total = sum(last_boost_p1.values())
    boost_p2_total = sum(last_boost_p2.values())
    diff_boost_total = boost_p1_total - boost_p2_total

    # ---- Statistiche temporali ----
    diff_boost_count = p1_boost_count - p2_boost_count
    diff_first_boost_turn = (p1_first_boost_turn or 31) - (p2_first_boost_turn or 31)

    # ---- Effective stat = base_stat × boost_multiplier ----

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

    # differenze reali
    diff_effective = {
        f"diff_effective_{stat}": effective_p1[stat] - effective_p2[stat]
        for stat in effective_p1
    }

    # power offensivo/difensivo
    p1_eff_off = effective_p1["atk"] + effective_p1["spa"]
    p1_eff_def = effective_p1["def"] + effective_p1["spd"]

    p2_eff_off = effective_p2["atk"] + effective_p2["spa"]
    p2_eff_def = effective_p2["def"] + effective_p2["spd"]

    diff_eff_off = p1_eff_off - p2_eff_off
    diff_eff_def = p1_eff_def - p2_eff_def

    # speed advantage reale (boolean)
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

        # **diff_effective,
        "diff_effective_offense": diff_eff_off,
        "diff_effective_defense": diff_eff_def,
        "p1_is_faster_effective": p1_is_faster,
    }
    return res


def calcola_feature_boost(timeline):
    totale_boost_p1 = 0
    totale_boost_p2 = 0
    for turno in timeline:
        # Funzione helper per calcolare la somma dei boost di un singolo Pokémon
        def somma_boosts(pokemon_state):
            somma = 0
            if pokemon_state and "boosts" in pokemon_state:
                # Somma tutti i valori di boost (atk, def, spa, spd, spe)
                somma = sum(pokemon_state["boosts"].values())
            return somma

        # Prova ad accedere allo stato del P1 (presumendo che sia una chiave nel JSON)
        if "p1_pokemon_state" in turno:
            boost_corrente_p1 = somma_boosts(turno["p1_pokemon_state"])
            totale_boost_p1 += boost_corrente_p1

        # Prova ad accedere allo stato del P2 (presumendo che sia una chiave nel JSON)
        if "p2_pokemon_state" in turno:
            boost_corrente_p2 = somma_boosts(turno["p2_pokemon_state"])
            totale_boost_p2 += boost_corrente_p2

    # Calcola la differenza
    #diff_boost = totale_boost_p1 - totale_boost_p2

    # Restituisce le feature
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
    """
    timeline: lista di 30 elementi, ciascuno con "p1_pokemon_state" o "p2_pokemon_state"
    prefix: 'p1' oppure 'p2'
    """
    # inizializza strutture
    freq = {eff: 0 for eff in important_effects}
    first_turn = {eff: None for eff in important_effects}

    for entry in timeline:
        turn = entry.get("turn", None)

        # prendi lo stato del pokemon corretto (p1 o p2)
        state_key = "p1_pokemon_state" if prefix == "p1" else "p2_pokemon_state"
        state = entry.get(state_key, {})

        effects = state.get("effects", [])
        # aggiorna freq e first_turn
        for eff in important_effects:
            if eff in effects:
                freq[eff] += 1
                if first_turn[eff] is None:
                    first_turn[eff] = turn

    # converte None → 31 per indicare "mai apparso"
    for eff in important_effects:
        if first_turn[eff] is None:
            first_turn[eff] = 31

    # impacchetta le feature
    features = {}

    for eff in important_effects:
        features[f"{prefix}_{eff}_freq"] = freq[eff]
        features[f"{prefix}_{eff}_first_turn"] = first_turn[eff]
    return features



# Output: {'p1_hp_pct_sum': 2.0, 'p2_hp_pct_sum': 1.9, 'diff_hp_pct': 0.1}
def create_features(data: list[dict], is_test=False) -> pd.DataFrame:
    feature_list = []
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
    #definiamo le features
    for battle in tqdm(data, desc="Extracting features"):
        battle_id = battle.get("battle_id")
        features = {}
        features.update(create_feature_instance(battle, pokemon_dict, []))
        features['battle_id'] = battle_id
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        battle_stats_results = calculate_battle_stats(battle)
        features.update(battle_stats_results)
        # features['diff_speed_first'] = battle_stats_results['diff_speed_first']
        # features['diff_speed_timeline'] = battle_stats_results['diff_speed_timeline']
        # features['diff_stat'] = battle_stats_results['diff_stat']
        #features['avg_stat_p1'] = battle_stats_results['avg_stat_p1']
        """
        """
        #non ancora usate sum e diff stat lead
        features['sum_stat_lead_p1'] = battle_stats_results['sum_stat_lead_p1']
        features['sum_stat_lead_p2'] = battle_stats_results['sum_stat_lead_p2']
        features['diff_stat_lead'] = battle_stats_results['diff_stat_lead']
        """
        """
        hp_result = extract_hp_features(battle)
        features['p1_hp_pct_sum'] = hp_result['p1_hp_pct_sum']
        features['p2_hp_pct_sum'] = hp_result['p2_hp_pct_sum']
        features['diff_hp_pct'] = hp_result['diff_hp_pct']
        features['hp_advantage_trend'] = hp_advantage_trend(battle)
        
        timeline = battle.get('battle_timeline', [])
        if timeline:
            #result = compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart, is_test, battle_id)
            off_potential_result = compute_avg_offensive_potential(timeline, pokemon_dict, type_chart, is_test, battle_id)
            #CHECK 84.40% (+/- 1.12%) =>  84.35% (+/- 1.01%)
            features['p1_type_advantage'] = off_potential_result['p1_type_advantage']
            features['p2_type_advantage'] = off_potential_result['p2_type_advantage']
            features['diff_type_advantage'] = off_potential_result['diff_type_advantage']
            
            status_anomali_result = conta_status_anomali(timeline)
            features['status_p1'] = status_anomali_result['status_p1']
            features['status_p2'] = status_anomali_result['status_p2']
            features['diff_status'] = status_anomali_result['diff_status']
            #voting: aggiunti major status slp frz
            features['major_status_p1'] = status_anomali_result['major_status_p1']
            features['major_status_p2'] = status_anomali_result['major_status_p2']
            features['major_status_diff'] = status_anomali_result['major_status_diff']
            
            moves_result = create_move_features(timeline)
            features['p1_move_power_weighted'] = moves_result['p1_move_power_weighted']
            features['p1_number_attacks'] = moves_result['p1_number_attacks']
            features['p1_number_status'] = moves_result['p1_number_status']
            
            features['p2_move_power_weighted'] = moves_result['p2_move_power_weighted']
            features['p2_number_attacks'] = moves_result['p2_number_attacks']
            features['p2_number_status'] = moves_result['p2_number_status']
            
            """
            """
            #non ancora usate priority
            features['diff_number_attack'] = moves_result['diff_number_attack']
            features['diff_number_status'] = moves_result['diff_number_status']
            features['p1_sum_negative_priority'] = moves_result['p1_sum_negative_priority']
            features['p2_sum_negative_priority'] = moves_result['p2_sum_negative_priority']
            features['diff_negative_priority'] = moves_result['diff_negative_priority']
            """
            """
            #boost
            features.update(calcola_feature_boost(timeline))
            features.update(extract_boost_features(timeline,battle["p1_team_details"][0],battle["p2_lead_details"]))
            
            #effetti (frequenza e prima volta)
            features.update(extract_effect_features_from_timeline(timeline, "p1"))
            features.update(extract_effect_features_from_timeline(timeline, "p2"))
            
        feature_list.append(features)
    return pd.DataFrame(feature_list).fillna(0)
train_df = create_features(train_data)
test_df = create_features(test_data)
#"""
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']
#"""
import random
#'p1_sum_positive_priority','p2_sum_positive_priority',   sempre 0
seed = random.randint(0, 10_000)
n_samples=10

#describe
# print(X[['p1_team_base_hp_mean', 'p1_team_base_hp_std']].sample(n=n_samples, random_state=seed))
# exit()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np # Importa numpy per gli intervalli di parametri
from xgboost import XGBClassifier

#BASE
VOTING = True
NEW_VOTING = True
BASE = True#False#True
FINAL_VOTING = True
if FINAL_VOTING:
    import time
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import cross_val_score
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.pipeline import Pipeline

    from xgboost import XGBClassifier


    # ============================================================
    # 1. FEATURE SELECTION: Permutation Importance
    # ============================================================
    def select_top_features(model, X, y, k=50, scoring='roc_auc'):
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


    # ============================================================
    # 2. COSTRUZIONE DEL VOTING MODEL (XGB + RF + LR)
    # ============================================================
    def build_voting_model():

        # Logistic Regression (regolarizzata)
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

        # Random Forest (meno overfitting)
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=4,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=1234
        )

        # XGBoost (modello principale)
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

        # Voting Ensemble
        model = VotingClassifier(
            estimators=[
                ('xgb', xgb),
                ('rf', rf),
                ('lr', lr)
            ],
            voting='soft',
            weights=[4, 1, 1],   # XGB più influente
            n_jobs=-1
        )

        return model


    # ============================================================
    # 3. TRAINING COMPLETO (feat selection → retrain → evaluate)
    # ============================================================
    def train_with_feature_selection(X, y, k=50):

        print("\n========== FASE 1: Training iniziale con tutte le feature ==========")
        base_model = build_voting_model()

        t0 = time.time()
        base_model.fit(X, y)
        print(f"Modello iniziale addestrato in {time.time()-t0:.2f}s")

        # Feature Selection
        selected_features, importance_df = select_top_features(base_model, X, y, k=k)

        print(f"\nTop-{k} feature selezionate:")
        print(selected_features)

        print("\n========== FASE 2: Retraining con feature selezionate ==========")
        final_model = build_voting_model()

        X_sel = X[selected_features]

        t1 = time.time()
        final_model.fit(X_sel, y)
        print(f"Retraining completato in {time.time()-t1:.2f}s\n")

        # Performance
        y_pred = final_model.predict(X_sel)
        y_proba = final_model.predict_proba(X_sel)[:, 1]

        acc_cv = cross_val_score(final_model, X_sel, y, cv=5, scoring='accuracy')
        auc_cv = cross_val_score(final_model, X_sel, y, cv=5, scoring='roc_auc')

        print("\n========== RISULTATI FINALI ==========")
        print(f"Training Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"Training AUC: {roc_auc_score(y, y_proba):.4f}")
        print(f"CV Accuracy: {acc_cv.mean():.4f} ± {acc_cv.std():.4f}")
        print(f"CV AUC: {auc_cv.mean():.4f} ± {auc_cv.std():.4f}")

        return final_model, selected_features, importance_df


    # ============================================================
    # 4. ESECUZIONE FINALE
    # ============================================================
    # Assumi che X, y siano già caricati

    model, features, importance_table = train_with_feature_selection(
        X, y, k=80
    )
    X_reduced = X[features]
    def correlation_pruning(X, threshold=0.90):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        print(f"Dropped {len(to_drop)} correlated features (>{threshold}).")

        return [f for f in X.columns if f not in to_drop]
    features = correlation_pruning(X_reduced, threshold=0.92)
    print("\nModello finale pronto!")

elif NEW_VOTING:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    # --- Logistic Regression (meta stabile, per feature quasi lineari) ---
    lr = Pipeline([
        ("scaler", StandardScaler()),   # LR migliora sempre molto con scaling
        ("lr", LogisticRegression(
            C=0.5,                # più regolare (per ridurre overfit)
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=1234
        ))
    ])

    # --- Random Forest (robusto, ottimo su high-dimensional noise) ---
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=1234
    )

    # --- XGBoost (il più forte) ---
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
        tree_method="hist",       # super veloce
        random_state=1234,
        n_jobs=-1
    )

    # --- Voting Ensemble ---
    model = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('lr', lr)
        ],
        voting='soft',
        weights=[3, 1, 2],   # XGB più forte → pesa di più
        n_jobs=-1
    )
  
elif VOTING:
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.tree import DecisionTreeClassifier

    rf = RandomForestClassifier(
        n_estimators=1100,
        max_depth=5,                # più profondo
        min_samples_leaf=2,          # contro overfitting
        max_features=0.2,            # più decorrelazione
        bootstrap=True,
        class_weight='balanced',     # migliora roc_auc
        random_state=1234,
        n_jobs=-1
    )

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=500,
        learning_rate=0.5,
        random_state=1234
    )

    # gb = GradientBoostingClassifier(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=3,
    #     subsample=0.8,
    #     random_state=1234
    # )
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=5,
        random_state=1234
    )

    from sklearn.inspection import permutation_importance
    import numpy as np
    import pandas as pd

    def select_top_features(model, X, y, k=40, scoring='roc_auc'):
        """
        Calcola la permutation importance e restituisce le Top-k feature.
        RITORNA: lista con i nomi delle feature selezionate.
        """
        print("Calcolo permutation importances... (potrebbe richiedere 20-40s)")

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

        # Importances
        importances_mean = result.importances_mean
        feature_names = np.array(X.columns)

        # Ordina per importanza
        idx_sorted = np.argsort(importances_mean)[::-1]

        # Seleziona top-k
        top_features = feature_names[idx_sorted][:k]
        top_scores = importances_mean[idx_sorted][:k]

        # Output leggibile
        importance_df = pd.DataFrame({
            "feature": top_features,
            "importance": top_scores
        })

        print("\nTOP FEATURE SELECTION:")
        print(importance_df.head(20))

        return list(top_features), importance_df
    
    lr = Pipeline([
        ("scaler", StandardScaler()),
        #("pca", PCA(n_components=0.95)), # keep 95% variance
        ("lr", LogisticRegression(
            C=1,
            max_iter=1000,
            solver='liblinear',
            #penalty='l1' 
        ))
    ])
    model = VotingClassifier(
        estimators=[
            ('rf', rf),
            # ('ada', ada),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft',
        weights=[1,1, 2],  #tune on performance
        n_jobs=-1
    )
    # scegli K = 40 (poi provo 30–60)
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
else:

    rf = RandomForestClassifier(random_state=1234, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=1234)
    log_reg = LogisticRegression(random_state=1234, max_iter=2000)

   

    stacked_model_base = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        final_estimator=XGBClassifier(random_state=1234, n_estimators=100, learning_rate=0.05),
        passthrough=True,
        n_jobs=-1
    )

    param_grid = {
        # --- Random Forest ---
        'rf__n_estimators': [100, 300, 500],
        'rf__max_depth': [None, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        
        # --- Gradient Boosting ---
        'gb__n_estimators': [100, 200, 300],
        'gb__learning_rate': [0.01, 0.05, 0.1],
        'gb__max_depth': [2, 3, 5],
        'gb__subsample': [0.8, 1.0],
        
        # --- XGBoost (meta-model) ---
        'final_estimator__n_estimators': [100, 200, 300],
        'final_estimator__learning_rate': [0.01, 0.05, 0.1],
        'final_estimator__max_depth': [3, 5, 7],
        'final_estimator__subsample': [0.8, 1.0],
        'final_estimator__colsample_bytree': [0.8, 1.0]
    }


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    from sklearn.model_selection import RandomizedSearchCV

    model = RandomizedSearchCV(
        estimator=stacked_model_base,
        param_distributions=param_grid,
        n_iter=50,  #prova anche con 100?
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        random_state=1234
    )

import itertools

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
import time
def final():
    
    # final_features = [
    #             'diff_type_advantage', 'p1_type_advantage', 
    #             'diff_status',
    #             'diff_speed_first', 'diff_stat',
    #             'p2_hp_pct_sum', 'diff_hp_pct',
    #             'p1_number_attacks', 'p1_number_status',
    #             'p2_sum_negative_priority', 'p1_move_power_weighted',#
    #             'boost_p1', 'boost_p2', 'diff_boost',#14
    #             'sum_stat_lead_p1', 'sum_stat_lead_p2',#16
    #         ]
    selected = features

    X_selected = X[selected]
    start = time.time()
    model.fit(X_selected, y)
    final_pipe = model
    #EVALUATE

    y_train_pred = final_pipe.predict(X_selected)
    y_train_proba = final_pipe.predict_proba(X_selected)[:, 1]

    #CHECK OVERFITTING
    acc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='accuracy')
    auc = cross_val_score(final_pipe, X_selected, y, cv=5, scoring='roc_auc')
    print("featureArray,accuracy_score_training,roc_auc_score,accuracy_cross_val_score,roc_auc_cross_val_score")
    print(f"{[f for f in selected]},\n[{int(time.time()-start)}sec-{len(selected)}feat]\n{accuracy_score(y, y_train_pred)}->{acc.mean():.4f} ± {acc.std():.4f}, {roc_auc_score(y, y_train_proba)}->{auc.mean():.4f} ± {auc.std():.4f}")

    predict_and_submit(test_df, selected, final_pipe)
final()
