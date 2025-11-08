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
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
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
        return 1 / mean_weakness
    else:
        return 1.0  # team senza debolezze → massima resistenza


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

        
        if p1_team:
            p1_mean_hp = np.mean([p.get('base_hp', 0) for p in p1_team])
            p1_mean_spe = np.mean([p.get('base_spe', 0) for p in p1_team])
            p1_mean_atk = np.mean([p.get('base_atk', 0) for p in p1_team])
            p1_mean_def = np.mean([p.get('base_def', 0) for p in p1_team])
            p1_mean_spd = np.mean([p.get('base_spd', 0) for p in p1_team])

            features['p1_mean_hp'] = p1_mean_hp
            #restyle RIMOSSO 83.89% (+/- 0.55%) => 83.98% (+/- 0.47%)
            features['p1_mean_spe'] = p1_mean_spe
            features['p1_mean_atk'] = p1_mean_atk
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
        features['diff_def'] = p1_lead_def - p2_def
        features['diff_spd'] =  p1_lead_spd - p2_spd
        
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
            #features['nr_pokemon_sconfitti_diff'] = nr_pokemon_sconfitti_p1-nr_pokemon_sconfitti_p2
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
            #features['p1_status_change'] = p1_status_change
            #features['p2_status_change'] = p2_status_change
            features['status_change_diff'] = p1_status_change - p2_status_change
            status_change_diff.append(features['status_change_diff'])

            #QUANTO IL TEAM è BILANCIATO (TIPI E VELOCITA)
            p1_types = [t for p in p1_team for t in p.get('types', []) if t != 'notype']
            #84.02% (+/- 0.58%) => 84.03% (+/- 0.57%)
            features['p1_type_diversity'] = len(set(p1_types))
            #!!
            p1_type_resistance = compute_team_resistance(p1_team, type_chart)
            result = compute_avg_type_advantage_over_timeline(timeline, pokemon_dict, type_chart)
            #features['p1_type_advantage'] = result['p1_type_advantage']
            #features['p2_type_advantage'] = result['p2_type_advantage']
            #features['diff_type_advantage'] = result['diff_type_advantage']

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

    # --- Build base pipeline ---
    steps = []
    if POLY_ENABLED:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    steps.append(("scaler", StandardScaler()))
    if USE_PCA:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))

    steps.append(("logreg", LogisticRegression(max_iter=2000, random_state=42)))
    pipe = Pipeline(steps)

    # --- Define parameter grid for GridSearchCV ---
    param_grid = [
        {
            'logreg__solver': ['liblinear'],
            'logreg__penalty': ['l1', 'l2'],
            'logreg__C': [0.01, 0.1, 1, 10],
        },
        {
            'logreg__solver': ['lbfgs'],
            'logreg__penalty': ['l2'],  # only L2 for lbfgs
            'logreg__C': [0.01, 0.1, 1, 10],
        }
    ]


    # --- Create GridSearchCV wrapper ---
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
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
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # --- Refit on all data automatically (refit=True) ---
    best_model = grid_search.best_estimator_

    return best_model
def train(X,y): 
    USE_PCA = False 
    POLY_ENABLED = False
    # se enabled 77.64% (+/- 0.69%) altrimenti 77.94% (+/- 0.35%) 
    steps = [] 
    if POLY_ENABLED: 
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False))) 
    #standardizza 
    steps.append(("scaler", StandardScaler())) 
    if USE_PCA: 
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full"))) # ~95% varianza 
    steps.append(("logreg", LogisticRegression(max_iter=2000, random_state=42))) 
    pipe = Pipeline(steps) 
    #kfold cross-validation 
    kfold = KFold(n_splits=5, shuffle=True, random_state=42) 
    # 5-fold CV 
    print("Training Logistic Regression con 5-Fold Cross-Validation...\n") 
    scores = cross_val_score(pipe, X, y, cv=kfold, scoring='accuracy', n_jobs=-1) 
    print(f"Cross-validation accuracies: {np.round(scores, 4)}") 
    print(f"Mean CV accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)") 
    #Training finale 
    pipe.fit(X, y) 
    print("\nFinal model trained on all training data.") 
    return pipe

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
