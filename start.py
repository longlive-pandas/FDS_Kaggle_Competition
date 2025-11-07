# -*- coding: utf-8 -*-

import json
import pandas as pd
import os
from scipy.stats import linregress
import sys



# Simplified Pokémon type effectiveness chart
# Values: 2.0 = super effective, 0.5 = not very effective, 0.0 = no effect
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
from collections import defaultdict
import copy

from collections import defaultdict
import copy

def moves_by_pokemon_list(timeline, unique=True):
    """
    Return per-player dict of {pokemon_name: [move_details, ...]}.
    
    Args:
        battle (dict): single battle record with 'battle_timeline'.
        unique (bool): if True, include each distinct move once per Pokémon (order of first use).
                       if False, include a move entry every time it was used.
    """
    result = {"p1": defaultdict(list), "p2": defaultdict(list)}
    seen_keys = {"p1": defaultdict(set), "p2": defaultdict(set)} if unique else None

    for turn in timeline or []:
        # P1
        p1_name = (turn.get("p1_pokemon_state") or {}).get("name")
        p1_move = turn.get("p1_move_details")
        if p1_name and p1_move:
            if unique:
                key = (
                    p1_move.get("name"),
                    p1_move.get("type"),
                    p1_move.get("category"),
                    p1_move.get("base_power"),
                    p1_move.get("accuracy"),
                    p1_move.get("priority"),
                )
                if key not in seen_keys["p1"][p1_name]:
                    seen_keys["p1"][p1_name].add(key)
                    result["p1"][p1_name].append(copy.deepcopy(p1_move))
            else:
                result["p1"][p1_name].append(copy.deepcopy(p1_move))

        # P2
        p2_name = (turn.get("p2_pokemon_state") or {}).get("name")
        p2_move = turn.get("p2_move_details")
        if p2_name and p2_move:
            if unique:
                key = (
                    p2_move.get("name"),
                    p2_move.get("type"),
                    p2_move.get("category"),
                    p2_move.get("base_power"),
                    p2_move.get("accuracy"),
                    p2_move.get("priority"),
                )
                if key not in seen_keys["p2"][p2_name]:
                    seen_keys["p2"][p2_name].add(key)
                    result["p2"][p2_name].append(copy.deepcopy(p2_move))
            else:
                result["p2"][p2_name].append(copy.deepcopy(p2_move))

    # cast defaultdicts to dicts
    result["p1"] = dict(result["p1"])
    result["p2"] = dict(result["p2"])
    return result

#0 Mean CV accuracy: 83.59% (+/- 0.64%)
#1 Mean CV accuracy: 83.62% (+/- 0.67%)
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

#2 Mean CV accuracy: 83.55% (+/- 0.63%)
"""
A small decrease (≈ 0.05 %) means:
The new status_duration_ratio feature adds noise but not independent signal.
The effect of bad statuses is already encoded in your existing features — especially p1_bad_status_advantage and status_change_diff, both of which have strong correlations (±0.5 – 0.6) with the target.
Logistic Regression, being linear, can’t easily “disentangle” two highly collinear features, so adding another version of the same concept just shifts weights slightly.
"""
def status_duration_ratio(battle):
    p1_turns_bad = sum(turn['p1_pokemon_state']['status'] != 'nostatus' for turn in battle['battle_timeline'])
    p2_turns_bad = sum(turn['p2_pokemon_state']['status'] != 'nostatus' for turn in battle['battle_timeline'])
    return (p2_turns_bad + 1) / (p1_turns_bad + 1)
#3 Pre-battle info
#Mean CV accuracy: 83.59% (+/- 0.64%)
"""
Your feature space is already highly saturated with the key linear signals — the Logistic Regression can’t extract more because:
Many of your engineered variables (HP diffs, status advantages, competitiveness metrics) are strongly correlated with one another.
Logistic Regression can only assign one weight per direction of correlation, so redundant inputs don’t add discriminative power.
You’re probably sitting near the model’s linear ceiling for this dataset (~83–84%).
This is good news — it means your data cleaning and baseline engineering are solid.
Now, the final 2–3% (to reach 86%) likely requires non-linear relationships or interaction modeling.
"""
def team_stat_variance(team):
    """Average variance of base stats within Player 1's team."""
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    # Compute variance of each stat across the 6 Pokémon, then take the mean
    return float(np.mean([np.var([p[s] for p in team]) for s in stats]))

def team_type_diversity(team):
    """Proportion of distinct Pokémon types in the team."""
    all_types = [t for p in team for t in p['types'] if t != 'notype']
    if not all_types:
        return 0.0
    return len(set(all_types)) / len(all_types)

def lead_stat_diff(p1_team, p2_lead):
    """Average stat difference between Player 1's team mean and Player 2's lead Pokémon."""
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    p1_mean = np.mean([[p[s] for s in stats] for p in p1_team], axis=0)
    p2_stats = np.array([p2_lead[s] for s in stats])
    return float(np.mean(p1_mean - p2_stats))

#5 p1_move_damage_mean, p2_move_damage_mean, diff_move_damage_mean
def move_damage_efficiency(battle):
    moves = [t.get('p1_move_details', {}) for t in battle['battle_timeline'] if t.get('p1_move_details')]
    if not moves: return 0
    total_damage = sum(m.get('base_power', 0) for m in moves)
    return total_damage / len(moves)  # avg damage per move
#6 p1_first_strike_ratio, tempo_advantage = first_strike_ratio - 0.5
"""
Cross-validation accuracies: [0.844  0.827  0.831  0.8375 0.835 ]
Mean CV accuracy: 83.49% (+/- 0.58%)

1. Logistic regression doesn’t automatically benefit from new features
Even though new features (like p1_move_damage_mean or tempo_advantage) sound informative,
your model is a linear classifier with L1 penalty.
If the new variables:
are correlated with existing ones (hp_diff_mean, diff_final_hp, etc.),
or noisy (e.g. move damage varying randomly per battle),
then logistic regression’s L1 regularization will zero them out,
or worse — slightly destabilize weights and hurt generalization by adding variance.
That’s why performance went from 83.52 → 83.49% — statistically flat, but slightly worse.
⚙️ 2. These specific features overlap conceptually with existing ones
New Feature	Likely Redundant With	Explanation
p1_move_damage_mean, diff_move_damage_mean	diff_final_hp, hp_diff_mean	Both describe how much damage one side deals over time.
p1_first_strike_ratio, tempo_advantage	hp_delta_trend	Both encode “initiative”: who tends to lead or inflict earlier HP drops.
So they add little new orthogonal signal — linear models can’t combine correlated predictors effectively.
🧩 3. Why it doesn’t mean they’re useless
These features might be valuable for:
nonlinear models (Random Forest, Gradient Boosting, XGBoost)
interaction terms (e.g. PolynomialFeatures, cross terms)
or temporal clustering if used with per-turn modeling.
But in your current setup (L1 LogisticRegressionCV), adding correlated variables = minor penalty.
"""
def first_strike_ratio(battle):
    timeline = battle.get("battle_timeline", [])
    if len(timeline) < 2:
        return 0.0

    p1_hp = [t["p1_pokemon_state"]["hp_pct"] for t in timeline if t.get("p1_pokemon_state")]
    p2_hp = [t["p2_pokemon_state"]["hp_pct"] for t in timeline if t.get("p2_pokemon_state")]

    # approximate: if p2_hp decreases before p1_hp does, p1 attacked first
    p1_first_hits = sum((p2_hp[i] - p2_hp[i+1]) > (p1_hp[i] - p1_hp[i+1]) for i in range(len(p1_hp)-1))
    ratio = p1_first_hits / (len(p1_hp) - 1)
    return ratio

#7 hp_momentum_flips, hp_flip_rate = flips / len(timeline)
"""perf drop
Cross-validation accuracies: [0.841  0.8285 0.832  0.8385 0.8355]
Mean CV accuracy: 83.51% (+/- 0.45%)
hp_momentum_flips is a temporal volatility feature — it counts how many times the HP advantage flips between the two players.
It measures “momentum chaos”: stable battles (few flips) vs. back-and-forth ones (many flips).

Why performance drops when adding more features
You’re already near the linear ceiling.
Logistic regression is extracting essentially all the linear signal your data has — roughly 83.6 ± 0.6 % looks like your model’s upper bound.
Additional features that are noisy or weakly correlated tend to only add variance, not information.
New features like hp_momentum_flips and hp_flip_rate are non-linear effects.
They’re based on the sequence dynamics of battles, which a linear model can’t capture well unless the relationship with player_won is monotonic.
Logistic regression assumes a smooth, single-direction effect — but volatility in battles isn’t monotonic:
Some flips = competitive battle (either could win)
Too few flips = complete dominance (p1 wins or loses early)
That kind of U-shape is invisible to linear models.
L1 regularization doesn’t help if new variables are weakly correlated.
Even though the L1 penalty tries to zero out irrelevant features, slight noise in the cross-validation folds can make the coefficients dance a bit → small accuracy drop.
⚙️ What this means for you
✅ Your base feature set is solid — most meaningful information is already encoded in HP, final differences, and status metrics.
⚠️ New dynamic features (momentum, tempo, move damage) will help only if you switch to:
tree-based models (RandomForest, XGBoost, LightGBM), or
polynomial / interaction expansion of selected features.
"""
def hp_momentum_flips(battle):
    timeline = battle.get("battle_timeline", [])
    if len(timeline) < 2:
        return 0.0
    hp_adv = np.array([
        t.get("p1_pokemon_state", {}).get("hp_pct", 0) -
        t.get("p2_pokemon_state", {}).get("hp_pct", 0)
        for t in timeline
    ])
    # Count how many times the advantage sign changes
    flips = np.sum(np.sign(hp_adv[1:]) != np.sign(hp_adv[:-1]))
    return float(flips)

#8 p1_lead_type_advantage, p2_lead_type_advantage, diff_type_advantage
def type_effectiveness(attacker_types, defender_types):
    """Compute average type effectiveness multiplier between attacker and defender."""
    if not attacker_types or not defender_types:
        return 1.0  # neutral if missing

    values = []
    for a in attacker_types:
        if a not in type_chart:
            continue
        for d in defender_types:
            values.append(type_chart[a].get(d, 1.0))  # default to neutral (1.0)
    return np.mean(values) if values else 1.0

#9 battle_duration, hp_loss_rate = diff_final_hp / battle_duration
def battle_duration(battle):
    return len([t for t in battle['battle_timeline'] if t['p1_pokemon_state']['hp_pct'] > 0 and
                                                     t['p2_pokemon_state']['hp_pct'] > 0])
#10 p1_team_imbalance, p2_team_imbalance, diff_team_imbalance
"""
Why battle_duration and hp_loss_rate helped
These two features add time-normalized efficiency, which Logistic Regression can model linearly:
Feature	Meaning	Why it helps
battle_duration	How long both sides were active	Distinguishes between decisive vs drawn-out matches
hp_loss_rate	HP difference per turn	Captures speed of dominance — short + large HP gap = strong linear win signal
They add orthogonal information to your HP and final-state metrics (diff_final_hp, hp_delta_trend), improving linear separability.
"""
def team_stat_imbalance(team):
    totals = [p['base_hp']+p['base_atk']+p['base_def']+p['base_spa']+p['base_spd']+p['base_spe'] for p in team]
    return np.std(totals) / np.mean(totals)
"""
attack_to_speed_ratio = diff_atk / (diff_spe + 1e-6)
hp_control_ratio = hp_diff_mean / hp_delta_std
status_pressure = status_change_diff / (hp_momentum_flips + 1)
"""
# --- Define the path to our data ---
COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('input', COMPETITION_NAME)

train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')
train_data = []

# Read the file line by line
print(f"Loading data from '{train_file_path}'...")


######analisi varie dati
import json
import pandas as pd
import json

prova_data = []
with open(train_file_path, "r") as f:
    prova_data = [json.loads(line) for line in f]
def extract_pokemon_types(data):
    """
    Extract a dictionary of all unique Pokémon names and their associated types
    from the dataset (p1_team_details and p2_lead_details across all battles).
    """
    pokemon_types = {}

    for battle in data:
        # --- Player 1 team ---
        p1_team = battle.get("p1_team_details", [])
        for p in p1_team:
            name = p.get("name")
            types = p.get("types", [])
            if name:
                if name not in pokemon_types:
                    pokemon_types[name] = set()
                pokemon_types[name].update(t for t in types if t != "notype")

        # --- Player 2 lead ---
        p2_lead = battle.get("p2_lead_details", {})
        if p2_lead:
            name = p2_lead.get("name")
            types = p2_lead.get("types", [])
            if name:
                if name not in pokemon_types:
                    pokemon_types[name] = set()
                pokemon_types[name].update(t for t in types if t != "notype")

    # Convert type sets to sorted lists for readability
    return {name: sorted(list(types)) for name, types in pokemon_types.items()}
# Assuming you loaded the data as before
# train_data = [json.loads(line) for line in open(train_file_path)]



from collections import defaultdict
import copy

from collections import OrderedDict

def _normalize_move(m):
    """Return a compact, hashable signature + a cleaned dict for a move."""
    if not m or not isinstance(m, dict):
        return None, None
    name = m.get("name")
    if not name:
        return None, None
    sig = (
        name,
        m.get("type"),
        m.get("category"),
        m.get("base_power"),
        m.get("accuracy"),
        m.get("priority"),
    )
    cleaned = {
        "name": name,
        "type": m.get("type"),
        "category": m.get("category"),
        "base_power": m.get("base_power"),
        "accuracy": m.get("accuracy"),
        "priority": m.get("priority"),
    }
    return sig, cleaned

def extract_moves_per_battle(battle):
    """
    Returns:
      {
        "p1": { pokemon_name: [ {move dict}, ... ], ... },
        "p2": { pokemon_name: [ {move dict}, ... ], ... },
      }
    Lists contain unique moves (by full signature), preserving first-seen order.
    """
    result = {"p1": {}, "p2": {}}

    # Ordered sets per (player,pokemon) to preserve order and deduplicate
    seen = {"p1": {}, "p2": {}}

    for turn in battle.get("battle_timeline", []) or []:
        # P1
        p1_state = turn.get("p1_pokemon_state") or {}
        p1_name  = p1_state.get("name")
        p1_move  = turn.get("p1_move_details")
        sig, move = _normalize_move(p1_move)
        if p1_name and sig:
            if p1_name not in seen["p1"]:
                seen["p1"][p1_name] = OrderedDict()
                result["p1"][p1_name] = []
            if sig not in seen["p1"][p1_name]:
                seen["p1"][p1_name][sig] = True
                result["p1"][p1_name].append(move)

        # P2
        p2_state = turn.get("p2_pokemon_state") or {}
        p2_name  = p2_state.get("name")
        p2_move  = turn.get("p2_move_details")
        sig, move = _normalize_move(p2_move)
        if p2_name and sig:
            if p2_name not in seen["p2"]:
                seen["p2"][p2_name] = OrderedDict()
                result["p2"][p2_name] = []
            if sig not in seen["p2"][p2_name]:
                seen["p2"][p2_name][sig] = True
                result["p2"][p2_name].append(move)

    return result
from collections import defaultdict

def extract_all_moves(data):
    """
    Scan all battles/timelines and collect unique moves with metadata + usage counts.

    Returns:
      {
        move_name: {
          "type": str|None,
          "category": str|None,
          "base_power": int|float|None,
          "accuracy": float|None,
          "priority": int|None,
          "usage_count": int
        }, ...
      }
    """
    moves = defaultdict(lambda: {
        "type": None, "category": None, "base_power": None,
        "accuracy": None, "priority": None, "usage_count": 0
    })

    for battle in data:
        for turn in battle.get("battle_timeline", []) or []:
            for key in ("p1_move_details", "p2_move_details"):
                mv = turn.get(key)
                sig, m = _normalize_move(mv)
                if not m:
                    continue
                name = m["name"]
                entry = moves[name]
                # keep first non-None metadata we see
                entry["type"]       = entry["type"]       if entry["type"] is not None       else m["type"]
                entry["category"]   = entry["category"]   if entry["category"] is not None   else m["category"]
                entry["base_power"] = entry["base_power"] if entry["base_power"] is not None else m["base_power"]
                entry["accuracy"]   = entry["accuracy"]   if entry["accuracy"] is not None   else m["accuracy"]
                entry["priority"]   = entry["priority"]   if entry["priority"] is not None   else m["priority"]
                entry["usage_count"] += 1

    return dict(moves)


# Single battle
#per_battle = extract_moves_per_battle(one_battle)
# per_battle["p1"]["starmie"] -> list of starmie’s unique moves (dicts)

# Whole dataset
all_moves = extract_all_moves(prova_data)
print(len(all_moves))
# sort by usage
top = sorted(all_moves.items(), key=lambda x: x[1]["usage_count"], reverse=True)[:10]
for name, info in top:
    print(name, info)


#exit(1)
# Supponiamo che tu abbia già caricato il file JSONL in train_data
# come lista di dict (uno per battle)
"""
Totale mosse con priorità diversa da 0: 1
  move_name  priority      type  category  count
0   counter        -1  FIGHTING  PHYSICAL   3454

moves_with_priority = []
with open(train_file_path, "r") as f:
    records = [json.loads(line) for line in f]
for battle in records:
    for turn in battle.get("battle_timeline", []):
        for key in ["p1_move_details", "p2_move_details"]:
            move = turn.get(key)
            if move and move.get("priority", 0) != 0:
                moves_with_priority.append({
                    "battle_id": battle["battle_id"],
                    "player": key.split("_")[0],  # p1 o p2
                    "move_name": move.get("name", None),
                    "priority": move.get("priority", 0),
                    "type": move.get("type", None),
                    "category": move.get("category", None),
                })

# Convertiamo in DataFrame
df_moves = pd.DataFrame(moves_with_priority)
print("df moves: ",df_moves)
# Mostriamo le mosse distinte con priorità ≠ 0 e il loro conteggio
distinct_moves = (
    df_moves.groupby(["move_name", "priority", "type", "category"])
    .size()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)

print(f"Totale mosse con priorità diversa da 0: {len(distinct_moves)}")
print(distinct_moves)
exit(1)
"""
"""
priority
     count
-1    3454
 0  467320
import json
import pandas as pd
from collections import Counter

# --- Load your data ---
with open(train_file_path, "r") as f:
    records = [json.loads(line) for line in f]

# --- Extract all priorities from both players ---
priorities = []

for battle in records:
    for turn in battle.get("battle_timeline", []):
        for side in ["p1_move_details", "p2_move_details"]:
            move = turn.get(side)
            if isinstance(move, dict):  # skip nulls
                prio = move.get("priority")
                if prio is not None:
                    priorities.append(prio)

# --- Count all unique priority values ---
priority_counts = Counter(priorities)

# --- Convert to DataFrame for easy display and plotting ---
priority_df = pd.DataFrame.from_dict(priority_counts, orient="index", columns=["count"]).sort_index()

print(priority_df)

sys.exit(1)


import json
import pandas as pd

# --- Load your JSONL file ---
with open(train_file_path, "r") as f:
    records = [json.loads(line) for line in f]

def analyze_nested_lists(records):
    #Scan all list-type fields and count inner missing values.
    list_fields = {}
    
    # --- Step 1: detect which top-level fields are lists ---
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, list):
                list_fields.setdefault(k, []).append(v)
    
    summary = {}

    # --- Step 2: iterate through each list field ---
    for field, lists in list_fields.items():
        field_missing = {}
        total_items = 0

        for lst in lists:
            if not isinstance(lst, list):
                continue
            for item in lst:
                if isinstance(item, dict):
                    total_items += 1
                    for key, val in item.items():
                        field_missing.setdefault(key, 0)
                        if val is None:
                            field_missing[key] += 1

        summary[field] = {"total_items": total_items, "missing_counts": field_missing}

    return summary

# --- Run analysis ---
nested_summary = analyze_nested_lists(records)

# --- Print results ---
for field, info in nested_summary.items():
    print(f"\n📂 Field: {field}")
    print(f"   Total inner items: {info['total_items']}")
    print("   Missing counts:")
    for subfield, count in info["missing_counts"].items():
        print(f"     {subfield:25s}: {count}")

sys.exit()
# Load all battles
with open(train_file_path, 'r') as f:
    records = [json.loads(line) for line in f]

# Flatten just top level
data = pd.json_normalize(records)

# Now inspect nested 'p1_team_details'
missing_counts = {}

for battle in records:
    team = battle.get("p1_team_details", [])
    for p in team:
        for key, value in p.items():
            if key not in missing_counts:
                missing_counts[key] = 0
            if value is None:
                missing_counts[key] += 1

print("Missing values inside p1_team_details:")
for k, v in missing_counts.items():
    print(f"{k:20s}: {v}")

sys.exit()
# Read and parse every line into a list of dicts
with open(train_file_path, 'r') as f:
    records = [json.loads(line) for line in f]

# Flatten nested fields into dot-separated column names
data = pd.json_normalize(records)

# Inspect the result
print(data.head())
print("\nColumn names:")
print(data.columns.tolist())

# Check missing values
print("\nMissing values per column:")
print(data.isnull().sum())
sys.exit()
# Read entire JSONL file into a DataFrame
data = pd.read_json(train_file_path, lines=True)

# Display first few rows
print("Preview:")
print(data.head())

# Show missing values per column
print("\nMissing values per column:")
print(data.isnull().sum())
#type check
###
def explore_structure(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: {type(v).__name__}")
            explore_structure(v, indent + 1)
    elif isinstance(obj, list) and obj:
        print(f"{prefix}[List of {len(obj)} elements, type {type(obj[0]).__name__}]")
        explore_structure(obj[0], indent + 1)
battle = None
with open(train_file_path, "r") as f:
    first_line = f.readline()
    battle = json.loads(first_line)
explore_structure(battle)
print("Stop")
sys.exit()
df = pd.read_json(train_file_path, lines=True)
df_flat = pd.json_normalize(df.to_dict(orient="records"))
print(df_flat.info())
with open(train_file_path, 'r') as f:
    first_line = f.readline()  # ✅ reads a single line as a string
    data = json.loads(first_line)  # parse it into a Python dict
    df = pd.json_normalize(data)   # flatten nested JSON if needed

print(df.dtypes)
print("Stop")
sys.exit()
"""
try:
    with open(train_file_path, 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            train_data.append(json.loads(line))

    print(f"Successfully loaded {len(train_data)} battles.")

    # Let's inspect the first battle to see its structure
    print("\n--- Structure of the first train battle: ---")
    if train_data:
        first_battle = train_data[0]

        # To keep the output clean, we can create a copy and truncate the timeline
        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2] # Show first 2 turns

        # Use json.dumps for pretty-printing the dictionary
        print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for #display)")


except FileNotFoundError:
    print(f"ERROR: Could not find the training file at '{train_file_path}'.")
    print("Please make sure you have added the competition data to this notebook.")

from tqdm.notebook import tqdm
import numpy as np
# --- Type effectiveness table (semplificata) ---
import numpy as np
TYPE_EFFECTIVENESS = {
    ('fire', 'grass'): 2.0,
    ('water', 'fire'): 2.0,
    ('electric', 'water'): 2.0,
    ('grass', 'water'): 2.0,
    ('ice', 'grass'): 2.0,
    ('psychic', 'poison'): 2.0,
    ('ground', 'electric'): 2.0,
    ('rock', 'fire'): 2.0,
    ('fighting', 'normal'): 2.0,
    ('ghost', 'psychic'): 2.0,
    # neutral = 1.0 if not found
}

import numpy as np

def type_advantage_dynamic(battle, pokemon_type_dict, type_chart):
    """
    Compute average type advantage over all turns for both players.

    Args:
        battle (dict): a single battle record with 'battle_timeline'.
        pokemon_type_dict (dict): {pokemon_name: [type1, type2, ...]}
        type_chart (dict): type effectiveness chart.

    Returns:
        dict: {
            "p1_avg_type_adv": float,
            "p2_avg_type_adv": float
        }
    """

    timeline = battle.get("battle_timeline", []) or []

    def compute_effectiveness(move_type, defender_types):
        """Compute summed type effectiveness of a move on a defending Pokémon."""
        if not move_type or not defender_types:
            return 1.0
        eff = 0.0
        for dtype in defender_types:
            eff += type_chart.get(move_type.capitalize(), {}).get(dtype.capitalize(), 1.0)
        return eff

    p1_scores, p2_scores = [], []

    for turn in timeline:
        # --- Player 1 move advantage ---
        p1_move = turn.get("p1_move_details")
        p2_state = turn.get("p2_pokemon_state") or {}
        if p1_move and p2_state:
            move_type = p1_move.get("type")
            defender = p2_state.get("name")
            defender_types = pokemon_type_dict.get(defender, [])
            p1_scores.append(compute_effectiveness(move_type, defender_types))

        # --- Player 2 move advantage ---
        p2_move = turn.get("p2_move_details")
        p1_state = turn.get("p1_pokemon_state") or {}
        if p2_move and p1_state:
            move_type = p2_move.get("type")
            defender = p1_state.get("name")
            defender_types = pokemon_type_dict.get(defender, [])
            p2_scores.append(compute_effectiveness(move_type, defender_types))

    # Compute averages (default to 1.0 if no moves)
    p1_avg = float(np.mean(p1_scores)) if p1_scores else 1.0
    p2_avg = float(np.mean(p2_scores)) if p2_scores else 1.0

    return {
        "p1_avg_type_adv": p1_avg,
        "p2_avg_type_adv": p2_avg,
        "diff_type_adv": p1_avg - p2_avg
    }


def type_advantage(team_types: list[str], opp_types: list[str]) -> float:
    """
    Compute the average type advantage multiplier for player1's team
    against opponent's lead types.
    """
    scores = []
    for t1 in team_types:
        for t2 in opp_types:
            scores.append(TYPE_EFFECTIVENESS.get((t1.lower(), t2.lower()), 1.0))
    return np.mean(scores) if scores else 1.0

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

def create_simple_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    #p1_bad_status_advantage = []
    status_change_diff = []
    pokemon_type_dict = extract_pokemon_types(data)
    for battle in tqdm(data, desc="Extracting features"):

        features = {}

        
        #print(f"Found {len(pokemon_type_dict)} Pokémon with type information.\n")
        # Print a sample
        for name, types in list(pokemon_type_dict.items())[:10]:
            print(f"{name}: {types}")
        res = type_advantage_dynamic(battle, pokemon_type_dict, type_chart)
        #features['p1_avg_type_adv'] = res['p1_avg_type_adv']
        features['p2_avg_type_adv'] = res['p2_avg_type_adv']
        #features['diff_type_adv'] = res['diff_type_adv']
        
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
            p1_mean_spa = np.mean([p.get('base_spa', 0) for p in p1_team])

            features['p1_mean_hp'] = p1_mean_hp
            #features['p1_mean_spe'] = p1_mean_spe#reverse greedy 83.89->83.94
            features['p1_mean_atk'] = p1_mean_atk
            features['p1_mean_def'] = p1_mean_def
            features['p1_mean_sp'] = p1_mean_spd
            #features['p1_mean_spa'] = p1_mean_spa#decrease score, increase std

            #PER UN CONFRONTO EQUO UTILIZZIAMO SOLO DATI DEL LEADER ANCHE NELLA SQUADRA 1 PER LE DIFFERENZE
            p1_lead_hp =  p1_team[0].get('base_hp', 0)
            p1_lead_spe = p1_team[0].get('base_spe', 0)
            p1_lead_atk = p1_team[0].get('base_atk', 0)
            p1_lead_def = p1_team[0].get('base_def', 0)
            p1_lead_spd =  p1_team[0].get('base_spd', 0)
            p1_lead_spa =  p1_team[0].get('base_spa', 0)


        # --- Player 2 Lead Features ---
        p2_hp = p2_spe = p2_atk = p2_def = p2_spd = p2_spa= 0.0
        p2_lead = battle.get('p2_lead_details')
        if p2_lead:
            # Player 2's lead Pokémon's stats
            p2_hp = p2_lead.get('base_hp', 0)
            p2_spe = p2_lead.get('base_spe', 0)
            p2_atk = p2_lead.get('base_atk', 0)
            p2_def = p2_lead.get('base_def', 0)
            p2_spd = p2_lead.get('base_spd', 0)
            p2_spa = p2_lead.get('base_spa', 0)


        # I ADD THE DIFFS/DELTAS
        features['diff_hp']  = p1_lead_hp  - p2_hp#68->87
        features['diff_spe'] = p1_lead_spe - p2_spe
        #features['diff_atk'] = p1_lead_atk - p2_atk#reverse greedy
        features['diff_def'] = p1_lead_def - p2_def
        features['diff_spd'] =  p1_lead_spd - p2_spd#83->87
        #features['diff_spa'] =  p1_lead_spa - p2_spa#83->87

        #8 new
        # --- Type advantage (lead vs. lead) ---
        p1_lead_types = [t for t in p1_team[0].get("types", []) if t != "notype"] if p1_team else []
        p2_lead_types = [t for t in p2_lead.get("types", []) if t != "notype"] if p2_lead else []
        p1_lead_type_advantage = type_effectiveness(p1_lead_types, p2_lead_types)
        p2_lead_type_advantage = type_effectiveness(p2_lead_types, p1_lead_types)
        #features["p1_lead_type_advantage"] = p1_lead_type_advantage
        #features["p2_lead_type_advantage"] = p2_lead_type_advantage
        #no increase no decrease
        #features["diff_type_advantage"] = p1_lead_type_advantage - p2_lead_type_advantage
        #new2
        #features['status_duration_ratio'] = status_duration_ratio(battle)

        #new3 - static feat
        """
        features['p1_team_stat_variance'] = team_stat_variance(battle['p1_team_details'])
        features['p1_type_diversity_ratio'] = team_type_diversity(battle['p1_team_details'])
        features['lead_stat_diff'] = lead_stat_diff(battle['p1_team_details'], battle['p2_lead_details'])
        """
        #DYNAMIC INFO
        #Chi mantiene più HP medi e conduce più turni,  nella maggior parte dei casi vince anche se la battaglia non è ancora finita
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

            # Convert lists to NumPy arrays for vectorized math
            p1_atk_boosts = np.array(p1_atk_boosts)
            p2_atk_boosts = np.array(p2_atk_boosts)
            p1_def_boosts = np.array(p1_def_boosts)
            p2_def_boosts = np.array(p2_def_boosts)

            # Retrieve average move damages
            p1_move_damage_mean = move_damage_efficiency({
                'battle_timeline': [t for t in timeline if t.get('p1_move_details')]
            })
            p2_move_damage_mean = move_damage_efficiency({
                'battle_timeline': [t for t in timeline if t.get('p2_move_details')]
            })

            # Combine boost * damage - opponent defense_boost
            # Measures per-turn offensive pressure

            #non incrementa lo score considerando boost atk difesa e damage
            # if len(p1_atk_boosts) and len(p2_def_boosts):
            #     p1_synergy_vals = (p1_atk_boosts * p1_move_damage_mean) - p2_def_boosts
            #     p2_synergy_vals = (p2_atk_boosts * p2_move_damage_mean) - p1_def_boosts
            #     #features["p1_offensive_synergy"] = np.mean(p1_synergy_vals)
            #     #features["p2_offensive_synergy"] = np.mean(p2_synergy_vals)
            # else:
            #     features["p1_offensive_synergy"] = 0.0
            #     #features["p2_offensive_synergy"] = 0.0

            #features["p2_offensive_synergy"] = (1 + mean_p2_atk_boost) * p2_move_damage_mean / (1 + p1_mean_def)
            """
            # --- Boost + Damage Combined Advantage ---

            # Calcolo media danno per player 1 e 2
            p1_move_damage_mean = move_damage_efficiency({
                'battle_timeline': [t for t in timeline if t.get('p1_move_details')]
            })
            p2_move_damage_mean = move_damage_efficiency({
                'battle_timeline': [t for t in timeline if t.get('p2_move_details')]
            })
            diff_move_damage_mean = p1_move_damage_mean - p2_move_damage_mean

            # Calcolo media differenze di boost nei 30 turni
            boost_keys = ["atk", "def", "spa", "spd", "spe"]
            turn_diffs = []
            for turn in timeline:
                p1_boosts = turn.get("p1_pokemon_state", {}).get("boosts", {})
                p2_boosts = turn.get("p2_pokemon_state", {}).get("boosts", {})
                if not p1_boosts or not p2_boosts:
                    continue
                diffs = [p1_boosts.get(k, 0) - p2_boosts.get(k, 0) for k in boost_keys]
                turn_diffs.append(np.mean(diffs))

            mean_boost_diff = np.mean(turn_diffs) if turn_diffs else 0.0

            # Combina boost e danno in una singola feature
            boost_damage_advantage = 0.0
            try:
                boost_damage_advantage = diff_move_damage_mean * mean_boost_diff
            except:
                boost_damage_advantage = 0.0
                print("err")#exit(diff_move_damage_mean,mean_boost_diff)
            #drop 0.1% !!rivedi
            #features["boost_damage_advantage"] = boost_damage_advantage
            """


            #level diff
            # --- Average level difference during battle (only when both levels known) ---
            # level_diffs = []

            # # Crea dizionario nome → livello per player 1
            # p1_levels = {p["name"]: p.get("level", np.nan) for p in battle.get("p1_team_details", [])}
            # # Player 2 ha solo il lead
            # p2_lead = battle.get("p2_lead_details", {})
            # p2_levels = {p2_lead.get("name"): p2_lead.get("level", np.nan)} if p2_lead else {}

            # for turn in timeline:
            #     p1_state = turn.get("p1_pokemon_state", {})
            #     p2_state = turn.get("p2_pokemon_state", {})
            #     if not p1_state or not p2_state:
            #         continue

            #     p1_name = p1_state.get("name")
            #     p2_name = p2_state.get("name")

            #     # Recupera i livelli, se disponibili
            #     p1_level = p1_levels.get(p1_name)
            #     p2_level = p2_levels.get(p2_name)

            #     # Considera solo se entrambi i livelli sono noti
            #     if p1_level is not None and not np.isnan(p1_level) and \
            #     p2_level is not None and not np.isnan(p2_level):
            #         level_diffs.append(p1_level - p2_level)

            # # Calcola media delle differenze (ignorando i turni senza dati)
            # mean_level_diff = np.mean(level_diffs) if level_diffs else 0.0
            # features["mean_level_diff"] = mean_level_diff
            
            #weighted priority decrease (-0.7) 83.77 => 83.70

            # --- MOVE PRIORITY FEATURES increase (+0.12)---83.77 => 83.89
            p1_priorities = []
            p2_priorities = []

            for turn in timeline:
                move1 = turn.get("p1_move_details")
                move2 = turn.get("p2_move_details")

                if isinstance(move1, dict) and move1.get("priority") is not None:
                    p1_priorities.append(move1["priority"])
                if isinstance(move2, dict) and move2.get("priority") is not None:
                    p2_priorities.append(move2["priority"])

            # Compute mean priority per player
            p1_avg_move_priority = np.mean(p1_priorities) if p1_priorities else 0.0
            p2_avg_move_priority = np.mean(p2_priorities) if p2_priorities else 0.0
            #features["p1_avg_move_priority"] = p1_avg_move_priority
            #features["p2_avg_move_priority"] = p2_avg_move_priority

            # Compute relative advantage
            features["priority_diff"] = p1_avg_move_priority - p2_avg_move_priority

            # Optional: fraction of turns where P1 had higher priority
            if p1_priorities and p2_priorities:
                min_len = min(len(p1_priorities), len(p2_priorities))
                higher_priority_turns = sum(p1_priorities[i] > p2_priorities[i] for i in range(min_len))
                priority_rate_advantage = higher_priority_turns / max(1, min_len)
                #features["priority_rate_advantage"] = priority_rate_advantage#reverse greedy
            else:
                #features["priority_rate_advantage"] = 0.0#reverse greedy
                pass

            #new feature: confronta stat se disponibili
            # --- Stat difference across turns ---
            #
            # mean_stat_diffs = {
            #     # "base_hp": [], 
            #     "base_atk": [],#83.77=>83.86% (+/- 0.50%) +0.09
            #     #  "base_def": [], #83.77=>83.72 -0.05
            #     "base_spa": [], #83.77=>83.73 -0.04; base_atk + base_spa + base_spe => 83.87% (+/- 0.54%) +0.10
            #     # "base_spd": [],#83.77=>83.73 -0.04
            #     "base_spe": []#83.77=>83.71 -0.06
            # }

            # std_stat_diffs = {
            #     # "base_hp": [], 
            #     #"base_atk": [],#83.77=>83.86% (+/- 0.50%) +0.09#reverse greedy
            #     #  "base_def": [], #83.77=>83.72 -0.05
            #     "base_spa": [], #83.77=>83.73 -0.04; base_atk + base_spa + base_spe => 83.87% (+/- 0.54%) +0.10
            #     # "base_spd": [],#83.77=>83.73 -0.04
            #     #"base_spe": []#83.77=>83.71 -0.06#reverse greedy
            # }
            # for t in timeline:
            #     p1_state = t.get("p1_pokemon_state", {})
            #     p2_state = t.get("p2_pokemon_state", {})

            #     # Retrieve names
            #     p1_name = p1_state.get("name")
            #     p2_name = p2_state.get("name")

            #     # Retrieve their base stats (if available)
            #     p1_stats = get_pokemon_stats(p1_team, p1_name) if p1_name else None
            #     p2_stats = None

            #     # p2: if same as lead, use lead; otherwise, None
            #     p2_lead = battle.get("p2_lead_details", {})
            #     if p2_name and p2_lead and p2_lead.get("name") == p2_name:
            #         p2_stats = {
            #              "base_hp": p2_lead.get("base_hp", 0),
            #             "base_atk": p2_lead.get("base_atk", 0),
            #              "base_def": p2_lead.get("base_def", 0),
            #             "base_spa": p2_lead.get("base_spa", 0),
            #              "base_spd": p2_lead.get("base_spd", 0),
            #             "base_spe": p2_lead.get("base_spe", 0)
            #         }

            #     # Skip turns where one Pokémon’s stats are unknown
            #     if not p1_stats or not p2_stats:
            #         continue

            #     # Compute differences
            #     for stat in mean_stat_diffs.keys():
            #         diff = p1_stats[stat] - p2_stats[stat]
            #         mean_stat_diffs[stat].append(diff)
            #     for stat in std_stat_diffs.keys():
            #         diff = p1_stats[stat] - p2_stats[stat]
            #         std_stat_diffs[stat].append(diff)

            # # --- Aggregate stat differences over the timeline ---
            # for stat, diffs in mean_stat_diffs.items():
            #     if diffs:
            #         features[f"mean_{stat}_diff_timeline"] = np.mean(diffs)#83.74=>83.87
            #     else:
            #         features[f"mean_{stat}_diff_timeline"] = 0.0
            # for stat, diffs in std_stat_diffs.items():
            #     if diffs:
            #         features[f"std_{stat}_diff_timeline"] = np.std(diffs)#83.78=>83.87
            #     else:
            #         features[f"std_{stat}_diff_timeline"] = 0.0

            stat_diffs = {
                # "base_hp": [], 
                "base_atk": [],#83.77=>83.86% (+/- 0.50%) +0.09
                #  "base_def": [], #83.77=>83.72 -0.05
                "base_spa": [], #83.77=>83.73 -0.04; base_atk + base_spa + base_spe => 83.87% (+/- 0.54%) +0.10
                # "base_spd": [],#83.77=>83.73 -0.04
                "base_spe": []#83.77=>83.71 -0.06
            }

            for t in timeline:
                p1_state = t.get("p1_pokemon_state", {})
                p2_state = t.get("p2_pokemon_state", {})

                # Retrieve names
                p1_name = p1_state.get("name")
                p2_name = p2_state.get("name")

                # Retrieve their base stats (if available)
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

                # Skip turns where one Pokémon’s stats are unknown
                if not p1_stats or not p2_stats:
                    continue

                # Compute differences
                for stat in stat_diffs.keys():
                    diff = p1_stats[stat] - p2_stats[stat]
                    stat_diffs[stat].append(diff)

            # --- Aggregate stat differences over the timeline ---
            stat_diff_excluded_reverse_greedy = [
                "std_base_atk_diff_timeline", "std_base_spa_diff_timeline"
            ]
            for stat, diffs in stat_diffs.items():
                std_stat_diff_name = f"std_{stat}_diff_timeline"
                if std_stat_diff_name not in stat_diff_excluded_reverse_greedy:
                    if diffs:
                        features[std_stat_diff_name] = np.std(diffs)#83.78=>83.87
                    else:
                        features[f"std_{stat}_diff_timeline"] = 0.0
                if diffs:
                    features[f"mean_{stat}_diff_timeline"] = np.mean(diffs)#83.74=>83.87
                    
                else:
                    features[f"mean_{stat}_diff_timeline"] = 0.0
                    
            #"""
            #SALUTE
            p1_hp = [t['p1_pokemon_state']['hp_pct'] for t in timeline if t.get('p1_pokemon_state')]
            p2_hp = [t['p2_pokemon_state']['hp_pct'] for t in timeline if t.get('p2_pokemon_state')]
            #salute media dei pokemon del primo giocatore
            #features['p1_mean_hp_pct'] = np.mean(p1_hp)
            #salute media dei pokemon del secondo giocatore ATTENZIONE FEATURE BUONE CORRELATE CON hp_diff_mean 75%,VALUTAZIONE DELL'EEFFETTO SU BASE SINGOLA (CON HP DIFF)
            #features['p2_mean_hp_pct'] = np.mean(p2_hp)
            #vantaggio medio in salute (media della differenza tra la salute dei pokemon del primo giocatore e quella dei pokemon del secondo giocatore)
            #features['hp_diff_mean'] = np.mean(np.array(p1_hp) - np.array(p2_hp))#reverse greedy
            
            
            """6 (performance drop)
            # --- FIRST STRIKE RATIO / TEMPO ADVANTAGE (uses helper) ---
            features['p1_first_strike_ratio'] = first_strike_ratio(battle)
            features['tempo_advantage'] = features['p1_first_strike_ratio'] - 0.5
            """
            #FINE TEST5V
            """
            # TEST5--- MOVE DAMAGE EFFICIENCY ---
            # Compute average damage per move for each player
            p1_moves = [t.get('p1_move_details', {}) for t in timeline if t.get('p1_move_details')]
            p2_moves = [t.get('p2_move_details', {}) for t in timeline if t.get('p2_move_details')]

            if p1_moves:
                p1_total_damage = sum(m.get('damage', 0) for m in p1_moves)
                features['p1_move_damage_mean'] = p1_total_damage / len(p1_moves)
            else:
                features['p1_move_damage_mean'] = 0.0

            if p2_moves:
                p2_total_damage = sum(m.get('damage', 0) for m in p2_moves)
                features['p2_move_damage_mean'] = p2_total_damage / len(p2_moves)
            else:
                features['p2_move_damage_mean'] = 0.0

            # Relative advantage: difference in mean damage dealt per move
            features['diff_move_damage_mean'] = (
                features['p1_move_damage_mean'] - features['p2_move_damage_mean']
            )
            #FINE  TEST5
            """

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
            #print(p1_hp_final)
            #numero di pockemon usati dal giocatore nei primi 30 turni
            features['p1_n_pokemon_use'] =len(p1_hp_final.keys())
            p2_n_pokemon_use = len(p2_hp_final.keys())
            #features['p2_n_pokemon_use'] = p2_n_pokemon_use#reverse greedy
            #differenza nello schieramento pockemon dopo 30 turni
            features['diff_final_schieramento']=features['p1_n_pokemon_use']-p2_n_pokemon_use
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
            #features['diff_final_hp']=diff_final_hp#reverse greedy 83.94->84.02

            #9 new
            # --- Battle duration and HP loss rate ---
            try:
                dur = battle_duration(battle)
            except Exception:
                dur = 0
            features["battle_duration"] = dur
            features["hp_loss_rate"] = (
                diff_final_hp / dur if dur > 0 else 0.0
            )

            







            #vedo anche come la salute media evolve nel tempo
            phases = 3 #early, mid, late game
            nr_turns = 30 #numero turni
            slice_idx = nr_turns // phases #slice index must be integer
            #print("slice_idx: ",slice_idx, "len p1_hp: ",len(p1_hp))
            #features['early_hp_mean_diff'] = np.mean(np.array(p1_hp[:slice_idx]) - np.array(p2_hp[:slice_idx]))#reverse greedy
            features['late_hp_mean_diff'] = np.mean(np.array(p1_hp[-slice_idx:]) - np.array(p2_hp[-slice_idx:]))
            
            
            

            #features['phases_hp_mean_diff'] = features['late_hp_mean_diff'] - features['early_hp_mean_diff']
            #77.94% (+/- 0.35%) => 77.94% (+/- 0.41%)
            hp_delta = np.array(p1_hp) - np.array(p2_hp)
            #features['hp_delta_trend'] = np.polyfit(range(len(hp_delta)), hp_delta, 1)[0]#reverse greedy
            #new
            features['hp_advantage_trend'] = hp_advantage_trend(battle)
            #6 new --- HP momentum (number of times advantage flips) ---
            """
            features['hp_momentum_flips'] = hp_momentum_flips(battle)
            features['hp_flip_rate'] = features['hp_momentum_flips'] / max(1, len(timeline))
            """
            #fluttuazioni negli hp (andamento della partita: stabile o molto caotica)
            #77.94% (+/- 0.41%) => 79.09% (+/- 1.02%)
            #features['p1_hp_std'] = np.std(p1_hp)#reverse greedy
            features['p2_hp_std'] = np.std(p2_hp)
            features['hp_delta_std'] = np.std(hp_delta)

            #9.2 new
            # --- Efficiency & stability metrics (#10) ---
            # 1️⃣ Damage per turn: how much advantage P1 gains in HP per turn
            # features["damage_per_turn"] = (
            #     diff_move_damage_mean / max(1, features["battle_duration"])
            # )
            """
            # 2️⃣ Early lead ratio: does early advantage translate into final dominance
            features["early_lead_ratio"] = (
                features["early_hp_mean_diff"] / (abs(features["diff_final_hp"]) + 1e-6)
            )

            # 3️⃣ HP stability ratio: volatility normalized by duration
            features["hp_stability_ratio"] = (
                features["hp_delta_std"] / max(1, features["battle_duration"])
            )
            """
            #fine 9.2

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
            #79.09% (+/- 1.02%) => 79.29% (+/- 0.92%)
            p1_types = [t for p in p1_team for t in p.get('types', []) if t != 'notype']
            #features['p1_type_diversity'] = len(set(p1_types))#reverse greedy

            MEDIUM_SPEED_THRESHOLD = 90 #medium-speed pokemon
            HIGH_SPEED_THRESHOLD = 100 #fast pokemon
            speeds = np.array([p.get('base_spe', 0) for p in p1_team])
            #features['p1_avg_speed_stat_battaglia'] = np.mean(np.array(speeds) > MEDIUM_SPEED_THRESHOLD)#reverse greedy
            features['p1_avg_high_speed_stat_battaglia'] = np.mean(np.array(speeds) > HIGH_SPEED_THRESHOLD)


            #COMBINAZIONI DI FEATURE
            #combino vantaggio negli hp con l'avere pochi status negativi
            #79.09% (+/- 1.02%) => 79.13% (+/- 1.06%)
            #features['hp_advantage_no_negative_status'] = features['hp_delta_trend'] * (1 - p1_negative_status_mean)
            #LA FEATURE è BELLA MA SUPER CORRELATA CON hp_delta_trend 95%
            #per il momento semplifico cosi capiamo poi è facile aggiungere

        # We also need the ID and the target variable (if it exists)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)

# Create feature DataFrames for both training and test sets
print("Processing training data...")
train_df = create_simple_features(train_data)

print("\nProcessing test data...")
test_data = []
with open(test_file_path, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
test_df = create_simple_features(test_data)

print("\nTraining features preview:")
#display(train_df.head())
train_df.head().to_json("train_df_head.json", orient="records", indent=2)
"""from sklearn.linear_model import LogisticRegression

# Define our features (X) and target (y)
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X_train = train_df[features]
y_train = train_df['player_won']

X_test = test_df[features]

# Initialize and train the model
print("Training a simple Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test data
print("Generating predictions on the test set...")
test_predictions = model.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'battle_id': test_df['battle_id'],
    'player_won': test_predictions
})

# Save the DataFrame to a .csv file
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' file created successfully!")
#display(submission_df.head())
"""

##new
"""
print(test_df.columns)
print(train_df.columns)
"""
"""
from sklearn.metrics import accuracy_score
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
"""

#train_df.describe()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Define features and target
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']

"""
no, ora faccio k-fold cross validation (sotto)

# Split training data into train and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""

"""### SCALING

# Create our scaler
scaler = StandardScaler()

# First, we want to fit our scaler to our training data and subsequently transform
# that training data through our scaler. This can all be done in a single command.
X_train = scaler.fit_transform(X_train)

# Next, we want to transform the test features by using the parameters learned
# from the training set
X_val = scaler.transform(X_val)

# Notice the values are now standardized
columns = X.columns
X_train_scaled_df = pd.DataFrame(X_train, columns = columns)
X_train_scaled_df.head()

### STUDIAMO LA CORRELAZIONE TRA FEATURE E OUTPUT E TRA FEATURE
"""







"""### PolynomialFeatures
crea nuove feature come potenze e relazioni tra le feature numeriche originali, per vedere relazioni non lineari; il modello cattura curvature e relazioni mantenendo la linearità nei parametri => purtroppo aggiunge $\binom{n+d}{n}$ feature (n=numero feature originali e d=degree=2 o altro intero) e aumenta il tempo di addestramento => valuta se usare una PCA per gestire l'esplosione dimensionale.

Usalo solo se le feature originali sono informative

### TRAIN AND SUBMIT
"""

"""
# Initialize and train the model
print("Training a simple Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate on validation set
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

#feat_cols = [c for c in train_df.columns if c not in ("player_won","battle_id")]

# PCA?
USE_PCA = False
POLY_ENABLED = False# se enabled 77.64% (+/- 0.69%) altrimenti 77.94% (+/- 0.35%)

steps = []
if POLY_ENABLED:
    steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
#standardizza
steps.append(("scaler", StandardScaler()))
if USE_PCA:
    steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))  # ~95% varianza
steps.append(("logreg", LogisticRegression(max_iter=2000, random_state=42)))

pipe = Pipeline(steps)

#kfold cross-validation
kfold = KFold(n_splits=7, shuffle=True, random_state=42)  # 7-fold CV
print("Training Logistic Regression con 7-Fold Cross-Validation...\n")
scores = cross_val_score(pipe, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation accuracies: {np.round(scores, 4)}")
print(f"Mean CV accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")

#Training finale
pipe.fit(X, y)
print("\nFinal model trained on all training data.")
"""
vecchio codice, senza k-fold cross-v

# Train
print("Training Logistic Regression (con scaler{}pca)...".format(" + " if USE_PCA else " senza "))
pipe.fit(X_train, y_train)
print("Model training complete.")

# Valutazione su validation
val_pred = pipe.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc*100:.2f}%")

# Nr componenti PCA usate
if USE_PCA:
    print("Componenti PCA usate:", pipe.named_steps["pca"].n_components_)
"""

"""### SUBMIT"""

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