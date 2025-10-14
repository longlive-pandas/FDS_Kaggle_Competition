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
