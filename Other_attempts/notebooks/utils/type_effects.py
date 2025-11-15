import numpy as np

# Tutti i 18 tipi in ordine standard (come nella tabella)
TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic",
    "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

# Matrice di efficacia (attacco → difesa)
# Valori presi direttamente dalla tabella ufficiale Pokémon
TYPE_MATRIX = np.array([
# NOR FIR WAT ELE GRA ICE FIG POI GRO FLY PSY BUG ROC GHO DRA DAR STE FAI
 [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0.5, 0,  1,  1,  0.5, 1],   # NORMAL
 [1,  0.5,0.5,1,  2,  2,  1,  1,  1,  1,  1,  2,  0.5, 1,  1,  1,  2,  0.5],  # FIRE
 [1,  2,  0.5,1,  0.5,1,  1,  1,  2,  1,  1,  1,  2,  1,  1,  1,  1,  1],    # WATER
 [1,  1,  2,  0.5,0.5,1,  1,  1,  0,  2,  1,  1,  1,  1,  1,  1,  1,  1],    # ELECTRIC
 [1,  0.5,2,  1,  0.5,1,  1,  0.5,2,  0.5,1,  0.5,2,  1,  1,  1,  0.5,1],    # GRASS
 [1,  0.5,0.5,1,  2,  0.5,1,  1,  2,  2,  1,  1,  1,  1,  2,  1,  0.5,1],    # ICE
 [2,  1,  1,  1,  1,  2,  1,  0.5,1,  0.5,0.5,0.5,2,  0,  1,  2,  2,  0.5],  # FIGHTING
 [1,  1,  1,  1,  2,  1,  1,  0.5,0.5,1,  1,  1,  0.5,0.5,1,  1,  0,  2],    # POISON
 [1,  2,  1,  2,  0.5,1,  1,  2,  1,  0,  1,  0.5,2,  1,  1,  1,  2,  1],    # GROUND
 [1,  1,  1,  0.5,2,  1,  2,  1,  1,  1,  1,  2,  0.5,1,  1,  1,  0.5,1],    # FLYING
 [1,  1,  1,  1,  1,  1,  2,  2,  1,  1,  0.5,1,  1,  1,  1,  0,  0.5,1],    # PSYCHIC
 [1,  0.5,1,  1,  2,  1,  0.5,0.5,1,  0.5,2,  1,  1,  0.5,1,  2,  0.5,0.5],  # BUG
 [1,  2,  1,  1,  1,  2,  0.5,1,  0.5,2,  1,  2,  1,  1,  1,  1,  0.5,1],    # ROCK
 [0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  1,  1,  2,  1,  0.5,1,  1],    # GHOST
 [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  1,  0.5,0],    # DRAGON
 [1,  1,  1,  1,  1,  1,  0.5,1,  1,  1,  2,  1,  1,  2,  1,  0.5,1,  0.5],  # DARK
 [1,  0.5,0.5,0.5,1,  2,  1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  0.5,2],    # STEEL
 [1,  1,  1,  1,  1,  1,  2,  0.5,1,  1,  1,  1,  1,  1,  2,  2,  0.5,1],    # FAIRY
])

# --- Funzione principale (interfaccia invariata) ---
def type_advantage(team_types: list[str], opp_types: list[str]) -> float:
    """
    Compute the average type advantage multiplier for player1's team
    against opponent's lead types.
    """
    scores = []
    for t1 in team_types:
        for t2 in opp_types:
            if t1.lower() in TYPES and t2.lower() in TYPES:
                i = TYPES.index(t1.lower())
                j = TYPES.index(t2.lower())
                scores.append(TYPE_MATRIX[i, j])
            else:
                scores.append(1.0)
    return np.mean(scores) if scores else 1.0
