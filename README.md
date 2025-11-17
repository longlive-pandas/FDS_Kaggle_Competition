
# FDS_Kaggle_Competition

A repository for the **FDS Kaggle competition**, focused on building a **binary classifier** for the Pokémon dataset.

Clone the repository with:

```bash
git clone https://github.com/longlive-pandas/FDS_Kaggle_Competition.git
```

---

## Accessing GitHub

If the repository is private, you must authenticate using either:

- a **Personal Access Token**  
  (GitHub → Profile → Settings → Developer Settings → Fine-grained Tokens)
- **SSH keys** configured on your machine

---

## Running the Code

This project uses **uv** for environment and dependency management.

To set up the environment:

```bash
uv sync
```

How to run it:

1. Open the root folder in **VS Code**
2. Run `final_consegna.py`

Alternatively you can run it on a terminal/bash window as `python final_consegna.py`

---

## Dependencies (from `pyproject.toml`)

```toml
dependencies = [
    "autofeat>=2.1.3",
    "ipykernel>=7.0.0",
    "ipywidgets>=8.1.7",
    "nbdime>=4.0.2",
    "pandas>=2.3.3",
    "pyarrow>=22.0.0",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
    "tqdm>=4.67.1",
    "xgboost>=3.1.1",
]
```

---

## Training Data Schema

The structure of the training data is described in `utils/schema.py`.

Below is a Mermaid ER diagram representing the full dataset:


```mermaid
erDiagram

    ROOT {
        bool player_won
        int battle_id
    }

    ROOT ||--o{ POKEMONDETAIL : "p1_team_details"
    ROOT ||--|| POKEMONDETAIL : "p2_lead_details"

    ROOT ||--o{ BATTLETIMELINE : "battle_timeline"

    BATTLETIMELINE {
        int turn
    }
    
    BATTLETIMELINE |o--|| POKEMONSTATE : "p1_pokemon_state, p2_pokemon_state"
    
    MOVEDETAILS }o--|| BATTLETIMELINE : "p1_move_details,
    p2_move_details"

    POKEMONSTATE {
        string name
        float hp_pct
        string status
        string[] effects
    }
    POKEMONSTATE ||--|| BOOSTS : "boosts"

    BOOSTS {
        int atk
        int def_
        int spa
        int spd
        int spe
    }

    MOVEDETAILS {
        string name
        string type
        string category
        int base_power
        float accuracy
        int priority
    }

    POKEMONDETAIL {
        string name
        int level
        string[] types
        int base_hp
        int base_atk
        int base_def
        int base_spa
        int base_spd
        int base_spe
    }

```

---
