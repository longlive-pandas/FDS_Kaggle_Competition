# FDS_Kaggle_Competition
A repo for the FDS kaggle competition on binary classifier for the Pokemon dataset

```bash
git clone https://github.com/longlive-pandas/FDS_Kaggle_Competition.git
```


### Utilizzare un personal access token o una chiave ssh per accedere su git (profile settings > dev settings > create a personal access token)

### Per eseguire lo script assicurarsi di avere un ambiente con uv (uv sinc) e eseguirlo da visual code per esempio come jupyter notebook (sotto la cartella notebook c'è il file main.ipynb)

### La lista delle dipendenze è nel file pyproject.toml


```toml 
dependencies = [ "ipykernel>=7.0.0", "ipywidgets>=8.1.7", "pandas>=2.3.3", "scikit-learn>=1.7.2", "tqdm>=4.67.1", ] 
``` 


### Uno schema dei dati di addestramento è mostrato nel file utils/schema.py

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
    BATTLETIMELINE ||--|| POKEMONSTATE : "p1_pokemon_state"
    BATTLETIMELINE ||--o| MOVEDETAILS : "p1_move_details"
    BATTLETIMELINE ||--|| POKEMONSTATE : "p2_pokemon_state"
    BATTLETIMELINE ||--o| MOVEDETAILS : "p2_move_details"

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
