"""
https://transform.tools/json-to-typescript
https://www.codeconvert.ai/typescript-to-python-converter?id=6c80c219-0b8d-4d4e-8b02-39142c4c9b15
"""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Boosts:
    atk: int
    def_: int  
    spa: int
    spd: int
    spe: int

@dataclass
class PokemonState:
    name: str
    hp_pct: float
    status: str
    effects: List[str]
    boosts: Boosts

@dataclass
class MoveDetails:
    name: str
    type: str
    category: str
    base_power: int
    accuracy: float
    priority: int


@dataclass
class PokemonDetail:
    name: str
    level: int
    types: List[str]
    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int



@dataclass
class BattleTimeline:
    turn: int
    p1_pokemon_state: PokemonState
    p1_move_details: Optional[MoveDetails] = None
    p2_pokemon_state: PokemonState
    p2_move_details: Optional[MoveDetails] = None

@dataclass
class Root:
    player_won: bool
    p1_team_details: List[PokemonDetail]
    p2_lead_details: PokemonDetail
    battle_timeline: List[BattleTimeline]
    battle_id: int