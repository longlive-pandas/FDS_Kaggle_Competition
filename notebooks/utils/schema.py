"""
https://transform.tools/json-to-typescript
https://www.codeconvert.ai/typescript-to-python-converter?id=6c80c219-0b8d-4d4e-8b02-39142c4c9b15
"""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Boosts:
    atk: int
    def_: int  # 'def' is a keyword in Python, so use 'def_' instead
    spa: int
    spd: int
    spe: int

@dataclass
class P1PokemonState:
    name: str
    hp_pct: int
    status: str
    effects: List[str]
    boosts: Boosts

@dataclass
class P1MoveDetails:
    name: str
    type: str
    category: str
    base_power: int
    accuracy: int
    priority: int

@dataclass
class Boosts2:
    atk: int
    def_: int
    spa: int
    spd: int
    spe: int

@dataclass
class P2PokemonState:
    name: str
    hp_pct: int
    status: str
    effects: List[str]
    boosts: Boosts2

@dataclass
class P2MoveDetails:
    name: str
    type: str
    category: str
    base_power: int
    accuracy: int
    priority: int

@dataclass
class P1TeamDetail:
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
class P2LeadDetails:
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
    p1_pokemon_state: P1PokemonState
    p1_move_details: Optional[P1MoveDetails] = None
    p2_pokemon_state: P2PokemonState
    p2_move_details: Optional[P2MoveDetails] = None

@dataclass
class Root:
    player_won: bool
    p1_team_details: List[P1TeamDetail]
    p2_lead_details: P2LeadDetails
    battle_timeline: List[BattleTimeline]
    battle_id: int