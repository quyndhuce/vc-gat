"""
Ontology-compatible positive Horn rules and rule attenuation (Appendix B).
"""

from typing import List, Tuple, Dict, Any

RULE_ATTENUATION: float = 0.90

ONTOLOGY_HORN_RULES = [
    {
        "id": "R1_loc_trans",
        "head": ("?x", "located_in", "?z"),
        "body": [("?x", "located_in", "?y"), ("?y", "located_in", "?z")],
        "weight": 0.90
    },
    {
        "id": "R2_part_loc",
        "head": ("?x", "located_in", "?z"),
        "body": [("?x", "part_of", "?y"), ("?y", "located_in", "?z")],
        "weight": 0.90
    },
    {
        "id": "R3_capital_loc",
        "head": ("?x", "located_in", "?z"),
        "body": [("?x", "located_in", "?y"), ("?y", "capital_of", "?z")],
        "weight": 0.90
    },
    {
        "id": "R4_sym_adjacent",
        "head": ("?y", "adjacent_to", "?x"),
        "body": [("?x", "adjacent_to", "?y")],
        "weight": 0.90
    },
    {
        "id": "R5_sub_arch",
        "head": ("?x", "architectural_style", "?z"),
        "body": [("?x", "part_of", "?y"), ("?y", "architectural_style", "?z")],
        "weight": 0.90
    },
    {
        "id": "R6_dynasty_prop",
        "head": ("?x", "dynasty", "?z"),
        "body": [("?x", "part_of", "?y"), ("?y", "dynasty", "?z")],
        "weight": 0.90
    },
    {
        "id": "R7_dedicated_trans",
        "head": ("?x", "dedicated_to", "?z"),
        "body": [("?x", "part_of", "?y"), ("?y", "dedicated_to", "?z")],
        "weight": 0.90
    },
    {
        "id": "R8_loc_has_part",
        "head": ("?y", "has_part", "?x"),
        "body": [("?x", "part_of", "?y")],
        "weight": 0.90
    },
    {
        "id": "R9_surrounds_sym",
        "head": ("?y", "surrounded_by", "?x"),
        "body": [("?x", "surrounds", "?y")],
        "weight": 0.90
    },
    {
        "id": "R10_monument_type",
        "head": ("?x", "instance_of", "?z"),
        "body": [("?x", "subclass_of", "?y"), ("?y", "instance_of", "?z")],
        "weight": 0.90
    },
    {
        "id": "R11_reign_period",
        "head": ("?x", "historical_period", "?z"),
        "body": [("?x", "dynasty", "?y"), ("?y", "historical_period", "?z")],
        "weight": 0.90
    },
    {
        "id": "R12_heritage_status",
        "head": ("?x", "cultural_status", "?z"),
        "body": [("?x", "part_of", "?y"), ("?y", "cultural_status", "?z")],
        "weight": 0.90
    }
]
