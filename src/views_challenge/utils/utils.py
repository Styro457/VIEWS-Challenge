"""
Shared utility functions for the VIEWS Challenge application.
"""
import json
from pathlib import Path
from typing import Optional

def _load_countries() -> dict[int, str]:
    """
    Returns a dictionary with the key being the code as an integer and value the country name.
    """
    path = Path(__file__).parent / "m49-list.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return {int(c["m49code"]): c["name"] for c in data["countries"]}

_country_names = _load_countries()

def decode_country(country_id: int) -> Optional[str]:
    """Returns country name based on M49 ID"""
    return _country_names.get(country_id)
