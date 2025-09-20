"""
Shared utility functions for the VIEWS Challenge application.
"""

import json
import os
from typing import Optional


def decode_country(country_id: int) -> Optional[str]:
    """Returns country name based on M49 ID"""
    try:
        # Path to country mapping file (in api directory)
        country_list_path = os.path.join(
            os.path.dirname(__file__), "api", "m49-list.json"
        )
        with open(country_list_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            countries_list = data["countries"]
            for country in countries_list:
                if int(country["m49code"]) == country_id:
                    return country["name"]
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return None
