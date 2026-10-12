import json
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

OPTIONAL_KEYS = {
    "seed",                  # null: seed aleatória a cada execução
    "bpm",                   # null: andamento dos MIDIs de referência
    "cookies",               # null: sem cookies para o YouTube
    "cookies_from_browser",  # null: sem cookies para o YouTube
    "soundfont",             # null: sintetizador aditivo (som mais artificial)
}

def load_defaults() -> Dict[str, Any]:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_defaults(values: Dict[str, Any]) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, ensure_ascii=False)
        f.write("\n")
