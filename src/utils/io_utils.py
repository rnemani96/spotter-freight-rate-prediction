import json
import yaml
from pathlib import Path
import pandas as pd
from typing import Union

def load_config(path: Union[str, Path]) -> dict:
    """Load configuration from a YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_json(data: dict, path: Path) -> None:
    """Save dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> dict:
    """Load dictionary from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file."""
    return pd.read_parquet(path)
