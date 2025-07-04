from pathlib import Path
import pandas as pd

def load_dataframe(data_dir: Path, dataset: str) -> pd.DataFrame:

  data_dir = data_dir / dataset
  df = pd.read_json(data_dir / 'parameters.jsonl', lines=True)
  df['filename'] = df['id'] + '.png'

  return df