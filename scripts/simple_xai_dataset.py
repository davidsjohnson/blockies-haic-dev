import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tqdm

import blockies
from blockies.scene_parameters import SceneParameters

from utils import load_dataframe


# generate sample vectors for a Blocky that modifies a single feature
def generate_sample_vectors(blocky_vector, feature_name, fmin, fmax, num_samples=10):

    feature_params = np.linspace(fmin, fmax, num_samples)
    sample_num = np.arange(1, num_samples+1)

    df = pd.DataFrame([blocky_vector] * num_samples)
    df[feature_name] = feature_params
    df['sample_num'] = sample_num
    df['id'] = df['id'] + '_' + feature_name + '_' + df['sample_num'].astype(str)

    blocky_vector['sample_num'] = 0

    df = pd.concat([pd.DataFrame([blocky_vector]), df], ignore_index=True)

    return df

# generate sample vectors for all blockies
def generate_sample_df(blocky_df, feature_name, fmin, fmax, num_samples=10):
    all_vectors = []
    for idx, blocky_vector in blocky_df.iterrows():
        sample_vectors = generate_sample_vectors(blocky_vector, feature_name, fmin, fmax, num_samples)
        all_vectors.append(sample_vectors)
    
    return pd.concat(all_vectors, ignore_index=True)

def create_simple_bending_dataset(ds_dir: Path, dataset: str, output_path: Path):
    # Load the dataset
    df = load_dataframe(ds_dir, dataset)

    # Generate sample vectors for each blocky
    feature_name = 'bending'
    fmin = 0.0
    fmax = 0.4
    num_samples = 10

    sample_df = generate_sample_df(df, feature_name, fmin, fmax, num_samples)

    # Save the sample vectors to a CSV file
    sample_df = sample_df.drop(columns=['filename', 'sample_num'], errors='ignore')
    sample_df.to_json(output_path, orient='records', lines=True)


def load_scene_parameters(param_file: str):
    """
    Load scene parameters from a JSON file.
    """
    with open(param_file, 'r') as f:
        params = [SceneParameters.load(json.loads(line)) for line in f.readlines()]
    return params

if __name__ == "__main__":
    ds_dir = Path('blockies_datasets/simple/bend_only')
    dataset = 'xai'
    output_path = ds_dir / f'{dataset}_experimental' / 'parameters_experimental.jsonl'
    img_dir = ds_dir / f'{dataset}_experimental' 

    img_dir.mkdir(parents=True, exist_ok=True)

    create_simple_bending_dataset(ds_dir, dataset, output_path)
    print(f"Simple bending dataset created at {output_path}")

    print('Generating dataset images')
    params = load_scene_parameters(output_path)

    for _ in tqdm.tqdm(blockies.render(
        params,
        n_processes=4,
        output_dir=img_dir,
        blender_dir=None,
        download_blender=False,
        print_output=False,
        print_cmd=False,
    ), total=len(params)):
        pass