import os
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import tqdm

import blockies
from blockies.scene_parameters import SceneParameters

from utils import load_dataframe


# generate sample vectors for a Blocky that modifies a single feature
def generate_sample_vectors(blocky_vector, feature_name, fmin, fmax, threshold, num_samples=10, id_start=1):

    feature_params = np.linspace(fmin, fmax, num_samples)
    sample_num = np.arange(id_start, id_start+num_samples)

    df = pd.DataFrame([blocky_vector] * num_samples)

    if feature_name != 'sphere_diff':
        # for all other features, we just set the feature value
        df[feature_name] = feature_params

    else:
        # sphere diff depends on main spherical to update the secondary spherical values
        if blocky_vector['main_spherical'] <= 0.5:
            df['sec_spherical'] = df['main_spherical'] + feature_params
        else:
            df['sec_spherical'] = df['main_spherical'] - feature_params

        df['sphere_diff'] = feature_params

        # make sure values are within valid values for the secondary spherical feature
        df['sec_spherical'] = df['sec_spherical'].clip(lower=0, upper=1.1)

    df['sample_num'] = sample_num
    df['id'] = df['id'] + '_' + feature_name + '_' + df['sample_num'].astype(str)

    blocky_vector['sample_num'] = 0

    df.loc[df[feature_name] >= threshold, 'obj_name'] = 'ocd'
    df.loc[df[feature_name] < threshold, 'obj_name'] = 'healthy'

    # we don't need the sphere_diff column any longer
    df = df.drop(columns=['sphere_diff'], errors='ignore')

    if id_start == 1:
        df = pd.concat([pd.DataFrame([blocky_vector]), df], ignore_index=True)

    return df

# generate sample vectors for all blockies
def generate_sample_df(blocky_df, feature_name, fmin, fmax, threshold, num_samples=10, id_start=1):
    all_vectors = []
    for idx, blocky_vector in blocky_df.iterrows():
        sample_vectors = generate_sample_vectors(blocky_vector, feature_name, fmin, fmax, threshold, num_samples, id_start)
        all_vectors.append(sample_vectors)
    
    return pd.concat(all_vectors, ignore_index=True)

def create_progressive_trait_dataset(feature_name: str, 
                                     feature_range: Union[tuple, list],
                                     threshold: float,
                                     ds_dir: Path, 
                                     dataset: str, 
                                     output_path: Path,
                                     num_samples:int =10):
    # Load the dataset
    df = load_dataframe(ds_dir, dataset)
    print(len(df), "blockies loaded")

    if type(feature_range) is tuple:
        feature_range = [feature_range]

    sample_dfs = []
    id_start = 1
    for r in feature_range:

        # Generate sample vectors for each blocky
        fmin = r[0]
        fmax = r[1]

        n = num_samples // len(feature_range)
        sample_dfs.append(generate_sample_df(df, feature_name, fmin, fmax, threshold, n, id_start))

        id_start += n

    # Concatenate all sample dataframes
    sample_df = pd.concat(sample_dfs, ignore_index=True)

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

    # trait_names = ['mutation_only', 'spherediff_only', 'stretchy_only'] # 'bend_only' already exists
    # trait_ranges = {
    #     'main_spherical': [(0.5, 1.0), (1.1, 1.25)],
    #     'sphere_diff': [(0.05, 0.3), (0.5, 0.75)],
    #     'arm_position': (0.0, 1.0),
    # }
    # ill_threshholds = {
    #     'mutation_only': 1.1,
    #     'spherediff_only': 0.5,
    #     'stretchy_only': 0.500001,
    # }


    trait_names = ['spherediff_only'] # 'bend_only' already exists
    trait_ranges = {
        'sphere_diff': [(0.05, 0.3), (0.5, 0.75)],
    }
    ill_threshholds = {
        'spherediff_only': 0.5,
    }

    for name, t, (feature, frange) in zip(trait_names, ill_threshholds.values(), trait_ranges.items()):
        ds_dir = Path(f'blockies_datasets/simple/{name}')
        dataset = 'xai'
        output_path = ds_dir / f'{dataset}_experimental' / 'parameters_experimental.jsonl'
        img_dir = ds_dir / f'{dataset}_experimental'

        img_dir.mkdir(parents=True, exist_ok=True)

        create_progressive_trait_dataset(feature, 
                                         frange,
                                         t,
                                         ds_dir, 
                                         dataset, 
                                         output_path)
        print(f"Simple {name} dataset created at {output_path}")


        print(f'Generating dataset images for {name} dataset')
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