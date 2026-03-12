import random
import ast
from pathlib import Path
import shutil
import shutil

import matplotlib.pyplot as plt

import pandas as pd
import tqdm

import blockies
from blockies.scene_parameters import SceneParameters
import utils

def load_scene_parameters(
        param_dicts: list
    ) -> list[SceneParameters]:
    """
    Load scene parameters from a list of parameter dictionaries.
    """
    params = [SceneParameters.load(p) for p in param_dicts]
    return params

def convert_old_params_df(
        params_df: pd.DataFrame,
        resolution: tuple[int, int]
    ) -> pd.DataFrame:
    """
    Convert a DataFrame of old two4two parameters to a the new blockies format

    Args:
        params_df: DataFrame containing the old parameters

    Returns:
        Dataframe containing the parameters in the new blockies format
    """
 
    print("Converting DataFrame to new blockies format...")
    df = params_df.copy().rename(
        columns={
            'spherical': 'main_spherical',
            'ill_spherical': 'sec_spherical',
        }
    )

    def get_secbones(num_bones):
        if num_bones == 1:
            return random.choice(['001', '010', '100'])
        elif num_bones == 2:
            return random.choice(['011', '101', '110'])
        elif num_bones == 3:
            return '111'
        else:
            raise ValueError(f"Invalid number of bones: {num_bones}")
        
    def get_rgba(color_int, cmap='coolwarm'):
        cmap = plt.get_cmap(cmap)
        return cmap(color_int)

    def process_ill_chars(ill_chars):
        for c in ['med_bend', 'med_sphere_diff', 'mutation_color']:
            ill_chars = [x for x in ill_chars if x != c]
        return ill_chars
    
    df['ill_chars'] = df['ill_chars'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
    )
    
    df = (
        df.assign(
            __module__=lambda x: x['__module__'].str.replace('two4two', 'blockies'),
            obj_name=lambda x: x["ill"].map({1: "ocd", 0: "healthy"}),
            ill_chars=lambda x: x["ill_chars"].apply(process_ill_chars),
            sec_bones=lambda x: x["num_diff"].apply(get_secbones),
            obj_color_rgba=lambda x: x["obj_color"].apply(get_rgba),
            bg_color_rgba=lambda x: x["bg_color"].apply(get_rgba),
            resolution=lambda x: [resolution] * len(x)
        )
        .assign(
            num_ill_chars=lambda x: x["ill_chars"].apply(len),
            label = df["obj_name"]
        )
        .drop(columns=['Unnamed: 0', 'filename', 'ill', 'num_diff', 'sphere_diff', 'pred'])
    )
    return df



def generate_blockies(
        csv_path: Path, 
        output_path: Path,
        convert: bool,
        resolution: tuple[int, int] = (256, 256)
    ):
    """
    Generates a set of blockes from a Blockies CSV by first converting
    to JSON format and then rendering the scenes using the blockies library.

    Args:
        csv_path: Path to the input CSV file containing the parameters
        output_path: Path to the directory where the rendered blockies will be saved
    """

    df = pd.read_csv(csv_path)
    df = convert_old_params_df(df, resolution) if convert else df
    params = df.to_dict(orient='records')
    params = load_scene_parameters(params)

    for _ in tqdm.tqdm(blockies.render(
        params,
        n_processes=4,
        output_dir=output_path,
        blender_dir=None,
        download_blender=False,
        print_output=False,
        print_cmd=False,
    ), total=len(params)):
        pass


def main():

    samples_url = 'https://www.hcxai-group.de/share/study_samples_study1.zip'
    datapath = Path('tmp/study_samples_study1')
    datapath.mkdir(parents=True, exist_ok=True)

    outpath = Path('tmp/xai_samples_converted')
    if outpath.exists():
        shutil.rmtree(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    samplepath = utils.download_file(samples_url, 
                                     'study_samples_study1.zip', 
                                     cache_dir=datapath, 
                                     extract=True, 
                                     force_download=False, 
                                     archive_folder="xai_samples")

    csv_path = samplepath / 'xai_samples_df.csv'
    generate_blockies(csv_path, outpath, convert=True, resolution=(256, 256))

if __name__ == "__main__":    
    main()