# Blockies: An application-grounded framework for assessing healthy trust in high-stakes human-AI collaboration

## Introduction
**Blockies** is a parametric dataset generator to create images for simulated diagnostic tasks that can be used for assessing human-AI collaboration tools. It extends the approach of the [Two4Two library](https://github.com/mschuessler/two4two), which uses arm position as the only class discriminator, to include multiple customizable traits that can be used as symptoms for the diagnosis of an illness in Blockies called *OCDegen*. This affords human-ai collaboration researchers a more challenging task for high-stakes decision-making.


```
To understand the inspiration behind **Blockies**, full details on the original Two4Two dataset can be found in their paper:
```
@inproceedings{
    sixt2022do,
    title={Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset},
    author={Leon Sixt and Martin Schuessler and Oana-Iuliana Popescu and Philipp Wei{\ss} and Tim Landgraf},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=v6s3HVjPerv}
}
```

## Installation
If you want to generate your own data, follow these instructions.
Currently, this project is not available through pip but has to be installed manually.


We suggest creating a python3 or conda environment instead of using your system python.

```
python3 -m venv ~/blockies_enviroment
source ~/blockies_enviroment/bin/activate
```

To install the **minimal installation** blockies-haic package change into the cloned directory and run setuptools.

```
cd blockies-haic
pip install .
```

To install the **installation including all requirements for generating your own training data** run:
```
pip install .[example_notebooks_data_generation]
```

To generate the default dataset on your own use the following commands:
```
blockies_render_dataset config/color_spher_bias.toml
```

## AAAI26 - Submission
The sampler and scene parameters used to generate the dataset for the AAAI26 submission are available in the `blockies/aaai26` directory.

Code for downloading and reviewing the datasets can be found in the `aaai26` folder. A Jupyter notebook is also included to review the selection of the datasets for the user study.
