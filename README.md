# Blockies: An application-grounded framework for assessing healthy trust in high-stakes human-AI collaboration

## Introduction
**Blockies** is a parametric dataset generator to create images for simulated diagnostic tasks that can be used for assessing human-AI collaboration tools. It extends the approach of the [Two4Two library](https://github.com/mschuessler/two4two), which uses arm position as the only class discriminator, to include multiple customizable traits that can be used as symptoms for the diagnosis of an illness in Blockies called *OCDegen*. This affords human-ai collaboration researchers a more challenging task for high-stakes decision-making.

More details are forthcoming. In the meantime, please review our paper for more information and cite it if you find it helpful.
```
 @article{Johnson_2025,
    title={Higher Stakes, Healthier Trust? An Application-Grounded Approach to Assessing Healthy Trust in High-Stakes Human-AI Collaboration},
    author={Johnson, David S.}, 
    note={arXiv:2503.03529 [cs]},
    number={arXiv:2503.03529}, 
    year={2025}, 
    url={http://arxiv.org/abs/2503.03529},
    DOI={10.48550/arXiv.2503.03529}, 
}

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

Download this repository:

```git
git clone https://github.com/davidsjohnson/blockies-haic.git

```

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

## IJCAI25 - Submission
Jupyter Notebooks to replicate and review the dataset and model used in the IJCAI submission are available in the folder `ijcai25`.  The notebooks also act as examples for working with an existing **Blockies** dataset.  

You can also generate the dataset yourself using the config file: `ijcai25_blockies.toml`.
