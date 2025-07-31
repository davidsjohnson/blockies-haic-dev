"""setup module of blockies."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blockies",
    version="0.0.1",
    author="Anonymous",
    author_email="anonymous@domain.com",
    description="Generate diagnostic based dataset for application-grounded assessment of human-ai collaboration.",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anonymous.com",
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.sh"],
        "blockies": ["py.typed"],
    },
    entry_points={
        'console_scripts': [
            'blockies_render_dataset=blockies.cli_tool:render_dataset',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    install_requires=[
        'imageio',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'toml',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'flake8',
            'flake8-annotations',
            'flake8-docstrings',
            'flake8-import-order',
            'mypy',
            'pdoc',
            'pytest',
            'pytest-cov',
            'torch',
            'torchvision',
        ],
        'example_notebooks_data_generation': [
            'numpy',
            'pandas',
            'notebook'
        ],
        'example_notebooks_model_training': [
            'tensorflow',
            'pandas',
            'notebook',
            'livelossplot'
        ]
    },
    python_requires='>=3.10'
)
