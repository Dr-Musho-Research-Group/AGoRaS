# Autonomous Generation of Reactions and Species-VAE

## AGoRaS-VAE

## Dependencies

To run this easily you will need [poetry](https://python-poetry.org/docs/). 
If not you can check out the `pyproject.toml` file to see what packages are needed.
But If you do decide to install poetry just follow the directions below.

## Installation

Step 1.

Clone the repository locally

Step 2.

Check if `poetry` is in your `Path`

Step 3.

Create a virtual environment with the necessary dependencies.

To do this all you have to do is run

> poetry install

> poetry shell

from the command line of your choice i.e. `bash`, `cmd`, `powershell`

## Train the model

To train the model and generate new equations run

> poetry run python models\AGoRaS_VAE.py

This will save the model, encoder and generator separately inside the `models` directory. 
This allows for the user to load the generator at any time to continue generating new equations.

Examples of how to use the generator to generate new equations see `generate_equations` function in `src.equation_generation`
