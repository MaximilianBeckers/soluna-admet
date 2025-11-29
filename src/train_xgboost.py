# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: soluna-admet
#     language: python
#     name: python3
# ---

import warnings

# %%
import pandas as pd

import utils

warnings.filterwarnings("ignore")

# %%
# read the training data
df = pd.read_csv("../data/peteani_et_al/protacdb2.0_zinc_chembl_dataset.csv")

# %%
df.columns

# %%
# get desriptors and fingerprints

df, fp_cols = utils.get_fingerprints(df, name_smiles_col="smiles")
df, descriptastorus_cols = utils.get_descriptastorus_properties(
    df, name_smiles_col="smiles"
)
df, rdkit_desc_cols = utils.get_rdkit_properties(df, name_smiles_col="smiles")

# %%
