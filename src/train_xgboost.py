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

# %%
import warnings

import numpy as np
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
feature_cols = fp_cols + descriptastorus_cols
target_cols = [
    "pred(rLM LogCLint)",
    "pred(hLM LogCLint)",
    "pred(mLM LogCLint)",
    "pred(minipigLM LogCLint)",
    "pred(cynoLM LogCLint)",
    "pred(dLM LogCLint)",
    "pred(LogFu-Rat)",
    "pred(LogFu-Human)",
    "pred(LogFu-Mouse)",
    "pred(LogFu-Dog)",
    "pred(LogFu-Monkey)",
    "pred(HPLC LogFu HSA)",
    "pred(LogFubrain)",
    "pred(LogFumic)",
    "pred(Direct NIBR LogP)",
    "pred(Direct NIBR LogD7.4)",
    "pred(LE-MDCKv2_LogPapp)",
    "pred(LE-MDCKv1_LogPapp)",
    "pred(Caco-2_LogPapp)",
    "pred(MDCK-MDR1_LogER)",
    "pred(logPAMPA)",
    "pred(logkobs)",
    "pred(CYP3A4_pIC50)",
    "pred(CYP2C9_pIC50)",
    "pred(CYP2D6_pIC50)",
]

df = df.dropna(cols=feature_cols + target_cols)

# %%
# get random split
random_split = np.array(["train"] * df.shape[0])
random_split[int(0.8 * len(random_split)) :] = "val"
random_split[int(0.9 * len(random_split)) :] = "test"
random_split.random.shuffle()
df["random_split"] = random_split


# %%

df_train = df[df["random_split"] == "train"]
df_val = df[df["random_split"] == "val"]
df_test = df[df["random_split"] == "test"]

# %%
