from __future__ import annotations

import os
import sys

import numpy as np
import sascorer
import torch
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, RDConfig
from rdkit.ML.Descriptors import MoleculeDescriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

torch.pi = torch.tensor(3.141592653589793)

if torch.cuda.is_available():
    print("use GPU")
    device = "cuda"
else:
    print("use CPU")
    device = "cpu"

# ****************************************************************************
# ********************* get descriptastorus descriptors **********************
# ****************************************************************************


def get_descriptastorus_properties(df, name_smiles_col="Structure"):
    # ----------------------------------------------------
    # ------------- ----- descriptastorus ----------------
    # ----------------------------------------------------
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    cdf_norm_cols = [i[0] + "_cdf_norm" for i in generator.columns]

    def rdkit_2d_normalized_features(smiles: str):
        # n.b. the first element is true/false if the descriptors were
        # properly computed
        results = generator.process(smiles)
        _, features = results[0], results[1:]
        return features

    smiles = df[name_smiles_col].to_numpy()
    num_compounds = smiles.size
    inds = np.arange(num_compounds)
    unique_smis, rec_inds = np.unique(smiles, return_inverse=True)

    tmp_data_arr = np.zeros((num_compounds, len(cdf_norm_cols)))

    for count, tmp_smi in enumerate(unique_smis):
        if count % 10000 == 0:
            print("Analyzing compound " + repr(count))

        tmp_inds = inds[rec_inds == count]
        try:
            tmp_props = rdkit_2d_normalized_features(tmp_smi)
            tmp_data_arr[tmp_inds, :] = np.array(tmp_props)
        except Exception:
            tmp_data_arr[tmp_inds, :] = np.nan

    # append the columns to the original dataframe
    for tmp_descriptor_ind in range(len(cdf_norm_cols)):
        df[cdf_norm_cols[tmp_descriptor_ind]] = tmp_data_arr[:, tmp_descriptor_ind]

    return df, cdf_norm_cols


# ****************************************************************************
# ************************** get RDKit descriptors ***************************
# ****************************************************************************


def get_rdkit_properties(df, name_smiles_col="Structure"):
    # calculate the features
    smiles = df[name_smiles_col].to_list()
    num_compounds = len(smiles)

    des_list = [x[0] for x in Descriptors._descList]
    des_list = des_list + ["SA"]
    destype_list = ["RDKit descriptors"] * len(des_list)

    smiles = df[name_smiles_col].to_numpy()
    num_compounds = smiles.size
    inds = np.arange(num_compounds)
    unique_smis, rec_inds = np.unique(smiles, return_inverse=True)

    tmp_data_arr = np.zeros((num_compounds, len(des_list))) * np.nan

    for count, tmp_smi in enumerate(unique_smis):
        if count % 10000 == 0:
            print("Analyzing compound " + repr(count))

        tmp_inds = inds[rec_inds == count]
        try:
            tmp_mol = Chem.MolFromSmiles(tmp_smi)
        except Exception:
            continue

        for tmp_descriptor_ind in range(len(des_list)):
            tmp_descriptor = des_list[tmp_descriptor_ind]

            if tmp_descriptor == "QED":
                tmp_descriptor_val = Chem.QED.qed(tmp_mol)
            elif tmp_descriptor == "SA":
                tmp_descriptor_val = sascorer.calculateScore(tmp_mol)
            else:
                calc = MoleculeDescriptors.MolecularDescriptorCalculator(
                    [
                        tmp_descriptor,
                    ]
                )
                descriptors = calc.CalcDescriptors(tmp_mol)
                tmp_descriptor_val = descriptors[0]

            tmp_data_arr[tmp_inds, tmp_descriptor_ind] = tmp_descriptor_val

    # append the columns to the original dataframe
    for tmp_descriptor_ind in range(len(des_list)):
        df[des_list[tmp_descriptor_ind]] = tmp_data_arr[:, tmp_descriptor_ind]

    # now norm all descriptors with respect to the number of atoms
    for tmp_descriptor in des_list:
        tmp_data = df[tmp_descriptor].to_numpy()

        tmp_data = 1000 * tmp_data / df["HeavyAtomCount"].to_numpy()
        df[tmp_descriptor + "/1000 HeavyAtoms"] = tmp_data

        des_list = des_list + [tmp_descriptor + "/1000 HeavyAtoms"]
        destype_list = destype_list + ["RDKit descriptors/1000 atoms"]

    return df, des_list


# ****************************************************************************
# ***************************** get fingerprints *****************************
# ****************************************************************************


def get_fingerprints(df, name_smiles_col="Structure"):
    # get fingerprints
    num_bits = 1024

    # get list of fingerprints from smiles
    fp_list = []
    for count, tmp_smi in enumerate(df[name_smiles_col].to_list()):
        if count % 10000 == 0:
            print("Analyzing compound " + repr(count))

        tmp_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(tmp_smi),
            2,
            nBits=num_bits,
        )
        fp_list.append(tmp_fp)

    # make array of fingerprints
    num_fp = len(fp_list)
    fp_array = np.zeros((num_fp, num_bits), dtype=np.int8)

    for tmp_fp in range(num_fp):
        tmp_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp_list[tmp_fp], tmp_array)
        fp_array[tmp_fp, :] = tmp_array

    # append them to dataframe
    fingerprint_cols = ["fingerprint_" + str(i) for i in range(fp_array.shape[1])]
    for i, tmp_col in enumerate(fingerprint_cols):
        df[tmp_col] = fp_array[:, i]

    return df, fingerprint_cols


# **************************************
# *** some pre-defined feature lists ***
# **************************************
descriptastorus_features = [
    "BalabanJ_cdf_norm",
    "BertzCT_cdf_norm",
    "Chi0_cdf_norm",
    "Chi0n_cdf_norm",
    "Chi0v_cdf_norm",
    "Chi1_cdf_norm",
    "Chi1n_cdf_norm",
    "Chi1v_cdf_norm",
    "Chi2n_cdf_norm",
    "Chi2v_cdf_norm",
    "Chi3n_cdf_norm",
    "Chi3v_cdf_norm",
    "Chi4n_cdf_norm",
    "Chi4v_cdf_norm",
    "EState_VSA1_cdf_norm",
    "EState_VSA10_cdf_norm",
    "EState_VSA11_cdf_norm",
    "EState_VSA2_cdf_norm",
    "EState_VSA3_cdf_norm",
    "EState_VSA4_cdf_norm",
    "EState_VSA5_cdf_norm",
    "EState_VSA6_cdf_norm",
    "EState_VSA7_cdf_norm",
    "EState_VSA8_cdf_norm",
    "EState_VSA9_cdf_norm",
    "ExactMolWt_cdf_norm",
    "FpDensityMorgan1_cdf_norm",
    "FpDensityMorgan2_cdf_norm",
    "FpDensityMorgan3_cdf_norm",
    "FractionCSP3_cdf_norm",
    "HallKierAlpha_cdf_norm",
    "HeavyAtomCount_cdf_norm",
    "HeavyAtomMolWt_cdf_norm",
    "Ipc_cdf_norm",
    "Kappa1_cdf_norm",
    "Kappa2_cdf_norm",
    "Kappa3_cdf_norm",
    "LabuteASA_cdf_norm",
    "MaxAbsEStateIndex_cdf_norm",
    "MaxAbsPartialCharge_cdf_norm",
    "MaxEStateIndex_cdf_norm",
    "MaxPartialCharge_cdf_norm",
    "MinAbsEStateIndex_cdf_norm",
    "MinAbsPartialCharge_cdf_norm",
    "MinEStateIndex_cdf_norm",
    "MinPartialCharge_cdf_norm",
    "MolLogP_cdf_norm",
    "MolMR_cdf_norm",
    "MolWt_cdf_norm",
    "NHOHCount_cdf_norm",
    "NOCount_cdf_norm",
    "NumAliphaticCarbocycles_cdf_norm",
    "NumAliphaticHeterocycles_cdf_norm",
    "NumAliphaticRings_cdf_norm",
    "NumAromaticCarbocycles_cdf_norm",
    "NumAromaticHeterocycles_cdf_norm",
    "NumAromaticRings_cdf_norm",
    "NumHAcceptors_cdf_norm",
    "NumHDonors_cdf_norm",
    "NumHeteroatoms_cdf_norm",
    "NumRadicalElectrons_cdf_norm",
    "NumRotatableBonds_cdf_norm",
    "NumSaturatedCarbocycles_cdf_norm",
    "NumSaturatedHeterocycles_cdf_norm",
    "NumSaturatedRings_cdf_norm",
    "NumValenceElectrons_cdf_norm",
    "PEOE_VSA1_cdf_norm",
    "PEOE_VSA10_cdf_norm",
    "PEOE_VSA11_cdf_norm",
    "PEOE_VSA12_cdf_norm",
    "PEOE_VSA13_cdf_norm",
    "PEOE_VSA14_cdf_norm",
    "PEOE_VSA2_cdf_norm",
    "PEOE_VSA3_cdf_norm",
    "PEOE_VSA4_cdf_norm",
    "PEOE_VSA5_cdf_norm",
    "PEOE_VSA6_cdf_norm",
    "PEOE_VSA7_cdf_norm",
    "PEOE_VSA8_cdf_norm",
    "PEOE_VSA9_cdf_norm",
    "RingCount_cdf_norm",
    "SMR_VSA1_cdf_norm",
    "SMR_VSA10_cdf_norm",
    "SMR_VSA2_cdf_norm",
    "SMR_VSA3_cdf_norm",
    "SMR_VSA4_cdf_norm",
    "SMR_VSA5_cdf_norm",
    "SMR_VSA6_cdf_norm",
    "SMR_VSA7_cdf_norm",
    "SMR_VSA8_cdf_norm",
    "SMR_VSA9_cdf_norm",
    "SlogP_VSA1_cdf_norm",
    "SlogP_VSA10_cdf_norm",
    "SlogP_VSA11_cdf_norm",
    "SlogP_VSA12_cdf_norm",
    "SlogP_VSA2_cdf_norm",
    "SlogP_VSA3_cdf_norm",
    "SlogP_VSA4_cdf_norm",
    "SlogP_VSA5_cdf_norm",
    "SlogP_VSA6_cdf_norm",
    "SlogP_VSA7_cdf_norm",
    "SlogP_VSA8_cdf_norm",
    "SlogP_VSA9_cdf_norm",
    "TPSA_cdf_norm",
    "VSA_EState1_cdf_norm",
    "VSA_EState10_cdf_norm",
    "VSA_EState2_cdf_norm",
    "VSA_EState3_cdf_norm",
    "VSA_EState4_cdf_norm",
    "VSA_EState5_cdf_norm",
    "VSA_EState6_cdf_norm",
    "VSA_EState7_cdf_norm",
    "VSA_EState8_cdf_norm",
    "VSA_EState9_cdf_norm",
    "fr_Al_COO_cdf_norm",
    "fr_Al_OH_cdf_norm",
    "fr_Al_OH_noTert_cdf_norm",
    "fr_ArN_cdf_norm",
    "fr_Ar_COO_cdf_norm",
    "fr_Ar_N_cdf_norm",
    "fr_Ar_NH_cdf_norm",
    "fr_Ar_OH_cdf_norm",
    "fr_COO_cdf_norm",
    "fr_COO2_cdf_norm",
    "fr_C_O_cdf_norm",
    "fr_C_O_noCOO_cdf_norm",
    "fr_C_S_cdf_norm",
    "fr_HOCCN_cdf_norm",
    "fr_Imine_cdf_norm",
    "fr_NH0_cdf_norm",
    "fr_NH1_cdf_norm",
    "fr_NH2_cdf_norm",
    "fr_N_O_cdf_norm",
    "fr_Ndealkylation1_cdf_norm",
    "fr_Ndealkylation2_cdf_norm",
    "fr_Nhpyrrole_cdf_norm",
    "fr_SH_cdf_norm",
    "fr_aldehyde_cdf_norm",
    "fr_alkyl_carbamate_cdf_norm",
    "fr_alkyl_halide_cdf_norm",
    "fr_allylic_oxid_cdf_norm",
    "fr_amide_cdf_norm",
    "fr_amidine_cdf_norm",
    "fr_aniline_cdf_norm",
    "fr_aryl_methyl_cdf_norm",
    "fr_azide_cdf_norm",
    "fr_azo_cdf_norm",
    "fr_barbitur_cdf_norm",
    "fr_benzene_cdf_norm",
    "fr_benzodiazepine_cdf_norm",
    "fr_bicyclic_cdf_norm",
    "fr_diazo_cdf_norm",
    "fr_dihydropyridine_cdf_norm",
    "fr_epoxide_cdf_norm",
    "fr_ester_cdf_norm",
    "fr_ether_cdf_norm",
    "fr_furan_cdf_norm",
    "fr_guanido_cdf_norm",
    "fr_halogen_cdf_norm",
    "fr_hdrzine_cdf_norm",
    "fr_hdrzone_cdf_norm",
    "fr_imidazole_cdf_norm",
    "fr_imide_cdf_norm",
    "fr_isocyan_cdf_norm",
    "fr_isothiocyan_cdf_norm",
    "fr_ketone_cdf_norm",
    "fr_ketone_Topliss_cdf_norm",
    "fr_lactam_cdf_norm",
    "fr_lactone_cdf_norm",
    "fr_methoxy_cdf_norm",
    "fr_morpholine_cdf_norm",
    "fr_nitrile_cdf_norm",
    "fr_nitro_cdf_norm",
    "fr_nitro_arom_cdf_norm",
    "fr_nitro_arom_nonortho_cdf_norm",
    "fr_nitroso_cdf_norm",
    "fr_oxazole_cdf_norm",
    "fr_oxime_cdf_norm",
    "fr_para_hydroxylation_cdf_norm",
    "fr_phenol_cdf_norm",
    "fr_phenol_noOrthoHbond_cdf_norm",
    "fr_phos_acid_cdf_norm",
    "fr_phos_ester_cdf_norm",
    "fr_piperdine_cdf_norm",
    "fr_piperzine_cdf_norm",
    "fr_priamide_cdf_norm",
    "fr_prisulfonamd_cdf_norm",
    "fr_pyridine_cdf_norm",
    "fr_quatN_cdf_norm",
    "fr_sulfide_cdf_norm",
    "fr_sulfonamd_cdf_norm",
    "fr_sulfone_cdf_norm",
    "fr_term_acetylene_cdf_norm",
    "fr_tetrazole_cdf_norm",
    "fr_thiazole_cdf_norm",
    "fr_thiocyan_cdf_norm",
    "fr_thiophene_cdf_norm",
    "fr_unbrch_alkane_cdf_norm",
    "fr_urea_cdf_norm",
    "qed_cdf_norm",
]

# targets for c-t modelling
targets_po = ["ka_po", "Cl_po", "Vc_po", "Q1_po", "Vp1_po", "Q2_po", "Vp2_po"]
targets_iv = ["CL_iv", "Vc_iv", "Q1_iv", "Vp1_iv", "Q2_iv", "Vp2_iv"]
targets_combined = targets_po + targets_iv
