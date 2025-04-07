import torch

PROTENIX_RNA_ID_2_AIDORNA_RESIDUES = {
    21: "A",
    22: "G",
    23: "C",
    24: "T",
    29: "N",
}

# AIDORNA_RES2ID = {
#     "A": 5,
#     "G": 6,
#     "C": 7,
#     "T": 8,   # Note that AIDO.RNA was trained with U being replaced by T.
#     "U": 9,
#     "N": 10,
# }

def get_rnalm_embeddings(input_feature_dict, rnalm, 
                        protenix_rna_id_2_rnalm_token=PROTENIX_RNA_ID_2_AIDORNA_RESIDUES):
    """
    Convert Protenix restypes to RNALM tokens.
    Args:
        input_feature_dict (dict): Dictionary containing the input features.
        rnalm (object): RNALM object. Default to be AIDO.RNA-650m.
        protenix_rna_id_2_rnalm_token (dict): Mapping from Protenix RNA IDs to RNALM tokens. Default is PROTENIX_RNA_ID_2_AIDORNA_RESIDUES.
    Returns:
        tensor: RNALM embeddings
    """
    protenix_restype = input_feature_dict['restype']
    device = protenix_restype.device
    seq_len = protenix_restype.shape[0]
    protenix_rna_id = torch.nonzero(protenix_restype == 1, as_tuple=False)[:,1].tolist()
    assert len(protenix_rna_id) == seq_len, "Sequence length mismatch"
    # convert to RNALM tokens
    rna_seq = "".join([protenix_rna_id_2_rnalm_token[id] for id in protenix_rna_id])
    # get embedding from RNALM
    rnalm_input = rnalm.transform({"sequences": [rna_seq]})    # add [cls] and [sep] automatically
    rnalm_input = {k: v.to(device) for k, v in rnalm_input.items() if type(v) == torch.Tensor}
    rnalm_embeddings = rnalm(rnalm_input)[:, 1:-1, :].squeeze()          # [N_token, c_rnalm]
    return rnalm_embeddings