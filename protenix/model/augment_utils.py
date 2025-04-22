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

aido_rna_650m_config = {
                "model.backbone": "aido_rna_650m",
                "model.backbone.from_scratch": True,
                "model.backbone.config_overwrites": {
                    "add_linear_bias": True,
                    "architectures": [
                        "RNABertForMaskedLM"
                    ],
                    "attention_probs_dropout_prob": 0.0,
                    "hidden_act": "swiglu",
                    "hidden_dropout_prob": 0.0,
                    "hidden_size": 1280,
                    "initializer_range": 0.02,
                    "intermediate_size": 3392,
                    "layer_norm_eps": 1e-05,
                    "max_position_embeddings": 1024,
                    "model_type": "rnabert",
                    "normalization_type": "LayerNorm",
                    "num_attention_heads": 20,
                    "num_hidden_layers": 33,
                    "pad_token_id": 0,
                    "position_embedding_type": "rope",
                    "rotary_percent": 1.0,
                    "seq_len_interpolation_factor": None,
                    "tokenizer_type": "BertWordPieceLowerCase",
                    "torch_dtype": "float32",
                    "transformers_version": "4.38.0.dev0",
                    "type_vocab_size": 2,
                    "use_cache": True,
                    "vocab_size": 16,
                }
            }


aido_rna_1b600m_config = {
                "model.backbone": "aido_rna_1b600m",
                "model.backbone.from_scratch": True,
                "model.backbone.config_overwrites": {
                    "add_linear_bias": True,
                    "architectures": [
                        "RNABertForMaskedLM"
                    ],
                    "attention_probs_dropout_prob": 0.0,
                    "hidden_act": "swiglu",
                    "hidden_dropout_prob": 0.0,
                    "hidden_size": 2048,
                    "initializer_range": 0.02,
                    "intermediate_size": 5440,
                    "layer_norm_eps": 1e-05,
                    "max_position_embeddings": 1024,
                    "model_type": "rnabert",
                    "normalization_type": "LayerNorm",
                    "num_attention_heads": 32,
                    "num_hidden_layers": 32,
                    "pad_token_id": 0,
                    "position_embedding_type": "rope",
                    "rotary_percent": 1.0,
                    "seq_len_interpolation_factor": None,
                    "tokenizer_type": "BertWordPieceLowerCase",
                    "torch_dtype": "float32",
                    "transformers_version": "4.38.0.dev0",
                    "type_vocab_size": 2,
                    "use_cache": True,
                    "vocab_size": 16
                    }
            }



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
    rna_seq = ""
    for id in protenix_rna_id:
        if id in protenix_rna_id_2_rnalm_token:
            rna_seq += protenix_rna_id_2_rnalm_token[id]
        else:
            rna_seq += "N"
    # get embedding from RNALM
    rnalm_input = rnalm.transform({"sequences": [rna_seq]})    # will add [cls] and [sep] automatically
    rnalm_input = {k: v.to(device) for k, v in rnalm_input.items() if type(v) == torch.Tensor}
    rnalm_embeddings = rnalm(rnalm_input)[:, 1:-1, :].squeeze()          # [N_token, c_rnalm]
    return rnalm_embeddings