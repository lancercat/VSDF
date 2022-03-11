def temporal_attention_v0( clips, module_dict, length):
    seq = module_dict["seq"];
    features = module_dict["feature_extractor"](clips.cuda())
    features = [f.contiguous() for f in features];

    A = module_dict["CAM"](features)
    out_emb = seq(features[-1], A, length);
    return out_emb, A;

#used in mk7 or later.
def temporal_attention_v2( clips, module_dict, length):
    seq = module_dict["seq"];
    features = module_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];

    A, pred_length = module_dict["TA"](features)
    out_emb = seq(features[-1], A, length);
    return out_emb, A, pred_length;
def temporal_attention_v2dt( clips, module_dict, length):
    seq = module_dict["seq"];
    features = module_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    featuresd = [f.detach() for f in features];
    A, pred_length = module_dict["TA"](featuresd)
    out_emb = seq(features[-1], A, length);
    return out_emb, A, pred_length;