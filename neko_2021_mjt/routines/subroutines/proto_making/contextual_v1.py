import torch;

def mk_proto_contextual_v1(label, module_dict):
    normprotos, plabel, gplabels, tdict, gtdict = module_dict["sampler"].model.sample_charset_by_text(label,
                                                                                                      use_sp=False)

    # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
    # cv2.imshow("alphabets",im);
    # print([tdict[label.item()] for label in plabel]);
    # cv2.waitKey(0);
    proto = module_dict["prototyper"](normprotos, use_sp=False);
    fsp = module_dict["semantic_branch"]()[:gtdict["[UNK]"] + 1];
    csp = module_dict["semantic_branch"](gplabels)
    return proto, fsp, csp, plabel, gplabels, tdict, gtdict

def mk_proto_contextual_v1fvb(label, module_dict):
    normprotos, plabel, gplabels, tdict, gtdict = module_dict["sampler"].model.sample_charset_by_text(label,
                                                                                                      use_sp=False)
    module_dict["prototyper"].model.freeze_bb_bn();
    proto = module_dict["prototyper"](normprotos, use_sp=False);
    module_dict["prototyper"].model.unfreeze_bb_bn();
    # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
    # cv2.imshow("alphabets",im);
    # print([tdict[label.item()] for label in plabel]);
    # cv2.waitKey(0);
    fsp = module_dict["semantic_branch"]()[:gtdict["[UNK]"] + 1];
    csp = module_dict["semantic_branch"](gplabels)
    return proto, fsp, csp, plabel, gplabels, tdict, gtdict
