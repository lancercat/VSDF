import glob
import os.path
import shutil
from neko_sdk.root import find_export_root
def testready(argv,modcfg,temeta,itr_override=None,miter=10000,rot=0,auf=True,maxT_overrider=None):
    from neko_2021_mjt.neko_abstract_jtr import neko_abstract_modular_joint_eval
    from neko_sdk.root import find_data_root;
    if (len(argv) > 2):
        export_path = argv[1];
        if (export_path == "None"):
            export_path = None;
        itk=argv[2];
        root=argv[3];
    else:
        export_path = find_export_root();
        itk="latest";
        root="jtrmodels";
    if(itr_override is not None):
        itk=itr_override;
    if(maxT_overrider is None):
        trainer = neko_abstract_modular_joint_eval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
                temeta=temeta
            ), miter
        );
    else:
        trainer= neko_abstract_modular_joint_eval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
                maxT_overrider
            ), miter
        );
    if not auf:
        import torch
        trainer.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
            torch.ones_like(trainer.modular_dict["pred"].model.UNK_SCR)*-6000000)

    globalcache,mdict=trainer.pretest(0)
    return trainer,globalcache,mdict;

def launchtest(argv,modcfg,itr_override=None,miter=10000,rot=0,auf=True,maxT_overrider=None):
    from neko_2021_mjt.neko_abstract_jtr import neko_abstract_modular_joint_eval
    from neko_sdk.root import find_data_root;
    if (len(argv) > 2):
        export_path = argv[1];
        if (export_path == "None"):
            export_path = None;
        itk=argv[2];
        root=argv[3];
    else:
        export_path = find_export_root();
        itk="latest";
        root="jtrmodels";
    if(itr_override is not None):
        itk=itr_override;
    if(maxT_overrider is None):
        trainer = neko_abstract_modular_joint_eval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
            ), miter
        );
    else:
        trainer= neko_abstract_modular_joint_eval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
                maxT_overrider
            ), miter
        );
    if not auf:
        import torch
        trainer.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
            torch.ones_like(trainer.modular_dict["pred"].model.UNK_SCR)*-6000000)
    trainer.val(9,9,rot);

def launchtest_image(src_path,export_path,itk,root,tskcfg,miter=10000,rot=0,auf=True):
    from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
    shutil.rmtree(export_path,ignore_errors=True);
    os.makedirs(export_path);
    tsk = neko_odan_eval_tasks(root,itk,None,tskcfg,miter);
    if not auf:
        import torch
        tsk.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
            torch.ones_like(tsk.modular_dict["pred"].model.UNK_SCR) * -6000000);
    proto,plabel,tdict,handle=tsk.get_proto_and_handle(0);

    images=glob.glob(os.path.join(src_path,"*.jpg"));
    bnames=[os.path.basename(i) for i in images];
    for i in range(len(images)):
        image_path=images[i];
        gt_path=os.path.join(export_path,bnames[i].replace("jpg","txt"));
        text,_,beams= handle.test_image(image_path,proto,plabel,tdict,h=32,w=100);
        with open(gt_path,"w+") as fp:
            fp.writelines([text[0],str(beams)]);

