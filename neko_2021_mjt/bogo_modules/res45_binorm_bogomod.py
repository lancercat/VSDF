class neko_res45_binorm_bogo:
    def cuda(this):
        pass;
    def freeze(this):
        this.container.eval();
    def freezebn(this):
        this.container.model.freezebnprefix(this.bnname);
    def unfreezebn(this):
        this.container.model.unfreezebnprefix(this.bnname);

    def unfreeze(this):
        this.container.model.train();
    def get_torch_module_dict(this):
        return this.container.model;
    def __init__(this,args,mod_dict):
        this.container=mod_dict[args["container"]];
        this.name=args["name"];
        this.bnname=args["name"].replace("res","bn");
        this.model=mod_dict[args["container"]].model.bogo_modules[args["name"]]
    def __call__(this,x):
        return this.model(x);

class neko_res45cco_binorm_bogo:
    def cuda(this):
        pass;
    def freeze(this):
        this.container.eval();
    def freezebn(this):
        this.container.model.freezebnprefix(this.bnname);
    def unfreezebn(this):
        this.container.model.unfreezebnprefix(this.bnname);

    def unfreeze(this):
        this.container.model.train();
    def get_torch_module_dict(this):
        return this.container.model;
    def __init__(this,args,mod_dict):
        this.container=mod_dict[args["container"]];
        this.name=args["name"];
        this.bnname=args["name"].replace("res","bn");
        this.model=mod_dict[args["container"]].model.bogo_modules[args["name"]]
    def __call__(this,x):
        return this.model(x);

