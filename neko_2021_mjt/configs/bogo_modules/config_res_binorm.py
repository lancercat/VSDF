from neko_2021_mjt.bogo_modules.res45_binorm_bogomod import neko_res45_binorm_bogo;
def config_bogo_resbinorm(container, bogoname):
    return {
        "bogo_mod": neko_res45_binorm_bogo,
        "args":
        {
            "container":container,
            "name":bogoname,
         }
    }