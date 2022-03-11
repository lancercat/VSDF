from neko_sdk.ocr_modules.charset.greek_simple import greek_simp_lwr,greek_simp_upr
from neko_2020nocr.dan.tasks.dscs import makept;
from neko_sdk.root import find_data_root

makept(None,[find_data_root()+"/greek/NotoSans-Regular.ttf"],
       find_data_root()+"/dicts/dabgreek.pt",
       set(greek_simp_lwr+greek_simp_upr),{},
       masters=greek_simp_upr,servants=greek_simp_lwr);
