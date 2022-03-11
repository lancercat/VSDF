
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
def print_cvt(cs):
    d={};
    with open("chsmap.txt","r") as fp:
        for l in fp:
            terms=l.strip().split(" ");
            if(cs is None):
                d[terms[0]]=terms[1:];
            else:
                if(d[terms[0]] in cs):
                    d[terms[0]] = terms[1:];
        print(d);