# https://stackoverflow.com/questions/33718144/do-arabic-characters-have-different-unicode-code-points-based-on-position-in-str
SHAPING = {
 u'\u0621' : ( u'\uFE80' ) ,
 u'\u0622' : ( u'\uFE81', u'\uFE82' ) ,
 u'\u0623' : ( u'\uFE83', u'\uFE84' ) ,
 u'\u0624' : ( u'\uFE85' , u'\uFE86' ) ,
 u'\u0625' : ( u'\uFE87' , u'\uFE88' ) ,
 u'\u0626' : ( u'\uFE89' , u'\uFE8B' , u'\uFE8C' , u'\uFE8A' ) ,
 u'\u0627' : ( u'\uFE8D' , u'\uFE8E' ) ,
 u'\u0628' : ( u'\uFE8F' , u'\uFE91' , u'\uFE92' , u'\uFE90' ) ,
 u'\u0629' : ( u'\uFE93' , u'\uFE94' ) ,
 u'\u062A' : ( u'\uFE95' , u'\uFE97' , u'\uFE98' , u'\uFE96' ) ,
 u'\u062B' : ( u'\uFE99' , u'\uFE9B' , u'\uFE9C' , u'\uFE9A' ) ,
 u'\u062C' : ( u'\uFE9D' , u'\uFE9F' , u'\uFEA0', u'\uFE9E' ) ,
 u'\u062D' : ( u'\uFEA1' , u'\uFEA3' , u'\uFEA4' , u'\uFEA2' ) ,
 u'\u062E' : ( u'\uFEA5' , u'\uFEA7' , u'\uFEA8' , u'\uFEA6' ) ,
 u'\u062F' : ( u'\uFEA9' , u'\uFEAA' ) ,
 u'\u0630' : ( u'\uFEAB'  , u'\uFEAC' ) ,
 u'\u0631' : ( u'\uFEAD' , u'\uFEAE' ) ,
 u'\u0632' : ( u'\uFEAF'  , u'\uFEB0' ) ,
 u'\u0633' : ( u'\uFEB1' , u'\uFEB3' , u'\uFEB4' , u'\uFEB2' ) ,
 u'\u0634' : ( u'\uFEB5' , u'\uFEB7' , u'\uFEB8' , u'\uFEB6' ) ,
 u'\u0635' : ( u'\uFEB9' , u'\uFEBB' , u'\uFEBC' , u'\uFEBA' ) ,
 u'\u0636' : ( u'\uFEBD' , u'\uFEBF' , u'\uFEC0' , u'\uFEBE' ) ,
 u'\u0637' : ( u'\uFEC1' , u'\uFEC3' , u'\uFEC4' , u'\uFEC2' ) ,
 u'\u0638' : ( u'\uFEC5' , u'\uFEC7' , u'\uFEC8' , u'\uFEC6' ) ,
 u'\u0639' : ( u'\uFEC9' , u'\uFECB' , u'\uFECC' , u'\uFECA' ) ,
 u'\u063A' : ( u'\uFECD' , u'\uFECF' , u'\uFED0', u'\uFECE' ) ,
 u'\u0640' : ( u'\u0640' ) ,
 u'\u0641' : ( u'\uFED1' , u'\uFED3' , u'\uFED4' , u'\uFED2' ) ,
 u'\u0642' : ( u'\uFED5' , u'\uFED7' , u'\uFED8' , u'\uFED6' ) ,
 u'\u0643' : ( u'\uFED9' , u'\uFEDB' , u'\uFEDC' , u'\uFEDA' ) ,
 u'\u0644' : ( u'\uFEDD' , u'\uFEDF' , u'\uFEE0', u'\uFEDE' ) ,
 u'\u0645' : ( u'\uFEE1' , u'\uFEE3' , u'\uFEE4' , u'\uFEE2' ) ,
 u'\u0646' : ( u'\uFEE5' , u'\uFEE7' , u'\uFEE8' , u'\uFEE6' ) ,
 u'\u0647' : ( u'\uFEE9' , u'\uFEEB' , u'\uFEEC' , u'\uFEEA' ) ,
 u'\u0648' : ( u'\uFEED' , u'\uFEEE' ) ,
 u'\u0649' : ( u'\uFEEF' , u'\uFEF0' ) ,
 u'\u064A' : ( u'\uFEF1' , u'\uFEF3' , u'\uFEF4' , u'\uFEF2' )
}
def arabicv1():
    all=[]
    for i in SHAPING:
        all.append(i);
        all+=list(SHAPING[i]);
    return all