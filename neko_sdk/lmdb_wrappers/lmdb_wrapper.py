import lmdb
import os
import cv2
class lmdb_wrapper:
    def set_meta(this):
        this.labels = {};
        this.writers = set();

    def __init__(this,lmdb_dir):
        this.root=lmdb_dir;
        os.makedirs(lmdb_dir, exist_ok=True)

        this.db=lmdb.open(lmdb_dir,map_size=1e11);
        this.load=0;
        this.txn=this.db.begin(write=True);


    def adddata_kv(this, ikvdict, tkvdict, rkvdict):
        iks=[];
        rks=[];
        tks=[];
        for ik in ikvdict:
            imageKey = (ik + '-%09d' % this.load).encode();
            iks.append(imageKey);
            this.txn.put(imageKey, cv2.imencode(".png", ikvdict[ik])[1]);
        for rk in rkvdict:
            rawKey = (rk + '-%09d' % this.load).encode();
            rks.append(rawKey);
            this.txn.put(rawKey, rkvdict[rk]);
        for tk in tkvdict:
            tKey = (tk + '-%09d'% this.load).encode() ;
            tks.append(tKey);
            this.txn.put(tKey, tkvdict[tk].encode());
        if (this.load % 500 == 0):
            this.txn.replace('num-samples'.encode(), str(this.load).encode());
            print("load:", this.load);
            this.txn.commit();
            del this.txn;
            this.txn = this.db.begin(write=True);
        this.load += 1;
        return iks,rks,tks;

    def end_this(this):
        this.txn.replace('num-samples'.encode(), str(this.load).encode());
        try:
            this.txn.commit();
            this.db.close();
        except:
            pass;


    def __del__(this):
        this.end_this();


