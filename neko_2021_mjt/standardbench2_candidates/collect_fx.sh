 RAT=localhost
 for i in $(ls */jtrmodels/*E1* | xargs dirname | uniq);
 do
    BNAME=$(dirname $i)
    rm $i/*_I*;
    rsync -ravz -e 'ssh -p 19998' ${BNAME} lasercat@${RAT}:/run/media/lasercat/shared/cvpr22candidates/${HOSTNAME}/${BNAME};
 done;
