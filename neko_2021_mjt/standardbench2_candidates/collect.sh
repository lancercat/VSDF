 RAT=${1}
 for i in $(ls */jtrmodels/*E1* | xargs dirname | uniq);
 do
    BNAME=$(dirname $i)
    rm $i/*_I*;
    rsync -ravz -e 'ssh -p 9000' ${BNAME} lasercat@${RAT}:/home/lasercat/netdata/${HOSTNAME}/${BNAME};
 done;