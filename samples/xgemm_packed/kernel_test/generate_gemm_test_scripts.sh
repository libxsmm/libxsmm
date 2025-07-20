#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
  RSAMPLESIZE=8
else
  SAMPLESIZE=${SSIZE}
  RSAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F64_F64_F64_F64' 'F32_F32_F32_F32' ; do
  for LD in 'eqld' 'gtld'; do
    for TRA in 0 1; do
      for TRB in 0 1; do
         MSTART=1
         NSTART=1
         KSTART=1
         RSTART=1
         MSTEP=1
         NSTEP=1
         KSTEP=1
         RSTEP=1

         # TODO: all the "continue" ifs should be handled by allow list outside of this script
         # transpose A or B is right now not supported
         if [[ ( "$TRA" == '1'  ||  "$TRB" == '1') ]]; then
           continue
         fi

         if [ "$PREC" == 'F64_F64_F64_F64' ] ; then
           OUTNAME="dgemm_packed_"
         elif [ "$PREC" == 'F32_F32_F32_F32' ] ; then
           OUTNAME="sgemm_packed_"
         else
           continue
         fi

         if [ "$TRA" == '0' ] ; then
           OUTNAME=$OUTNAME"n"
         else
           OUTNAME=$OUTNAME"t"
         fi

         if [ "$TRB" == '0' ] ; then
           OUTNAME=$OUTNAME"n_"$LD
         else
           OUTNAME=$OUTNAME"t_"$LD
         fi

         OUTNAME=$OUTNAME".slurm"

         #echo "Copying "$TPLNAME" to "$OUTNAME
         sed "s/PREC=0/PREC=\"${PREC}\"/g" ${HERE}/gemm_packed_kernel.tpl \
         | sed "s/TRA=0/TRA=${TRA}/g" \
         | sed "s/TRB=0/TRB=${TRB}/g" \
         | sed "s/MSTART/${MSTART}/g" \
         | sed "s/MSTEP/${MSTEP}/g" \
         | sed "s/NSTART/${NSTART}/g" \
         | sed "s/NSTEP/${NSTEP}/g" \
         | sed "s/KSTART/${KSTART}/g" \
         | sed "s/KSTEP/${KSTEP}/g" \
         | sed "s/KSTART/${KSTART}/g" \
         | sed "s/KSTEP/${KSTEP}/g" \
         | sed "s/RSTART/${RSTART}/g" \
         | sed "s/RSTEP/${RSTEP}/g" \
         | sed "s/RSAMPLESIZE/${RSAMPLESIZE}/g" \
         | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
         >${HERE}/${OUTNAME}

         # for gt we need to touch up the script
         if [ "$LD" == 'gtld' ] ; then
           sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ '100 105 110'/g" ${HERE}/${OUTNAME} >${TMPFILE}
           cp ${TMPFILE} ${HERE}/${OUTNAME}
         fi

         # trB we need to switch LDB
         if [ "$TRA" == '1' ] ; then
           sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(k) + ' ' + str(k) + ' ' + str(m)/g" ${HERE}/${OUTNAME} >${TMPFILE}
           cp ${TMPFILE} ${HERE}/${OUTNAME}
         fi

         # trB we need to switch LDB
         if [ "$TRB" == '1' ] ; then
           sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" ${HERE}/${OUTNAME} >${TMPFILE}
           cp ${TMPFILE} ${HERE}/${OUTNAME}
         fi

         chmod 755 ${HERE}/${OUTNAME}
      done
    done
  done
done
