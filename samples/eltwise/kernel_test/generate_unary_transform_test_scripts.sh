#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

TMPFILE2=$(mktemp)
trap 'rm ${TMPFILE2}' EXIT

for PREC in 'I8' 'I16' 'I32' 'I64' 'BF8' 'HF8' 'BF16' 'F16' 'F32' 'F64'; do
  for TYPE in 'T' 'R' 'S' 'V' 'W' 'Q' 'N' 'M' 'X' 'Y' 'Z' 'B' 'C' 'D'; do
    for LD in 'eqld' 'gtld'; do
      TPPNAME="none"
      OUTNAME="unary_transform_"
      PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')
      MSTART=1
      MSTEP=1

      # only transpose works for higher precision
      if [[ ("$TYPE" != 'T') && (("$PREC" == 'F32') || ("$PREC" == 'I32') || ("$PREC" == 'F64') || ("$PREC" == 'I64')) ]]; then
        continue
      fi

      # some transforms work only for 16bit
      if [[ (("$TYPE" == 'R') || ("$TYPE" == 'C') || ("$TYPE" == 'D') || ("$TYPE" == 'V') || ("$TYPE" == 'B') || ("$TYPE" == 'Q')) && (("$PREC" == 'I8') || ("$PREC" == 'BF8') || ("$PREC" == 'HF8')) ]]; then
        continue
      fi

      # some transforms work only for 8bit
      if [[ (("$TYPE" == 'N') || ("$TYPE" == 'M')) && (("$PREC" == 'I16') || ("$PREC" == 'BF16') || ("$PREC" == 'F16')) ]]; then
        continue
      fi

      # get TPP name
      if [ "$TYPE" == 'T' ] ; then
        TPPNAME="transpose"
      elif [ "$TYPE" == 'R' ] ; then
        TPPNAME="vnni2_to_vnni2T"
        MSTART=2
        MSTEP=2
      elif [ "$TYPE" == 'S' ] ; then
        TPPNAME="vnni4_to_vnni4T"
        MSTART=4
        MSTEP=4
      elif [ "$TYPE" == 'V' ] ; then
        TPPNAME="norm_to_vnni2"
        MSTART=2
        MSTEP=2
      elif [ "$TYPE" == 'W' ] ; then
        TPPNAME="norm_to_vnni4"
        MSTART=4
        MSTEP=4
      elif [ "$TYPE" == 'Q' ] ; then
        TPPNAME="norm_to_vnni4T"
        MSTART=4
        MSTEP=4
      elif [ "$TYPE" == 'B' ] ; then
        TPPNAME="norm_to_vnni2T"
        MSTART=2
        MSTEP=2
      elif [ "$TYPE" == 'C' ] ; then
        TPPNAME="vnni2T_to_norm"
        MSTART=2
        MSTEP=2
      elif [ "$TYPE" == 'D' ] ; then
        TPPNAME="vnni4T_to_norm"
        MSTART=4
        MSTEP=4
      elif [ "$TYPE" == 'N' ] ; then
        TPPNAME="vnni4_to_norm"
        MSTART=4
        MSTEP=4
      elif [ "$TYPE" == 'M' ] ; then
        TPPNAME="vnni4_to_vnni2"
        MSTART=4
        MSTEP=4
      elif [ "$TYPE" == 'X' ] ; then
        TPPNAME="padn"
        MSTART=4
        MSTEP=1
      elif [ "$TYPE" == 'Y' ] ; then
        TPPNAME="padm"
        MSTART=4
        MSTEP=1
      elif [ "$TYPE" == 'Z' ] ; then
        TPPNAME="padnm"
        MSTART=4
        MSTEP=1
      else
        continue
      fi

      OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

      # generate script by sed
      sed "s/PREC=0/PREC=\"${PREC}\"/g" unary_transform.tpl \
      | sed "s/TRANS_OP=0/TRANS_OP=${TYPE}/g" \
      | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
      | sed "s/MSTART/${MSTART}/g" \
      | sed "s/MSTEP/${MSTEP}/g" \
      >${OUTNAME}

      # for gt we need to touch up the script
      if [ "$LD" == 'gtld' ] ; then
        sed "s/+ str(m) + '_' + LDOTPL/+ '100_100'/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      fi

      if [ "$TYPE" == 'T' ] ; then
        sed "s/LDOTPL/str(n)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'R' ] ; then
        sed "s/LDOTPL/str(n)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'S' ] ; then
        sed "s/LDOTPL/str(n)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'V' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'W' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'Q' ] ; then
        sed "s/LDOTPL/str(n)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'B' ] ; then
        sed "s/LDOTPL/str(n)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'C' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} > ${TMPFILE2}
        sed "s/+ str(m) + '_' +/+ str(n) + '_' +/g" ${TMPFILE2} > ${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'D' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} > ${TMPFILE2}
        sed "s/+ str(m) + '_' +/+ str(n) + '_' +/g" ${TMPFILE2} > ${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'N' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'M' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'X' ] ; then
        sed "s/LDOTPL/str(m)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'Y' ] ; then
        sed "s/LDOTPL/str(int((m + 3)\/4)\*4)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      elif [ "$TYPE" == 'Z' ] ; then
        sed "s/LDOTPL/str(int((m + 3)\/4)\*4)/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      else
        continue
      fi

      chmod 755 ${OUTNAME}
    done
  done
done
