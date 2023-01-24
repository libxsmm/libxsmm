#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

for PREC in 'I8' 'I16' 'I32' 'I64' 'BF8' 'HF8' 'BF16' 'F16' 'F32' 'F64'; do
  for TYPE in 'T' 'R' 'S' 'V' 'W' 'Q' 'N' 'M' 'X' 'Y' 'Z'; do
    for LD in 'eqld' 'gtld'; do
      TPPNAME="none"
      OUTNAME="unary_transform_"
      PRECLC=`echo "$PREC" | awk '{print tolower($0)}'`
      MSTART=1
      MSTEP=1

      # only transpose works for higher precision
      if [[ ("$TYPE" != 'T') && (("$PREC" == 'F32') || ("$PREC" == 'I32') || ("$PREC" == 'F64') || ("$PREC" == 'I64')) ]]; then
        continue
      fi

      # some transforms work only for 16bit
      if [[ (("$TYPE" == 'R') || ("$TYPE" == 'V') || ("$TYPE" == 'Q')) && (("$PREC" == 'I8') || ("$PREC" == 'BF8') || ("$PREC" == 'HF8')) ]]; then
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
        sed -i "s/+ str(m) + '_' + LDOTPL/+ '100_100'/g" ${OUTNAME}
      fi

      if [ "$TYPE" == 'T' ] ; then
        sed -i "s/LDOTPL/str(n)/g" ${OUTNAME}
      elif [ "$TYPE" == 'R' ] ; then
        sed -i "s/LDOTPL/str(n)/g" ${OUTNAME}
      elif [ "$TYPE" == 'S' ] ; then
        sed -i "s/LDOTPL/str(n)/g" ${OUTNAME}
      elif [ "$TYPE" == 'V' ] ; then
        sed -i "s/LDOTPL/str(m)/g" ${OUTNAME}
      elif [ "$TYPE" == 'W' ] ; then
        sed -i "s/LDOTPL/str(m)/g" ${OUTNAME}
      elif [ "$TYPE" == 'Q' ] ; then
        sed -i "s/LDOTPL/str(n)/g" ${OUTNAME}
      elif [ "$TYPE" == 'N' ] ; then
        sed -i "s/LDOTPL/str(m)/g" ${OUTNAME}
      elif [ "$TYPE" == 'M' ] ; then
        sed -i "s/LDOTPL/str(m)/g" ${OUTNAME}
      elif [ "$TYPE" == 'X' ] ; then
        sed -i "s/LDOTPL/str(m)/g" ${OUTNAME}
      elif [ "$TYPE" == 'Y' ] ; then
        sed -i "s/LDOTPL/str(int((m + 3)\/4)\*4)/g" ${OUTNAME}
      elif [ "$TYPE" == 'Z' ] ; then
        sed -i "s/LDOTPL/str(int((m + 3)\/4)\*4)/g" ${OUTNAME}
      else
        continue
      fi

      chmod 755 ${OUTNAME}
    done
  done
done
