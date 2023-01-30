#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'BF8' 'HF8' 'F16' 'BF16' 'F32' 'F64'; do
  for RED_OP in 0 1; do
    for LD in 'eqld' 'gtld'; do
      for RED_VARS in 0 1 2; do
        for IDX in 0 42; do
          for RED_ROWS in 0 1; do
            for ACC in 0 1; do
              for IDX_TYPE in 0 1; do
                for RECORD_IDX in 0 1; do
                  TPPNAME="none"
                  OUTNAME="unary_reduce_"
                  PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')
                  RED_X=0
                  RED_X2=0

                  # only cpy TPP has low precision compute
                  if [[ ("$IDX" == '42') && ("$PREC" == 'F64') ]]; then
                    continue
                  fi
                  if [[ ("$IDX" == '42') && ("$RED_VARS" != '0') ]]; then
                    continue
                  fi
                  if [[ ("$IDX" == '42') && ("$ACC" != '0') ]]; then
                    continue
                  fi
                  if [[ ("$IDX" == '42') && ("$RED_ROWS" != '0') ]]; then
                    continue
                  fi

                  # max case that don't exists
                  if [[ ("$RED_OP" == '1') && ("$RED_VARS" != '0') ]]; then
                    continue
                  fi
                  if [[ ("$RED_OP" == '1') && ("$ACC" != '0') ]]; then
                    continue
                  fi
                  if [[ ("$RED_OP" == '1') && ("$IDX" == '42') ]]; then
                    if [ "$PREC" == 'F16' ]; then
                      continue
                    fi
                    if [ "$PREC" == 'BF8' ]; then
                      continue
                    fi
                    if [ "$PREC" == 'HF8' ]; then
                      continue
                    fi
                  fi

                  # idx_type and record_idx relevant only for indexed reduce cols
                  if [[ ("$IDX_TYPE" == '1') && ("$IDX" != '42') ]]; then
                    continue
                  fi
                  if [[ ("$RECORD_IDX" == '1') && ("$IDX" != '42') ]]; then
                    continue
                  fi
                  # record idx relevant only for max op
                  if [[ ("$RECORD_IDX" == '1') && ("$RED_OP" != '1') ]]; then
                    continue
                  fi

                  # get TPP name
                  if [ "$RED_OP" == '0' ] ; then
                    TPPNAME="add"
                  elif [ "$RED_OP" == '1' ] ; then
                    TPPNAME="max"
                  else
                    continue
                  fi

                  if [ "$RED_ROWS" == '0' ] ; then
                    TPPNAME=${TPPNAME}_cols
                  elif [ "$RED_ROWS" == '1' ] ; then
                    TPPNAME=${TPPNAME}_rows
                  fi

                  if [ "$IDX" == '42' ] ; then
                    TPPNAME=${TPPNAME}_idx
                    if [ "$IDX_TYPE" == '0' ] ; then
                      TPPNAME=${TPPNAME}_i64
                    else
                      TPPNAME=${TPPNAME}_i32
                    fi
                     TPPNAME=${TPPNAME}_argop${RECORD_IDX}
                  fi

                  if [ "$RED_VARS" == '0' ] ; then
                    RED_X=1
                    TPPNAME=${TPPNAME}_x
                  elif [ "$RED_VARS" == '1' ] ; then
                    RED_X2=1
                    TPPNAME=${TPPNAME}_x2
                  elif [ "$RED_VARS" == '2' ] ;  then
                    RED_X=1
                    RED_X2=1
                    TPPNAME=${TPPNAME}_x_x2
                  fi

                  if [ "$ACC" == '0' ]; then
                    TPPNAME=${TPPNAME}_overwrite
                  else
                    TPPNAME=${TPPNAME}_initacc
                  fi

                  OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

                  # generate script by sed
                  sed "s/PREC=0/PREC=\"${PREC}\"/g" unary_reduce.tpl \
                  | sed "s/REDUCE_X=0/REDUCE_X=${RED_X}/g" \
                  | sed "s/REDUCE_X2=0/REDUCE_X2=${RED_X2}/g" \
                  | sed "s/REDUCE_ROWS=0/REDUCE_ROWS=${RED_ROWS}/g" \
                  | sed "s/REDUCE_OP=0/REDUCE_OP=${RED_OP}/g" \
                  | sed "s/N_IDX=0/N_IDX=${IDX}/g" \
                  | sed "s/USE_ACC=0/USE_ACC=${ACC}/g" \
                  | sed "s/IDX_TYPE=0/IDX_TYPE=${IDX_TYPE}/g" \
                  | sed "s/RECORD_IDX=0/RECORD_IDX=${RECORD_IDX}/g" \
                  | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
                  >${OUTNAME}

                  # for gt we need to touch up the script
                  if [ "$LD" == 'gtld' ] ; then
                    sed "s/+ str(m) + '_' + str(m)/+ '100_100'/g" ${OUTNAME} >${TMPFILE}
                    cp ${TMPFILE} ${OUTNAME}
                  fi

                  chmod 755 ${OUTNAME}
                done
              done
            done
          done
        done
      done
    done
  done
done
