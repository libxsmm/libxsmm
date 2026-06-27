#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

FORMAT=FORMATDESC

# NVFP4 uses a 16-element block, MXFP4/MXBF8 use a 32-element block.
if [ "${FORMAT}" == "nvfp4" ]; then
  BLOCK=16
else
  BLOCK=32
fi

for LD in 'eqld' 'gtld'; do
  OUTNAME="${HERE}/unary_quant_${FORMAT}_${LD}.sh"

  # generate script by sed
  sed "s/FORMATVAL/${FORMAT}/g" ${HERE}/unary_quant_fp.tpl \
  | sed "s/BLOCKVAL/${BLOCK}/g" \
  | sed "s/LDMODEVAL/${LD}/g" \
  | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
  >${OUTNAME}

  chmod 755 ${OUTNAME}
done
