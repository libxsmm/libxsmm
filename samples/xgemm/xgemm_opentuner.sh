#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)

# start each series from scratch
RM=$(which rm)

# Please note that the set of tuned parameters is
# currently POT-related by intention (cache issue).
#
${RM} -rf ${HERE}/opentuner.db
${HERE}/xgemm_opentuner.py --no-dups 256,512,768 $*

${RM} -rf ${HERE}/opentuner.db
${HERE}/xgemm_opentuner.py --no-dups 1024,1280,1536,1792 $*

${RM} -rf ${HERE}/opentuner.db
${HERE}/xgemm_opentuner.py --no-dups 2048,2304,2560,2816 $*

${RM} -rf ${HERE}/opentuner.db
${HERE}/xgemm_opentuner.py --no-dups 3072,3328,3584,3840 $*

${RM} -rf ${HERE}/opentuner.db
${HERE}/xgemm_opentuner.py --no-dups 4096,4352,4608,4864 $*
