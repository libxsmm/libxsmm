#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
RM=$(which rm)

# start from scratch (range of sizes may have changed)
${RM} -rf ${HERE}/opentuner.db

${HERE}/xgemm_opentuner.py --no-dups $*
