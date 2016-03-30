#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
FIND=$(which find)
EXT=png

${FIND} ${HERE} -name \*.${EXT} -type f -exec mogrify -trim -transparent-color white {} \;

