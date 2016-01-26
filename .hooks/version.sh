#!/bin/sh

ECHO=$(which echo)
CUT=$(which cut)
GIT=$(which git)

NAME=$(${GIT} name-rev --name-only HEAD)
MAIN=$(${GIT} describe --tags --abbrev=0)
REVC=$(${GIT} describe --tags | ${CUT} -d- -f2)

${ECHO} ${NAME}-${MAIN}-${REVC}
