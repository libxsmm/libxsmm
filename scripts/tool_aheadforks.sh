#!/bin/bash

REPO=hfp
PROJECT=libxsmm
AHEADMIN=$1

if [ "" = "${AHEADMIN}" ] || [ "0" = "$((0<AHEADMIN))" ]; then
  AHEADMIN=1
fi

FORKS=$(wget -qO- https://api.github.com/repos/${REPO}/${PROJECT}/forks \
      | sed -n 's/ *\"login\": \"\(..*\)\".*/\1/p')
ABRANCH=$(wget -qO- https://api.github.com/repos/${REPO}/${PROJECT} \
      | sed -n 's/ *\"default_branch\": \"\(..*\)\".*/\1/p')
BBRANCH=${ABRANCH}

AHEAD=0
for FORK in ${FORKS}; do
  AHEADBY=$(wget -qO- https://api.github.com/repos/${REPO}/${PROJECT}/compare/${REPO}:${ABRANCH}...${FORK}:${BBRANCH} \
          | sed -n 's/ *\"ahead_by\": \"\(..*\)\".*/\1/p')
  if [ "0" != "$((AHEADMIN<=AHEADBY))" ]; then
    echo "Fork \"${FORK}\" is ahead by ${AHEADBY} fork(s)."
    AHEAD=$((AHEAD+1))
  fi
done

if [ "0" = "${AHEAD}" ]; then
  echo "None of the forks is ahead."
fi

