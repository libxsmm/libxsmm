#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

PATTERNS="*.c *.cc *.cpp *.cxx *.h *.hpp *.hxx *.f *.F90 *.fh *.sh *.py *.yml *.slurm *.txt *.md Makefile*"
HERE=$(cd "$(dirname "$0")"; pwd -P)

CODEFILE=${HERE}/../.codefile
KEYFILE=${HERE}/../keywords.txt

if [ -e ${CODEFILE} ]; then
  PATTERNS="$(cat ${CODEFILE})"
fi

if [ ! -e ${KEYFILE} ]; then
  >&2 echo "Error: No file ${KEYFILE} found!"
  exit 1
fi

# check for any pending replacement which overlays local view of repository
if [ "" != "$(git replace -l)" ]; then
  >&2 echo "Error: found pending replacements!"
  >&2 echo "Run: \"git replace -l | xargs -i git replace -d {}\" to cleanup"
  >&2 echo "Run: \"git filter-branch -- --all\" to apply (not recommended!)"
  exit 1
fi

for KEYWORD in $(cat ${KEYFILE}); do
  echo "Searching for ${KEYWORD}..."
  # Search all commit messages regardless of the file type
  REVS=$(git log -i --grep=${KEYWORD} --oneline | cut -d' ' -f1)
  for REV in ${REVS}; do
    # Unix timestamp (sort key)
    STM=$(git show -s --format=%ct ${REV})
    # Author information
    WHO=$(git show -s --format=%an ${REV})
    echo "Found ${REV} (${WHO})"
    HITS+="${STM} ${REV}\n"
    LF=true
  done
  # Search the content of the diffs matching the given file types
  for PATTERN in ${PATTERNS}; do
    REVS=$(git log -i -G${KEYWORD} --oneline "${PATTERN}" | cut -d' ' -f1)
    for REV in ${REVS}; do
      # Unix timestamp (sort key)
      STM=$(git show -s --format=%ct ${REV})
      # Author information
      WHO=$(git show -s --format=%an ${REV})
      echo "Found ${REV} (${WHO})"
      HITS+="${STM} ${REV}\n"
      LF=true
    done
  done
  if [ "" != "${LF}" ]; then
    echo; LF=""
  fi
done

# make unique by SHA, sort from older to newer, and drop timestamp (sort key)
HIST=$(echo -e "${HITS}" | sort -uk2 | sort -nuk1 | cut -d' ' -f2)
HITS=$(echo -n "${HIST}" | wc -l | tr -d " ")

if [ "0" != "${HITS}" ]; then
  echo
  echo "Potentially ${HITS} infected revisions (newer to older):"
  echo "${HIST}" | tac
  exit 1
fi

echo "Successfully Completed."

