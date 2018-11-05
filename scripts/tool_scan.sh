#!/bin/sh
#############################################################################
# Copyright (c) 2016-2018, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Hans Pabst (Intel Corp.)
#############################################################################

PATTERNS="*.c *.cpp *.h *.hpp *.f *.F90 *.fh *.sh *.py *.yml *.txt *.md Makefile"
HERE=$(cd $(dirname $0); pwd -P)

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
HIST=$(echo -e ${HITS} | sort -uk2 | sort -nuk1 | cut -d' ' -f2)
HITS=$(echo -n "${HIST}" | wc -l)

if [ "0" != "${HITS}" ]; then
  echo
  echo "Potentially ${HITS} infected revisions (newer to older):"
  echo "${HIST}" | tac
  exit 1
fi

echo "Successfully Completed."

