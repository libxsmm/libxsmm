#!/bin/bash
#############################################################################
# Copyright (c) 2016-2019, Intel Corporation                                #
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

GREP=$(command -v grep)
SORT=$(command -v sort)
CUT=$(command -v cut)
TR=$(command -v tr)
WC=$(command -v wc)

if [ "" != "${GREP}" ] && \
   [ "" != "${SORT}" ] && \
   [ "" != "${CUT}" ] && \
   [ "" != "${TR}" ] && \
   [ "" != "${WC}" ];
then
  if [ $(command -v lscpu) ]; then
    NS=$(lscpu | ${GREP} -m1 "Socket(s)" | ${TR} -d " " | ${CUT} -d: -f2)
    if [ "" = "${NS}" ]; then NS=1; fi
    NC=$((NS*$(lscpu | ${GREP} -m1 "Core(s) per socket" | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$((NC*$(lscpu | ${GREP} -m1 "Thread(s) per core" | ${TR} -d " " | ${CUT} -d: -f2)))
  elif [ -e /proc/cpuinfo ]; then
    NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
    if [ "" = "${NS}" ] || [ "" = "${NS}" ]; then NS=1; fi
    NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
  elif [ "Darwin" = "$(uname)" ]; then
    NS=$(sysctl hw.packages | ${CUT} -d: -f2 | tr -d " ")
    NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | tr -d " ")
    NT=$(sysctl hw.logicalcpu | ${CUT} -d: -f2 | tr -d " ")
  fi
  if [ "" != "${NC}" ] && [ "" != "${NT}" ]; then
    HT=$((NT/NC))
  else
    NS=1 NC=1 NT=1 HT=1
  fi
  if [ "" != "$(command -v numactl)" ]; then
    NN=$(numactl -H | ${GREP} available: | ${CUT} -d' ' -f2)
  else
    NN=${NS}
  fi
  if [ "-ns" = "$1" ] || [ "--sockets" = "$1" ]; then
    echo "${NS}"
  elif [ "-nc" = "$1" ] || [ "--cores" = "$1" ]; then
    echo "${NC}"
  elif [ "-nt" = "$1" ] || [ "--threads" = "$1" ]; then
    echo "${NT}"
  elif [ "-ht" = "$1" ] || [ "--smt" = "$1" ]; then
    echo "${HT}"
  elif [ "-nn" = "$1" ] || [ "--numa" = "$1" ]; then
    echo "${NN}"
  elif [ "-h" = "$1" ] || [ "--help" = "$1" ]; then
    echo "$0 [-ns|--sockets] [-nc|--cores] [-nt|--threads] [-ht|--smt] [-nn|--numa]"
  else
    echo -e "sockets\t: ${NS}"
    echo -e "cores\t: ${NC}"
    echo -e "threads\t: ${NT}"
    echo -e "smt\t: ${HT}"
    echo -e "numa:\t: ${NN}"
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

