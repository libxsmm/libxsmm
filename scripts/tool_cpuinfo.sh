#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

GREP=$(command -v grep)
SORT=$(command -v sort)
CUT=$(command -v cut)
TR=$(command -v tr)
WC=$(command -v wc)

if [ "${GREP}" ] && [ "${SORT}" ] && [ "${CUT}" ] && [ "${TR}" ] && [ "${WC}" ]; then
  if [ "$(command -v lscpu)" ]; then
    NS=$(lscpu | ${GREP} -m1 "Socket(s)" | ${TR} -d " " | ${CUT} -d: -f2)
    if [ ! "${NS}" ]; then NS=1; fi
    if [[ ${NS} =~ ^[1-9][0-9]*$ ]]; then
      NC=$((NS*$(lscpu | ${GREP} -m1 "Core(s) per socket" | ${TR} -d " " | ${CUT} -d: -f2)))
      NT=$(lscpu | ${GREP} -m1 "CPU(s)" | ${TR} -d " " | ${CUT} -d: -f2)
      if [ ! "${NT}" ]; then
        NT=$((NC*$(lscpu | ${GREP} -m1 "Thread(s) per core" | ${TR} -d " " | ${CUT} -d: -f2)))
      fi
    else
      NS=$(lscpu | ${GREP} -m1 "Cluster(s)" | ${TR} -d " " | ${CUT} -d: -f2)
      NC=$((NS*$(lscpu | ${GREP} -m1 "Core(s) per cluster" | ${TR} -d " " | ${CUT} -d: -f2)))
      NT=$(lscpu | ${GREP} -m1 "CPU(s)" | ${TR} -d " " | ${CUT} -d: -f2)
      if [ ! "${NT}" ]; then
        NT=$((NC*$(lscpu | ${GREP} -m1 "Thread(s) per core" | ${TR} -d " " | ${CUT} -d: -f2)))
      fi
    fi
  elif [ -e /proc/cpuinfo ]; then
    NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
    if [ ! "${NS}" ]; then NS=1; fi
    NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
  elif [ "Darwin" = "$(uname)" ]; then
    NS=$(sysctl hw.packages    | ${CUT} -d: -f2 | ${TR} -d " ")
    NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | ${TR} -d " ")
    NT=$(sysctl hw.logicalcpu  | ${CUT} -d: -f2 | ${TR} -d " ")
  fi
  if [ "${NC}" ] && [ "${NT}" ]; then
    HT=$((NT/NC))
    NC=$((NT/HT))
  else
    NS=1 NC=1 NT=1 HT=1
  fi
  if [ "$(command -v numactl)" ]; then
    NN=$(numactl -H | ${GREP} "available:" | ${CUT} -d' ' -f2)
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
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
