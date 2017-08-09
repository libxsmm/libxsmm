#!/bin/bash

if [ $1 = "-h" ]
then
  echo "Usage: $(basename $0) matrices.txt iters numa (1-mcdram/0-DDR)"
  exit
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_HW_SUBSET=1T
  export KMP_AFFINITY=compact,granularity=fine
  export KMP_AFFINITY=proclist=[1-67],granularity=thread,explicit,norespect
  export OMP_NUM_THREADS=67
else
  echo "using environment OMP settings!"
fi


_fn=${1:-"deepbench_matrices.txt"}
_it=${2:-100}
NUMA=${3:-1}

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

#----bgemm parameters
_MB_="16 24 32 64"
_NB_="16 24 32 64"
_KB_="16 32 24 48 64 96 128 192"
#mb1="0.1 0.2 0.3 0.4 0.5 0.8 0.16 0.32 1 2 4 8 10 16"
#nb1="0.1 0.2 0.3 0.4 0.5 0.8 0.16 0.32 1 2 4 8 10 16"
#kb1="0.1 0.2 0.3 0.4 0.5 0.8 0.16 0.32 1 2 4 8 10 16"
#kb2="0.1 0.2 0.3 0.4 0.5 0.8 0.16 0.32 1 2 4 8 10 16"
MBT=4096
NBT=4096
KBT=4096
mb11="0.1 0.2 0.4 0.8"
nb11="0.1"
kb11="0.1"
mb12="0.1 0.2 0.4 0.8"
nb12="0.1 0.2 0.4 0.8"
kb12="0.1 0.2 0.4 0.8"
kb2="0.1 0.2 0.4 0.5 0.8 0.16 0.32"
order="0 1 2"
perflog="perfSweep.log"

function bgemm_test {
  echo "M=$M N=$N K=$K it=$it"
    bin="$NUMACTL ./bgemm"
    log="$((M))_$((N))_$((K)).out"
    for _mb in $mb
    do
      for _nb in $nb
      do
        for _kb in $kb
        do
          for _mb1 in $mb1
          do
            for _nb1 in $nb1
            do
              for _kb1 in $kb1
              do
                for _kb2 in $kb2
                do
                  for _o in $order
                  do
                    if [ $(bc <<< "$_mb1 < 1") -eq 1 ]; then
                      IFS="." read temp _MB1 <<< $_mb1
                    else
                      _MB1=$(($M/$_mb1))
                    fi
                    if [ $(bc <<< "$_nb1 < 1") -eq 1 ]; then
                      IFS="." read temp _NB1 <<< $_nb1
                    else
                      _NB1=$(($N/$_nb1))
                    fi
                    if [ $(bc <<< "$_kb1 < 1") -eq 1 ]; then
                      IFS="." read temp _KB1 <<< $_kb1
                    else
                      _KB1=$(($K/$_kb1))
                    fi
                    if [ $(bc <<< "$_kb2 < 1") -eq 1 ]; then
                      IFS="." read temp _KB2 <<< $_kb2
                    else
                      _KB2=$(($K/$_kb2))
                    fi
                    echo "$bin $M $N $K $_mb $_nb $_kb $_o $it $_MB1 $_NB1 $_KB1 $_KB2"
                    echo "----------------------------------------------------------------------" >> $log
                    echo "$bin $M $N $K $_mb $_nb $_kb $_o $it $_MB1 $_NB1 $_KB1 $_KB2" >> $log
                    $bin $M $N $K $_mb $_nb $_kb $_o $it $_MB1 $_NB1 $_KB1 $_KB2 >> $log
                    echo " " >> $log
                  done
                done
              done
            done
          done
        done
      done
    done
    best=$(grep LIBXSMM $log |  awk ' BEGIN { val = 0 } { if ($2 > val) val = $2 } END { print val }')
    echo "$_b,$best" >> $perflog
}


function run_bsgemm {
M=$1
N=$2
K=$3
_AT=$4
_BT=$5
if [[ $# -gt 5 ]]
then
  mb=$6
  nb=$7
  kb=$8
else
  mb=32
  nb=32
  kb=32
fi
#_it=$9
#_bin=$7


if [[ "$mb" -gt "$M" ]]
then
  mb=$M
fi
if [[ "$nb" -gt "$N" ]]
then
  nb=$N
fi
if [[ "$kb" -gt "$K" ]]
then
  kb=$K
fi
if [[ "$((M % mb))" -gt 0 ]]
then
  #mb=$M
  M=$((mb*(M/mb+1)))
fi
if [[ "$((N % nb))" -gt 0 ]]
then
  #nb=$N
  N=$((nb*(N/nb+1)))
fi
if [[ "$((K % kb))" -gt 0 ]]
then
  #kb=$K
  K=$((kb*(K/kb+1)))
fi

mb=$_MB_
nb=$_NB_
kb=$_KB_

if [[ "$M" -gt "$MBT" ]]; then
  mb1=$mb12
else
  mb1=$mb11
fi
if [[ "$N" -gt "$NBT" ]]; then
  nb1=$nb12
else
  nb1=$nb11
fi
if [[ "$K" -gt "$KBT" ]]; then
  kb1=$kb12
else
  kb1=$kb11
fi

_Trans=0
if [[ "$_AT" == "T" ]]
then
  _Trans=1
  echo "!!! $M $N $K $_AT $_BT - Not supported !!!, doing $N $M $K N N instead"
  t_M=$M
  t_mb=$mb
  M=$N
  N=$t_M
  mb=$nb
  nb=$t_mb
fi
if [[ "$_BT" == "T" ]]
then
  _Trans=2
  echo "!!! $M $N $K $_AT $_BT - Not supported !!!, doing $M $K $N N N instead"
  t_K=$K
  t_kb=$kb
  K=$N
  N=$t_K
  kb=$nb
  nb=$t_kb
fi

if [[ "$M" -gt "4000" ]]; then
  if [[ "$N" -gt "4000" ]]; then
    if [[ "$K" -gt "4000" ]]; then
      it=100
    fi
  fi
fi

bgemm_test
echo "--------------------------------------------------------------------------------------"
}

nc=$(wc -l $_fn)
idx=1

cat $_fn | while read line
do
  if [ ! -z "$line" ]; then
    echo -n "($idx/$nc)"
    it=$_it
    run_bsgemm $line
  fi
  idx=$((idx+1))
done

