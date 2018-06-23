#!/bin/bash
#SBATCH --partition=clx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=2:00:00

export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,compact,1,0
export CHECK=1
ITERS=1000

# Initialize Env vars
export FWD_BF=1
export BWD_BF=1
export UPD_BF=1
export FWD_2D_BLOCKING=1
export BWD_2D_BLOCKING=1
export UPD_2D_BLOCKING=1
export FWD_ROW_TEAMS=1
export FWD_COLUMN_TEAMS=1
export BWD_ROW_TEAMS=1
export BWD_COLUMN_TEAMS=1
export UPD_ROW_TEAMS=1
export UPD_COLUMN_TEAMS=1
export IFM_SUBTASKS=1
export OFM_SUBTASKS=1

# Tune FWD
MB=2016
BFN=24
IFM=1024
OFM=1024

for PASS in 'FWD' 'BWD' 'UPD'; do
  if [ $PASS == "FWD" ]
  then
    PASS_ARG='F'
  fi
  if [ $PASS == "BWD" ]
  then
    PASS_ARG='B'
  fi
  if [ $PASS == "UPD" ]
  then
    PASS_ARG='U'
  fi

  echo ${PASS_ARG}

  rm -f ${PASS}_TUNING_${MB}_${IFM}_${OFM}
  touch ${PASS}_TUNING_${MB}_${IFM}_${OFM}


  THREADS=28
  export OMP_NUM_THREADS=${THREADS}
  export ${PASS}_2D_BLOCKING=0

  if [ $PASS == "UPD" ]
  then
    for IFMSUBTASKS in 1 2; do
      export IFM_SUBTASKS=${IFMSUBTASKS}
      for OFMSUBTASKS in 1 2; do
        export OFM_SUBTASKS=${OFMSUBTASKS}
        for BFM in 32 64; do
          for BFACC in 1 2 4 7 14 21 42 84; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}
          done
        done
      done
    done
  else
    for BFM in 32 64; do
      for BFACC in 1 2 4 8; do
        export ${PASS}_BF=${BFACC}
        ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}
      done
    done
  fi

  export ${PASS}_2D_BLOCKING=1
  export IFM_SUBTASKS=1
  export OFM_SUBTASKS=1
  if [ $PASS == "UPD" ]
  then
    for ROWS in 7 14; do
      export ${PASS}_ROW_TEAMS=${ROWS}
      COLUMNS=$((THREADS / ROWS))
      export ${PASS}_COLUMN_TEAMS=${COLUMNS}
      for BFM in 32 64; do
        for BFACC in 1 2 4 7 14 21 42 84; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}
        done
      done
    done
  else
    for ROWS in 7 14; do
      export ${PASS}_ROW_TEAMS=${ROWS}
      COLUMNS=$((THREADS / ROWS))
      export ${PASS}_COLUMN_TEAMS=${COLUMNS}
      for BFM in 32 64; do
        for BFACC in 1 2 4 8; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}
        done
      done
    done
  fi
done


#srun -n 1 ./layer_example_bf16 ${ITERS} ${MB} 512 512 0 F B 32 32 32
#srun -n 1 ./layer_example_bf16 ${ITERS} ${MB} 100 1024 0 F B 32 32 50
#srun -n 1 ./layer_example_bf16 ${ITERS} ${MB} 1024 1 0 F B 32 32 32


