#!/usr/bin/env bash
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

MB=2160
BFN=24

# Tune layer (100,1024)
IFM=100
OFM=1024
for THREADS in 20 24; do
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

    rm -f ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
    touch ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}

    export OMP_NUM_THREADS=${THREADS}
    export ${PASS}_2D_BLOCKING=0

    if [ $PASS == "UPD" ]
    then
      for IFMSUBTASKS in 1 2; do
        export IFM_SUBTASKS=${IFMSUBTASKS}
        for OFMSUBTASKS in 1 2; do
          export OFM_SUBTASKS=${OFMSUBTASKS}
          for BFM in 32 64; do
            for BFACC in 1 2 3 6 9 10 15 30 45 90; do
              export ${PASS}_BF=${BFACC}
              ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 50 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
            done
          done
        done
      done
      for IFMSUBTASKS in 1; do
        export IFM_SUBTASKS=${IFMSUBTASKS}
        for OFMSUBTASKS in 1 2; do
          export OFM_SUBTASKS=${OFMSUBTASKS}
          for BFM in 32 64; do
            for BFACC in 1 2 3 6 9 10 15 30 45 90; do
              export ${PASS}_BF=${BFACC}
              ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 100 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
            done
          done
        done
      done
    fi

    if [ $PASS == "FWD" ]
    then
      for BFM in 32 64; do
        for BFACC in 1; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 100 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
        done
        for BFACC in 1 2 ; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 50 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
        done
      done
    fi

    if [ $PASS == "BWD" ]
    then
      for BFM in 32 64; do
        for BFACC in 1 2 4 8; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 50 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 100 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
        done
      done
    fi

    export ${PASS}_2D_BLOCKING=1
    export IFM_SUBTASKS=1
    export OFM_SUBTASKS=1
    if [ $PASS == "UPD" ]
    then
      for COLUMNS in 2 4; do
        export ${PASS}_COLUMN_TEAMS=${COLUMNS}
        ROWS=$((THREADS / COLUMNS))
        export ${PASS}_ROW_TEAMS=${ROWS}
        for BFM in 32 64; do
          for BFACC in 1 2 3 6 9 10 15 30 45 90; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 50 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 100 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
        done
      done
    fi

    if [ $PASS == "FWD" ]
    then
      for COLUMNS in 2 4; do
        export ${PASS}_COLUMN_TEAMS=${COLUMNS}
        ROWS=$((THREADS / COLUMNS))
        export ${PASS}_ROW_TEAMS=${ROWS}
        for BFM in 32 64; do
          for BFACC in 1; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 100  >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
          for BFACC in 1 2; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 50  >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
        done
      done
    fi

    if [ $PASS == "BWD" ]
    then
      for COLUMNS in 2 4; do
        export ${PASS}_COLUMN_TEAMS=${COLUMNS}
        ROWS=$((THREADS / COLUMNS))
        export ${PASS}_ROW_TEAMS=${ROWS}
        for BFM in 32 64; do
          for BFACC in 1 2 4 8; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 100 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} ${BFM} 50 >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
        done
      done
    fi
  done
done

# Tune layer (1024,1)
IFM=1024
OFM=1
for THREADS in 20 24; do
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

    rm -f ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
    touch ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}

    export OMP_NUM_THREADS=${THREADS}
    export ${PASS}_2D_BLOCKING=0

    if [ $PASS == "UPD" ]
    then
      for IFMSUBTASKS in 1 2; do
        export IFM_SUBTASKS=${IFMSUBTASKS}
        for OFMSUBTASKS in 1; do
          export OFM_SUBTASKS=${OFMSUBTASKS}
          for BFM in 16 32 64; do
            for BFACC in 1 2 3 6 9 10 15 30 45 90; do
              export ${PASS}_BF=${BFACC}
              ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} 1 ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
            done
          done
        done
      done
    fi

    if [ $PASS == "FWD" ]
    then
      for BFM in 16 32 64; do
        for BFACC in 1 2 4 8; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} 1 ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
        done
      done
    fi


    if [ $PASS == "BWD" ]
    then
      for BFM in 16 32 64; do
        for BFACC in 1; do
          export ${PASS}_BF=${BFACC}
          ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} 1 ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
        done
      done
    fi

    export ${PASS}_2D_BLOCKING=1
    export IFM_SUBTASKS=1
    export OFM_SUBTASKS=1
    if [ $PASS == "UPD" ]
    then
      for COLUMNS in 2 4; do
        export ${PASS}_COLUMN_TEAMS=${COLUMNS}
        ROWS=$((THREADS / COLUMNS))
        export ${PASS}_ROW_TEAMS=${ROWS}
        for BFM in 16 32 64; do
          for BFACC in 1 2 3 6 9 10 15 30 45 90; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} 1 ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
        done
      done
    fi

    if [ $PASS == "FWD" ]
    then
      for COLUMNS in 2 4; do
        export ${PASS}_COLUMN_TEAMS=${COLUMNS}
        ROWS=$((THREADS / COLUMNS))
        export ${PASS}_ROW_TEAMS=${ROWS}
        for BFM in 16 32 64; do
          for BFACC in 1 2 4 8; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} 1 ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
        done
      done
    fi

    if [ $PASS == "BWD" ]
    then
      for COLUMNS in 2 4; do
        export ${PASS}_COLUMN_TEAMS=${COLUMNS}
        ROWS=$((THREADS / COLUMNS))
        export ${PASS}_ROW_TEAMS=${ROWS}
        for BFM in 16 32 64; do
          for BFACC in 1; do
            export ${PASS}_BF=${BFACC}
            ./layer_example_bf16 ${ITERS} ${MB} ${IFM} ${OFM} 0 ${PASS_ARG} B ${BFN} 1 ${BFM} >> ${PASS}_TUNING_${MB}_${IFM}_${OFM}_threads_${THREADS}
          done
        done
      done
    fi

  done
done

