#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
CPU=${HERE}/../../../scripts/tool_cpuinfo.sh

EXP_ID=0
N_CORES=$(${CPU} -nc)

for op in 0 1 2 3 4
do
  for redop in 0 1 2 3
  do
    for oporder in 0 1
    do
      for regvecin in 0 1
      do
        # When op is copy then only supported order is VECIDX_VECIN is uding regular vec in
        if [ "$op" == '0' ] && [ "$oporder" == '0' ] && [ "$regvecin" == '1' ]; then
          continue
        fi
        for implicitidx in 0 1
        do
          # When regvecin is regular then only implicit idx
          if [ "$regvecin" == '1' ] && [ "$implicitidx" == '0' ]; then
            continue
          fi

          for argopmode in 0 1 2 3
          do
            # For redop NONE/ADD there are no argopmode supported
            if [ "$redop" == '0' ] && [ "$argopmode" -gt '0' ]; then
              continue
            fi
            if [ "$redop" == '1' ] && [ "$argopmode" -gt '0' ]; then
              continue
            fi
            if [ "$redop" == '0' ] && [ "$regvecin" == '0' ]; then
              continue
            fi
            if [ "$op" == '0' ] && [ "$oporder" == '0' ] && [ "$argopmode" == '1' ]; then
              continue
            fi
            if [ "$op" == '0' ] && [ "$oporder" == '0' ] && [ "$argopmode" == '3' ]; then
              continue
            fi
            if [ "$op" == '0' ] && [ "$oporder" == '1' ] && [ "$argopmode" == '2' ]; then
              continue
            fi
            if [ "$op" == '0' ] && [ "$oporder" == '1' ] && [ "$argopmode" == '3' ]; then
              continue
            fi
            if [ "$regvecin" == '1' ] && [ "$argopmode" -gt '0' ]; then
              continue
            fi

            for usebf16 in 0 1
            do

              if [ "$op" == '0' ] ; then
                NAME="opreduce_op_copy_"
              elif [ "$op" == '1' ] ; then
                NAME="opreduce_op_add_"
              elif [ "$op" == '2' ] ; then
                NAME="opreduce_op_mul_"
              elif [ "$op" == '3' ] ; then
                NAME="opreduce_op_sub_"
              else
                NAME="opreduce_op_div_"
              fi

              if [ "$redop" == '0' ] ; then
                NAME=$NAME"redop_none_oporder_"$oporder"_regvecin_"$regvecin"_implicitidx_"$implicitidx"_argopmode_"$argopmode"_"
              elif [ "$redop" == '1' ] ; then
                NAME=$NAME"redop_add_oporder_"$oporder"_regvecin_"$regvecin"_implicitidx_"$implicitidx"_argopmode_"$argopmode"_"
              elif [ "$redop" == '2' ] ; then
                NAME=$NAME"redop_max_oporder_"$oporder"_regvecin_"$regvecin"_implicitidx_"$implicitidx"_argopmode_"$argopmode"_"
              else
                NAME=$NAME"redop_min_oporder_"$oporder"_regvecin_"$regvecin"_implicitidx_"$implicitidx"_argopmode_"$argopmode"_"
              fi

              if [ "$usebf16" == '0' ] ; then
                NAME=$NAME"32b.sh"
              else
                NAME=$NAME"16b.sh"
              fi

              echo $NAME
              CORE_ID=$((${EXP_ID} % 56))

              taskset -c ${CORE_ID} ./${NAME} > result_${NAME} &

              EXP_ID=$((${EXP_ID} + 1))

              if (( $EXP_ID % $N_CORES == 0 )) ; then
                echo "Waiting for a batch of experiments to finish...."
                wait
              fi

            done
          done
        done
      done
    done
  done
done


