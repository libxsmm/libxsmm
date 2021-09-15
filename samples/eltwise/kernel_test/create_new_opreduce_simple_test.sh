#!/bin/bash



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

              cp opreduce_template.sh $NAME
              sed "s/OP=0/OP=${op}/g" -i $NAME
              sed "s/OPRED=1/OPRED=${redop}/g" -i $NAME
              sed "s/OPORDER=1/OPORDER=${oporder}/g" -i $NAME
              sed "s/REGVECIN=1/REGVECIN=${regvecin}/g" -i $NAME
              sed "s/IMPLICITIDX=1/IMPLICITIDX=${implicitidx}/g" -i $NAME
              sed "s/OPARG=0/OPARG=${argopmode}/g" -i $NAME
              sed "s/USE_BF16=0/USE_BF16=${usebf16}/g" -i $NAME


              # these loops inside each script
              #for idxtype in 0 1
              #do
              #  for scale in 0 1
              #  do
              #    for eqld in 0 1
              #    do
              #      N_SCRIPTS=$((${N_SCRIPTS} + 1))
              #    done
              #  done
              #done
            done
          done
        done
      done
    done
  done
done



