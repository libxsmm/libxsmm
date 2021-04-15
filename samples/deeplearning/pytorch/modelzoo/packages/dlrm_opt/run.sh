

if [ "x$BASE_DIR" == "x" ] ; then
echo "SET BASE_DIR first" && exit 1
fi

source $BASE_DIR/miniconda3/bin/activate ref

for ARCH in clx bdx amd-rome ; do
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_small_${ARCH}_tppref.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_small.sh
done

source $BASE_DIR/miniconda3/bin/activate tpp

for ARCH in clx bdx amd-rome ; do
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_small_${ARCH}_tpp.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_small.sh --arch-mlp-impl=xsmm
done

source $BASE_DIR/miniconda3/bin/activate avx512

for ARCH in clx ; do
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_small_${ARCH}_avx512.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_small.sh
done

source $BASE_DIR/miniconda3/bin/activate ref

for ARCH in clx bdx amd-rome ; do
pushd ../dlrm_ref
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_small_${ARCH}_ref.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_small.sh
popd
done


