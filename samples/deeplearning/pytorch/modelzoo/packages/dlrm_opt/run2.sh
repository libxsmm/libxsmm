

if [ "x$BASE_DIR" == "x" ] ; then
echo "SET BASE_DIR first" && exit 1
fi

source $BASE_DIR/miniconda3/bin/activate ref

for ARCH in clx bdx amd-rome ; do
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_mlperf_2k_${ARCH}_tppref.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_mlperf.sh --mini-batch-size=2048
done

source $BASE_DIR/miniconda3/bin/activate tpp

for ARCH in clx bdx amd-rome ; do
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_mlperf_2k_${ARCH}_tpp.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_mlperf.sh --arch-mlp-impl=xsmm --mini-batch-size=2048
done

source $BASE_DIR/miniconda3/bin/activate avx512

for ARCH in clx ; do
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_mlperf_2k_${ARCH}_avx512.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_mlperf.sh --mini-batch-size=2048
done

source $BASE_DIR/miniconda3/bin/activate ref

for ARCH in clx bdx amd-rome ; do
pushd ../dlrm_ref
sbatch -n 1 -N 1 -p ${ARCH}trb -o ${BASE_DIR}/results/dlrm_mlperf_2k_${ARCH}_ref.txt $BASE_DIR/run_scripts/run_single.sh bash cmd_mlperf.sh --mini-batch-size=2048
popd
done

