

if [ "x$BASE_DIR" == "x" ] ; then
echo "SET BASE_DIR first" && exit 1
fi

source $BASE_DIR/miniconda3/bin/activate tpp

ARCH=clx
NP=2
sbatch -n $NP -N $(( (NP+1) / 2 )) -p ${ARCH}trb -w pcl-skx38 -o ${BASE_DIR}/results/scale_dlrm_large_${ARCH}_tpp_${NP}s.txt $BASE_DIR/run_scripts/run_dist.sh bash cmd_large.sh --arch-mlp-impl=xsmm --mini-batch-size=$(( NP * 1024 ))
NP=1
sbatch -n $NP -N $(( (NP+1) / 2 )) -p ${ARCH}trb -w pcl-skx36 -o ${BASE_DIR}/results/scale_dlrm_large_${ARCH}_tpp_${NP}s.txt $BASE_DIR/run_scripts/run_dist.sh bash cmd_large.sh --arch-mlp-impl=xsmm --mini-batch-size=$(( NP * 1024 ))
for NP in 4 8 16 32 64 ; do
sbatch -n $NP -N $(( (NP+1) / 2 )) -p ${ARCH}trb -o ${BASE_DIR}/results/scale_dlrm_large_${ARCH}_tpp_${NP}s.txt $BASE_DIR/run_scripts/run_dist.sh bash cmd_large.sh --arch-mlp-impl=xsmm --mini-batch-size=$(( NP * 1024 ))
done

