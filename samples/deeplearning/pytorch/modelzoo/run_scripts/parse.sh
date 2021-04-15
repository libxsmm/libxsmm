for ARCH in 1 2 4 8 16 32 64; do echo -n "${ARCH}S " ; cat results/scale_dlrm_small_clx_tpp_${ARCH}s.txt | awk '/Finished training it ..\/20/ {sum += $9; cou++;} END {print sum/cou  "   " cou;}' ; done

for ARCH in 2 4 8 16 32 64; do echo -n "${ARCH}S " ; cat /nfs_home/ddkalamk/bert/transformers/examples/question-answering/tpp_fp32_${ARCH}s.txt | awk '/Step:\ [1-46-9].,/ {sum += $NF; cou++;} END {print sum/cou  "   " cou;}' ; done

