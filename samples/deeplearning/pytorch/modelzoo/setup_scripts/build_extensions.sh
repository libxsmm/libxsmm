SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd)"
BASE_DIR=`dirname ${SCRIPT_DIR}`
PREFIX=${BASE_DIR}/miniconda3

if [ "x${LIBXSMM_ROOT}" == "x" ] ; then
  echo "Please set LIBXSMM_ROOT first"
  exit 1
fi


# REF env
source ${PREFIX}/bin/activate ref
for DIR in pcl_bert_tpp_ref pcl_embedding_bag_tpp_ref ; do
  cd $DIR
  python setup.py clean
  rm -rf build dist
  python setup.py install
  cd ..
done

# OPT env
source ${PREFIX}/bin/activate tpp
for DIR in pcl_bert_tpp_opt pcl_embedding_bag_tpp_opt pcl_mlp_tpp ; do
  cd $DIR
  python setup.py clean
  rm -rf build dist
  python setup.py install
  cd ..
done

# AVX512 env
source ${PREFIX}/bin/activate avx512
for DIR in pcl_bert_avx512 pcl_embedding_bag_avx512 pcl_mlp_tpp ; do
  cd $DIR
  python setup.py clean
  rm -rf build dist
  python setup.py install
  cd ..
done

