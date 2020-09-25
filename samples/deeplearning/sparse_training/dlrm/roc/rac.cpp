#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <parallel/algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>
#include <chrono>
#include <numeric>

//using namespace tbb;
using namespace std;

static PyObject* roc_auc_score(PyObject* self, PyObject* args);

static PyMethodDef ScoreMethods[] = {
  {"roc_auc_score", roc_auc_score, METH_VARARGS, "roc_auc_score"},
  {NULL, NULL, 0, NULL}
};

static PyModuleDef rocaucscoremodule = {PyModuleDef_HEAD_INIT, "roc_auc_score",
                                        "Modul with one function: roc_auc_score", -1, ScoreMethods};

PyMODINIT_FUNC PyInit_roc_auc_score() {
    PyObject* module = PyModule_Create(&rocaucscoremodule);
    if (!module) {
        return NULL;
    }
    import_array();
    return module;
}

template <typename TYPE>
double roc_auc_score_(PyArrayObject* actual_numpy, PyArrayObject* prediction_numpy, int size, double &log_loss, double &accuracy) {

  TYPE* actual = (TYPE*)PyArray_DATA(actual_numpy);
  TYPE* prediction = (TYPE*)PyArray_DATA(prediction_numpy);

  vector<TYPE> predictedRank(size, 0.0);

  int nPos = 0, nNeg = 0;

#pragma omp parallel for reduction(+:nPos)
  for(int i = 0; i < size; i++)
    nPos += (int)actual[i];

  nNeg = size - nPos;
  //printf("nPos = %d nNeg = %d, Total = %d\n", nPos, nNeg, size);

  vector<TYPE> pos((int)nPos);
  vector<TYPE> neg((int)nNeg);

  double acc = 0.0;
  double loss = 0.0;
#pragma omp parallel for reduction(+:acc,loss)
  for(int i = 0; i < size; i++) {
    auto rpred = roundf(prediction[i]);
    if(actual[i] == rpred) acc += 1;
    loss += (actual[i]*log(prediction[i])) + ((1-actual[i]) * log(1-prediction[i]));
  }

  accuracy = acc / size;
  log_loss = -loss/size;

  vector<pair<TYPE, int> > v_sort(size);
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
      v_sort[i] = make_pair(prediction[i], i);
  }

  __gnu_parallel::sort(v_sort.begin(), v_sort.end(), [](auto &left, auto &right) {
  //std::sort(v_sort.begin(), v_sort.end(), [](auto &left, auto &right) {
      return left.first < right.first;
  });

  int r = 1;
  int n = 1;
  size_t i = 0;

  while (i < size) {
      size_t j = i;
      while ((j < (v_sort.size() - 1)) && (v_sort[j].first == v_sort[j + 1].first)) {
          j++;
      }
      n = j - i + 1;
      for (size_t j = 0; j < n; ++j) { // parallel this
          int idx = v_sort[i+j].second;
          predictedRank[idx] = r + ((n - 1) * 0.5);
      }
      r += n;
      i += n;
  }

  double filteredRankSum = 0;
#pragma omp parallel for reduction(+:filteredRankSum)
  for (size_t i = 0; i < size; ++i) {
    if (actual[i] == 1) {
      filteredRankSum += predictedRank[i];
    }
  }
  //printf("sum = %f\n", (filteredRankSum - (nPos*((nPos+1)/2))));
  double score = (filteredRankSum - ((double)nPos*((nPos+1.0)/2.0))) / ((double)nPos * nNeg);
  return score;

}

static PyObject* roc_auc_score(PyObject* self, PyObject* args) {

    PyArrayObject* actual_numpy = NULL;
    PyArrayObject* prediction_numpy = NULL;

    int n;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &actual_numpy, &PyArray_Type, &prediction_numpy)) {
        return NULL;
    }

    int nd = PyArray_NDIM(actual_numpy);
    if (nd != 1) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown shape data");
        return NULL;
    }

    nd = PyArray_NDIM(prediction_numpy);
    if (nd != 1) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown shape data");
        return NULL;
    }

    const int size = (int)PyArray_SHAPE(actual_numpy)[0];
    double score, loss, acc;

    switch (PyArray_TYPE(actual_numpy)) {
        case NPY_FLOAT:
            score = roc_auc_score_<float>(actual_numpy, prediction_numpy, size, loss, acc);
            break;
        case NPY_DOUBLE:
            score = roc_auc_score_<double>(actual_numpy, prediction_numpy, size, loss, acc);
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "Unknown source data type");
            return NULL;
    }

    return Py_BuildValue("(ddd)", score, loss, acc);
}

