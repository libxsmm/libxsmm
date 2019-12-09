/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#pragma once
#include <map>
#include <list>
#include <vector>
#include <algorithm>
#include <set>
#include <omp.h>
#include <sys/time.h>
#include <stdlib.h>
#include "proto/gxm.pb.h"
#include "Engine.fwd.hpp"
#include "MLNode.fwd.hpp"
#include "Config.hpp"
#include "Task.hpp"
#include "common.hpp"
#include "Solver.hpp"
#include "libxsmm.h"
#ifdef USE_MLSL
#include "mpi.h"
#endif

using namespace std;
using namespace gxm;

extern int iter;

#ifdef USE_MLSL
#include "mlsl.hpp"
//using namespace MLSL;
#endif

#define TRAIN 0
#define VAL 1
#define TEST 2
#define START_GUARD_BAND 64
#define END_GUARD_BAND 64
#define CANARY 0x7F
#define NDIFFS 10

struct dupChecker_ {
  inline dupChecker_() : tmpSet() {}
  inline bool operator()(Task *t) {
    return tmpSet.insert(t).second;
  }
  private:
    std::set<Task *> tmpSet;
};

class MLEngine
{
  protected:
    NTGParameter ntgparam_;
    NodeParameter np_;
    SolverParameter sparam_;
#ifdef USE_MLSL
    MLSL::Distribution *data_parallelism;
    MLSL::Session *session_;
#endif
    vector<MLNode*> ntg_;
    list<Task*> etg_[3]; // 0 - Training, 1 - Validation, 2 - testing
    SolverParams *solverParams_;
    SolverNode* solver_;
    Tensor* tenScratch_;
    TensorBuf* tenScratchBuf_;

    struct TensorPair
    {
      string name;
      Tensor* t;
    };
    typedef list<TensorPair> TensorList;
    typedef TensorList::iterator Iter;
    typedef map<string, Iter> Tmap;

    Tmap inTensorMap_, outTensorMap_, weightTensorMap_, biasTensorMap_, statsTensorMap_;
    TensorList defTList_, inTList_, outTList_, wTList_, biasTList_, statsTList_;

    bool inferenceOnly_, load_from_checkpoint_;
    string checkpoint_dir_, checkpoint_format_;
    int num_epochs_, exec_mode_, current_epoch_, current_batch_;
    int data_type_;
    int num_machines_, num_machine_groups_, num_threads_;
    int batch_size_, num_train_batches_, num_test_batches_, num_test_views_;
    int global_node_id_;
    float lr_, *wt_lr_mult_[NUM_NUMA_NODES], *wt_decay_mult_[NUM_NUMA_NODES];
    float *bias_lr_mult_[NUM_NUMA_NODES], *bias_decay_mult_[NUM_NUMA_NODES];
    float scf_=0;

    void *input_buf_=NULL;
    void *fact_buf_=NULL, *bact_buf_=NULL, *wbuf_=NULL;
    void *weight_buf_[NUM_NUMA_NODES]={NULL}, *wdiff_buf_[NUM_NUMA_NODES]={NULL};
    void *winc_buf_[NUM_NUMA_NODES]={NULL}, *lpweight_buf_[NUM_NUMA_NODES]={NULL};
    void *lpwdiff_buf_[NUM_NUMA_NODES]={NULL};
#if 1
    void *bias_buf_[NUM_NUMA_NODES]={NULL}, *bidiff_buf_[NUM_NUMA_NODES]={NULL};
    void *biinc_buf_[NUM_NUMA_NODES]={NULL}, *stats_buf_[NUM_NUMA_NODES]={NULL};
#else
    void *bias_buf_=NULL, *bidiff_buf_=NULL;
    void *biinc_buf_=NULL, *stats_buf_=NULL;
#endif
    int total_weights_, total_biases_, orig_total_weights_;
    void *scratch[NUM_NUMA_NODES]={NULL};

    vector<int> input_can_ptr;
    vector<int> fact_can_ptr, bact_can_ptr;
    vector<int> wt_can_ptr, wdiff_can_ptr, winc_can_ptr;
    vector<int> bias_can_ptr, stats_can_ptr, bidiff_can_ptr, biinc_can_ptr;
#ifdef USE_MLSL
    vector<MLSL::Operation*> wtgrad_comms_vec, bias_grad_comms_vec, combo_grad_comms_vec;
#endif
    int ic, fac, bac, wtc, wdc, wic, bic, sic, bidc, biic;

    void create_schedule(int);
    void optimize_schedule(int);
    void allocate_tensor_memory(Tensor*, int, void*);
    void clear_history(TensorList);
    int find_in_nodeTypeList(string);
    void checkpoint(TensorList L, int);
    void read_checkpoint_file(TensorBuf*, string, string);
    void load_checkpoint(TensorList, int, string);
    void canary_check(void*, vector<int>&, int);
    void allocate_memory(string, TensorList, int, vector<int>&, int*, long long int*);
    void* allocate_gradient_tensor(TensorList, int, int, long long int);
    void insertSplitNodes(NTGParameter& p, NTGParameter* ps);
    void convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len, int numa_node);
    void convert_f32_bf16(float** in, libxsmm_bfloat16** out, int len);
    void convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len);
    void waitForComms(string);

  public:
    MLEngine() {}
    virtual ~MLEngine() {}

    void create(int mode, string ntgConfig, string solverConfig);
    bool register_tensor(string name, int type, Tensor* t);
    Tensor* get_tensor(string name, int type);
    void execute_on_thread(int num_threads, MLNode* node, void (*fname)(int tid));
    void set_global_strategy(MachineParameter* mparam);
    void run(int mode);

    SolverNode* getSolver() { return solver_; }
    TensorBuf* getScratchBuffer() { return tenScratchBuf_; }

    bool is_inference_only() { return inferenceOnly_; }

    int get_num_threads() { return num_threads_; }
    int get_num_machines() { return num_machines_; }
    int get_num_machine_groups() { return num_machine_groups_; }
    int get_num_epochs() { return num_epochs_;}
    int get_current_epoch() { return current_epoch_; }
    int get_current_batch() { return current_batch_; }
    int get_execution_mode() { return exec_mode_; }
    int get_global_node_id() { return global_node_id_; }
    int get_num_train_batches() { return num_train_batches_; }
    int get_num_test_batches() { return num_test_batches_; }
    int get_num_test_views() {return num_test_views_; }
    int get_batch_size() { return batch_size_; }
    float get_scaling_factor() { return scf_; }
#ifdef USE_MLSL
    vector<MLSL::Operation*>& get_wtgrad_comms_vec() { return wtgrad_comms_vec; }
    vector<MLSL::Operation*>& get_bias_grad_comms_vec() { return bias_grad_comms_vec; }
    vector<MLSL::Operation*>& get_combo_grad_comms_vec() { return combo_grad_comms_vec; }
#endif

    void set_batch_size(int b) {batch_size_ = b; }
    void set_num_train_batches(int ntrainb) {num_train_batches_ = ntrainb; }
    void set_num_test_batches(int ntestb) {num_test_batches_ = ntestb; }
    void set_num_test_views(int ntestv) {num_test_views_ = ntestv; }
    void set_learning_rate(float lr) { lr_ = lr; }
    void set_scaling_factor(float scf) { scf_ = scf; }
#ifdef USE_MLSL
    MLSL::Distribution* get_distribution() { return data_parallelism; }
    MLSL::Session *get_session() { return session_; }
#endif

};

