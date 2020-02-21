#ifndef _OPENMP
#define EIGEN_USE_THREADS
#endif
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "lstm_fwd.h"
#include "lstm_bwd.h"

#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T, bool XSMM_OP=false>
class XsmmLSTMCellOp : public OpKernel {
 public:
  explicit XsmmLSTMCellOp(OpKernelConstruction* ctx) : OpKernel(ctx), cached_batch_size(-2), cached_input_size(-2), cached_cell_size(-2), xsmm_handle(nullptr), cached_num_threads(-1), w_in_kcck_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    if(XSMM_OP) OP_REQUIRES_OK(ctx, ctx->GetAttr("w_in_kcck", &w_in_kcck_));
    OP_REQUIRES(ctx, use_peephole_ == false, errors::InvalidArgument("Peephole is not supported for XsmmLSTMCell"));
    printf("\nUsing XsmmLSTMCellFwd: forget_bias=%g\n", forget_bias_);
  }

#if 0
  bool UsesOmp() override {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
  }
#endif

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

    // Allocate our output tensors.
    Tensor* i_tensor = nullptr;
    //OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
    //                        {"h_prev"}, "i",
    //                        TensorShape({batch_size, cell_size}), &i_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "i", TensorShape({batch_size, cell_size}), &i_tensor));

    Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("cs", TensorShape({batch_size, cell_size}),
                                  &cs_tensor));

    Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("f", TensorShape({batch_size, cell_size}),
                                  &f_tensor));

    Tensor* o_tensor = nullptr;
    //OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
    //                        {"cs_prev"}, "o",
    //                        TensorShape({batch_size, cell_size}), &o_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "o", TensorShape({batch_size, cell_size}), &o_tensor));

    Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("ci", TensorShape({batch_size, cell_size}),
                                  &ci_tensor));

    Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("co", TensorShape({batch_size, cell_size}),
                                  &co_tensor));

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("h", TensorShape({batch_size, cell_size}),
                                  &h_tensor));

    //auto h_prev_ptr = h_prev_tensor->flat<T>();
    //auto h_ptr = h_tensor->flat<T>();
    //h_ptr(0) = h_prev_ptr(0) + 0.1;

    //printf("\nXsmmLSTMCell: parameters N=%ld C=%ld K=%ld forget_bias=%g\n", batch_size, input_size, cell_size, forget_bias_);

/*
    print_tensor_ptr("x", x_tensor);
    print_tensor_ptr("cs_prev", cs_prev_tensor);
    print_tensor_ptr("h_prev", h_prev_tensor);
    print_tensor_ptr("w", w_tensor);
    print_tensor_ptr("b", b_tensor);
    print_tensor_ptr("cs_prev_b4", cs_prev_tensor);
    //print_tensor_ptr("wci", wci_tensor);
    //print_tensor_ptr("wcf", wcf_tensor);
    //print_tensor_ptr("wco", wco_tensor);
    print_tensor_ptr("i", i_tensor);
    print_tensor_ptr("cs", cs_tensor);
    print_tensor_ptr("f", f_tensor);
    print_tensor_ptr("o", o_tensor);
    print_tensor_ptr("ci", ci_tensor);
    print_tensor_ptr("co", co_tensor);
    print_tensor_ptr("h", h_tensor);
    volatile int debug = 1;
    printf("Address of debug = %p  tid = %ld \n", &debug, (long)gettid());
    //while(debug == 1) { }
*/
    int offset_r = input_size * cell_size * 4;
    const float *xt = get_tensor_ptr(x_tensor);
    const float *csp = get_tensor_ptr(cs_prev_tensor);
    const float *hp = get_tensor_ptr(h_prev_tensor);
    const float *w = get_tensor_ptr(w_tensor);
    const float *r = get_tensor_ptr(w_tensor)+offset_r;
    const float *b = get_tensor_ptr(b_tensor);
    float *cst = get_tensor_ptr(cs_tensor);
    float *ht = get_tensor_ptr(h_tensor);
    float *it = get_tensor_ptr(i_tensor);
    float *ft = get_tensor_ptr(f_tensor);
    float *ot = get_tensor_ptr(o_tensor);
    float *cit = get_tensor_ptr(ci_tensor);
    float *cot = get_tensor_ptr(co_tensor);

#if 0
    lstm_fwd(batch_size, input_size, cell_size, 1, forget_bias_,
        xt,
        csp,
        hp,
        w,
        r,
        b,
        cst,
        ht,
        it,
        ft,
        ot,
        cit,
        cot);
#else
#if defined(_OPENMP)
      int nThreads = omp_get_max_threads(); /* number of threads */
#else
#ifndef DISABLE_EIGEN_THREADS
      const DeviceBase::CpuWorkerThreads* worker_threads =
        ctx->device()->tensorflow_cpu_worker_threads();

      int nThreads = worker_threads->num_threads;
#else
      int nThreads = 1; /* number of threads */
#endif
#endif

    if(xsmm_handle == nullptr || batch_size != cached_batch_size || input_size != cached_input_size || cell_size != cached_cell_size || cached_num_threads != nThreads) {
      if(xsmm_handle != nullptr) {
        //printf("Destroying existing libxsmm handle New NCK = (%d %d %d), old NCK = (%d %d %d)\n", batch_size, input_size, cell_size, cached_batch_size, cached_input_size, cached_cell_size);
        lstm_fwd_destroy( xsmm_handle );
        xsmm_handle = nullptr;
      }
      //printf("Creating new libxsmm handle NCK = (%d %d %d) nThreads = %d\n", batch_size, input_size, cell_size, nThreads);
      xsmm_handle = lstm_fwd_create( batch_size, input_size, cell_size, 1, nThreads, forget_bias_, (w_in_kcck_ ? 1 : 0),
          xt,
          csp,
          hp,
          w,
          r,
          b,
          cst,
          ht,
          it,
          ft,
          ot,
          cit,
          cot);

      cached_batch_size = batch_size;
      cached_input_size = input_size;
      cached_cell_size = cell_size;
      cached_num_threads = nThreads;

      OP_REQUIRES(ctx, xsmm_handle != nullptr, errors::InvalidArgument("lstm_fwd_create)_ returned null Xsmm handle"));
    }
    else {
      //printf("Reusing existing libxsmm handle\n");
    }
    lstm_fwd_set_ptr( xsmm_handle,
                       forget_bias_,
                       1,
                       xt,
                       csp,
                       hp,
                       w,
                       r,
                       b,
                       cst,
                       ht,
                       it,
                       ft,
                       ot,
                       cit,
                       cot );

#if defined(_OPENMP)
#pragma message "Using OPENMP Threading"
#if 0
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      //printf("TID %3d: executing lstm_fwd_execute_st OS tid = %6d\n", tid, gettid());
      lstm_fwd_execute_st( xsmm_handle, tid );
    }
#else
    lstm_fwd_execute_omp( xsmm_handle );
#endif
#else
#pragma message "Using EIGEN Threading"
#ifndef DISABLE_EIGEN_THREADS
    BlockingCounter count(cached_num_threads);
    for (int i = 0; i < cached_num_threads; ++i) {
      worker_threads->workers->Schedule([=, &count]() {
          lstm_fwd_execute_st( xsmm_handle, i );
          count.DecrementCount();
          });
    }
    count.Wait();

#else
#pragma message "NOT using threading"
    lstm_fwd_execute_st( xsmm_handle, 0 );
#endif
#endif
#endif

/*
    print_tensor_ptr("x", x_tensor);
    print_tensor_ptr("cs_prev", cs_prev_tensor);
    print_tensor_ptr("h_prev", h_prev_tensor);
    print_tensor_ptr("w", w_tensor);
    print_tensor_ptr("b", b_tensor);
    print_tensor_ptr("i", i_tensor);
    print_tensor_ptr("cs", cs_tensor);
    print_tensor_ptr("f", f_tensor);
    print_tensor_ptr("o", o_tensor);
    print_tensor_ptr("ci", ci_tensor);
    print_tensor_ptr("co", co_tensor);
    print_tensor_ptr("h", h_tensor);
*/
  }

 private:
  float forget_bias_;
  float cell_clip_;
  bool use_peephole_;
  bool w_in_kcck_;

  int cached_batch_size, cached_input_size, cached_cell_size, cached_num_threads;
  void *xsmm_handle;

  void print_tensor_ptr(const char *name, const Tensor* t) {
    auto ptr = t->flat<T>();
    const T* p = ptr.data();
    int dims = t->dims();
    if(dims > 0) printf("  XsmmLSTM: %-10s: [%d", name, t->dim_size(0));
    else printf("  XsmmLSTM: %-10s: [", name);
    for(int i = 1; i < dims; i++) printf(", %d", t->dim_size(i));
    printf("] @%p (%lld)   %g\n", p, t->NumElements(), p[0]);
    //for(int i = 0; i < t->NumElements(); i++)
    //  printf("DUMP:    %-10s %6d %12g\n", name, i, p[i]);
  }

  T *get_tensor_ptr(Tensor* t) { return t->flat<T>().data(); }
  const T *get_tensor_ptr(const Tensor* t) { return t->flat<T>().data(); }
};

#define REGISTER_KERNEL(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                          \
      Name("LSTMBlockCell").Device(DEVICE_CPU).TypeConstraint<T>("T").Label("xsmm"), \
      XsmmLSTMCellOp<CPUDevice, T>);
REGISTER_KERNEL(float);
//REGISTER_KERNEL(bfloat16);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                          \
      Name("XsmmLSTMCell").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      XsmmLSTMCellOp<CPUDevice, T, true>);
REGISTER_KERNEL(float);
#undef REGISTER_KERNEL


template <typename Device, typename T>
class XsmmLSTMCellGradOp : public OpKernel {
 public:
  explicit XsmmLSTMCellGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), cached_batch_size(-2), cached_input_size(-2), cached_cell_size(-2), xsmm_handle(nullptr), cached_num_threads(-1), w_in_kcck_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("w_in_kcck", &w_in_kcck_));
    printf("\nUsing XsmmLSTMCellBwd\n");
  }
#if 0
  explicit XsmmLSTMCellGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    printf("\nUsing XsmmLSTMCellBwd\n");
  }
#endif

#if 0
  bool UsesOmp() override {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
  }
#endif
  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wT_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_t", &wT_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_tensor));

    const Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_tensor));

    const Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_tensor));

    const Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_tensor));

    const Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_tensor));

    const Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_tensor));

    const Tensor* cs_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

    OP_REQUIRES(ctx, i_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "i.dim_size(0) != batch_size: ", i_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, i_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "i.dim_size(1) != cell_size: ", i_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, cs_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "cs.dim_size(0) != batch_size: ", cs_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "cs.dim_size(1) != cell_size: ", cs_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, f_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "f.dim_size(0) != batch_size: ", f_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, f_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "i.dim_size(1) != cell_size: ", f_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, o_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "o.dim_size(0) != batch_size: ", o_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, o_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "o.dim_size(1) != cell_size: ", o_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, ci_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "ci.dim_size(0) != batch_size: ", ci_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, ci_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "ci.dim_size(1) != cell_size: ", ci_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, co_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "co.dim_size(0) != batch_size: ", co_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, co_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "co.dim_size(1) != cell_size: ", co_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "cs_grad_tensor.dims(0) != batch_size: ",
                    cs_grad_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_grad_tensor.dims(1) != cell_size: ",
                                        cs_grad_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_grad_tensor.dims(0) != batch_size: ",
                                        h_grad_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_grad_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_grad_tensor.dims(1) != cell_size: ",
                                        h_grad_tensor->dim_size(1), " vs. ",
                                        cell_size));

    // Allocate our output tensors.
    Tensor* cs_prev_grad_tensor = nullptr;
    /*OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"cs_grad"}, "cs_prev_grad",
                 TensorShape({batch_size, cell_size}), &cs_prev_grad_tensor)); */
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "cs_prev_grad", TensorShape({batch_size, cell_size}), &cs_prev_grad_tensor));

    Tensor* h_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "h_prev_grad", TensorShape({batch_size, cell_size}), &h_prev_grad_tensor));

    Tensor* x_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "x_grad", TensorShape({batch_size, input_size}), &x_grad_tensor));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "w_grad", TensorShape({input_size+cell_size, 4*cell_size}), &w_grad_tensor));

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "b_grad", TensorShape({4*cell_size}), &b_grad_tensor));

    Tensor* wci_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"wci"}, "wci_grad", wci_tensor->shape(), &wci_grad_tensor));

    Tensor* wcf_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"wcf"}, "wcf_grad", wcf_tensor->shape(), &wcf_grad_tensor));

    Tensor* wco_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"wco"}, "wco_grad", wco_tensor->shape(), &wco_grad_tensor));

    // Allocate our temp tensors.
    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &di_tensor));

    const Device& device = ctx->eigen_device<Device>();

    print_tensor_ptr("xt", x_tensor);
    print_tensor_ptr("csp", cs_prev_tensor);
    print_tensor_ptr("hp", h_prev_tensor);
    print_tensor_ptr("w", w_tensor);
    print_tensor_ptr("b", b_tensor);
    print_tensor_ptr("it", i_tensor);
    print_tensor_ptr("cst", cs_tensor);
    print_tensor_ptr("ft", f_tensor);
    print_tensor_ptr("ot", o_tensor);
    print_tensor_ptr("cit", ci_tensor);
    print_tensor_ptr("cot", co_tensor);
    print_tensor_ptr("dcst", cs_grad_tensor);
    print_tensor_ptr("dht", h_grad_tensor);
    print_tensor_ptr("dcsp", cs_prev_grad_tensor);
    print_tensor_ptr("dhp", h_prev_grad_tensor);
    print_tensor_ptr("dxt", x_grad_tensor);
    print_tensor_ptr("dw", w_grad_tensor);
    print_tensor_ptr("db", b_grad_tensor);

    //printf("w_tensor = %p, wT_tensor = %p\n", get_tensor_ptr(w_tensor), get_tensor_ptr(wT_tensor));
    int w_in_trans = (get_tensor_ptr(w_tensor) != get_tensor_ptr(wT_tensor));

    int offset_r = input_size * cell_size * 4;
    const float *xt = get_tensor_ptr(x_tensor);
    const float *csp = get_tensor_ptr(cs_prev_tensor);
    const float *hp = get_tensor_ptr(h_prev_tensor);
    const float *w = get_tensor_ptr(wT_tensor);
    const float *r = get_tensor_ptr(wT_tensor)+offset_r;
    const float *b = get_tensor_ptr(b_tensor);
    const float *it = get_tensor_ptr(i_tensor);
    const float *cst = get_tensor_ptr(cs_tensor);
    const float *ft = get_tensor_ptr(f_tensor);
    const float *ot = get_tensor_ptr(o_tensor);
    const float *cit = get_tensor_ptr(ci_tensor);
    const float *cot = get_tensor_ptr(co_tensor);
    const float *dcs = get_tensor_ptr(cs_grad_tensor);
    const float *dht = get_tensor_ptr(h_grad_tensor);
    float *dcspt = get_tensor_ptr(cs_prev_grad_tensor);
    float *dhpt = get_tensor_ptr(h_prev_grad_tensor);
    float *dxt = get_tensor_ptr(x_grad_tensor);
    float *dw = get_tensor_ptr(w_grad_tensor);
    float *dr = get_tensor_ptr(w_grad_tensor)+offset_r;
    float *db = get_tensor_ptr(b_grad_tensor);

#if 0
    lstm_bwd(batch_size, input_size, cell_size, 1,
        xt,
        csp,
        hp,
        0,
        w,
        r,
        cst,
        it,
        ft,
        ot,
        cit,
        cot,
        dcs,
        dht,
        dxt,
        dcspt,
        dhpt,
        dw,
        dr,
        db );
#else
#if defined(_OPENMP)
      int nThreads = omp_get_max_threads(); /* number of threads */
#else
#ifndef DISABLE_EIGEN_THREADS
      const DeviceBase::CpuWorkerThreads* worker_threads =
        ctx->device()->tensorflow_cpu_worker_threads();

      int nThreads = worker_threads->num_threads;
#else
      int nThreads = 1; /* number of threads */
#endif
#endif

    if(xsmm_handle == nullptr || batch_size != cached_batch_size || input_size != cached_input_size || cell_size != cached_cell_size || cached_num_threads != nThreads) {
      if(xsmm_handle != nullptr) {
        //printf("Destroying existing libxsmm handle New NCK = (%d %d %d), old NCK = (%d %d %d)\n", batch_size, input_size, cell_size, cached_batch_size, cached_input_size, cached_cell_size);
        lstm_bwd_destroy( xsmm_handle );
        xsmm_handle = nullptr;
      }
      //printf("Creating new libxsmm handle NCK = (%d %d %d) nThreads = %d\n", batch_size, input_size, cell_size, nThreads);
      xsmm_handle = lstm_bwd_create( batch_size, input_size, cell_size, 1, nThreads, (w_in_kcck_ ? 1 : 0), w_in_trans,
          xt,
          csp,
          hp,
          0,
          w,
          r,
          cst,
          it,
          ft,
          ot,
          cit,
          cot,
          dcs,
          dht,
          dxt,
          dcspt,
          dhpt,
          dw,
          dr,
          db );

      cached_batch_size = batch_size;
      cached_input_size = input_size;
      cached_cell_size = cell_size;
      cached_num_threads = nThreads;

      OP_REQUIRES(ctx, xsmm_handle != nullptr, errors::InvalidArgument("lstm_bwd_create)_ returned null Xsmm handle"));
    }
    else {
      //printf("Reusing existing libxsmm handle\n");
    }
    lstm_bwd_set_ptr( xsmm_handle, w_in_trans,
                       1,
                       xt,
                       csp,
                       hp,
                       0,
                       w,
                       r,
                       cst,
                       it,
                       ft,
                       ot,
                       cit,
                       cot,
                       dcs,
                       dht,
                       dxt,
                       dcspt,
                       dhpt,
                       dw,
                       dr,
                       db );

#if defined(_OPENMP)
#if 0
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      lstm_bwd_execute_st( xsmm_handle, tid );
    }
#else
    lstm_bwd_execute_omp( xsmm_handle );
#endif
#else
#ifndef DISABLE_EIGEN_THREADS
    BlockingCounter count(cached_num_threads);
    for (int i = 0; i < cached_num_threads; ++i) {
      worker_threads->workers->Schedule([=, &count]() {
          lstm_bwd_execute_st( xsmm_handle, i );
          count.DecrementCount();
          });
    }
    count.Wait();

#else
    lstm_bwd_execute_st( xsmm_handle, 0 );
#endif
#endif
#endif
  }

 protected:
  bool use_peephole_;
  bool w_in_kcck_;

  int cached_batch_size, cached_input_size, cached_cell_size, cached_num_threads;
  void *xsmm_handle;

  void print_tensor_ptr(const char *name, const Tensor* t) {
    return;
    auto ptr = t->flat<T>();
    const T* p = ptr.data();
    int dims = t->dims();
    if(dims > 0) printf("  XsmmLSTM: %-10s: [%d", name, t->dim_size(0));
    else printf("  XsmmLSTM: %-10s: [", name);
    for(int i = 1; i < dims; i++) printf(", %d", t->dim_size(i));
    printf("] @%p (%lld)   %g\n", p, t->NumElements(), p[0]);
    //for(int i = 0; i < t->NumElements(); i++)
    //  printf("DUMP:    %-10s %6d %12g\n", name, i, p[i]);
  }
   T *get_tensor_ptr(Tensor* t) { return t->flat<T>().data(); }
  const T *get_tensor_ptr(const Tensor* t) { return t->flat<T>().data(); }
};

#define REGISTER_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("XsmmLSTMCellGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      XsmmLSTMCellGradOp<CPUDevice, T>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

template <typename Device, typename T, bool USE_CUBLAS>
class XsmmFusedLSTMOp : public OpKernel {
 public:
  explicit XsmmFusedLSTMOp(OpKernelConstruction* ctx) : OpKernel(ctx), cached_batch_size(-2), cached_input_size(-2), cached_cell_size(-2), cached_timelen(-2), xsmm_handle(nullptr), cached_num_threads(-1) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_residue", &use_residue_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_dropout", &use_dropout_));
    OP_REQUIRES(ctx, use_peephole_ == false, errors::InvalidArgument("Peephole is not supported for XsmmLSTMCell"));
    printf("\nUsing XsmmFusedLSTMFwd: forget_bias=%g\n", forget_bias_);
  }

#if 0
  bool UsesOmp() override {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
  }
#endif

  void Compute(OpKernelContext* ctx) override {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));
    const int64 timelen = x->dim_size(0);
    const int64 batch_size = x->dim_size(1);
    const int64 input_size = x->dim_size(2);

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));
    OP_REQUIRES(ctx, cs_prev_tensor->dims() == 2,
                errors::InvalidArgument("cs_prev must be 2D"));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    if (batch_size * input_size % 2 == 1) {
      LOG(WARNING) << "XsmmFusedLSTMOp is inefficient when both batch_size and "
                   << "input_size are odd. You are using: batch_size="
                   << batch_size << ", input_size=" << input_size;
    }
    if (batch_size * cell_size % 2 == 1) {
      LOG(WARNING) << "XsmmFusedLSTMOp is inefficient when both batch_size and "
                   << "cell_size are odd. You are using: batch_size="
                   << batch_size << ", cell_size=" << cell_size;
    }

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));
    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));
    OP_REQUIRES(ctx, w_tensor->dims() == 2,
                errors::InvalidArgument("w must be 2D"));
    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));
    OP_REQUIRES(ctx, wci_tensor->dims() == 1,
                errors::InvalidArgument("wci must be 1D"));
    OP_REQUIRES(ctx, wci_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "wci.dim_size(0) != cell_size: ", wci_tensor->dim_size(0),
                    " vs. ", cell_size));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));
    OP_REQUIRES(ctx, wcf_tensor->dims() == 1,
                errors::InvalidArgument("wcf must be 1D"));
    OP_REQUIRES(ctx, wcf_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "wcf.dim_size(0) != cell_size: ", wcf_tensor->dim_size(0),
                    " vs. ", cell_size));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));
    OP_REQUIRES(ctx, wco_tensor->dims() == 1,
                errors::InvalidArgument("wco must be 1D"));
    OP_REQUIRES(ctx, wco_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "wco.dim_size(0) != cell_size: ", wco_tensor->dim_size(0),
                    " vs. ", cell_size));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));
    OP_REQUIRES(ctx, b_tensor->dims() == 1,
                errors::InvalidArgument("b must be 1D"));
    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

    TensorShape batch_cell_shape({timelen, batch_size, cell_size});
    Tensor* i_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("i", batch_cell_shape, &i_out));

    Tensor* cs_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("cs", batch_cell_shape, &cs_out));

    Tensor* f_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("f", batch_cell_shape, &f_out));

    Tensor* o_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("o", batch_cell_shape, &o_out));

    Tensor* ci_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("ci", batch_cell_shape, &ci_out));

    Tensor* co_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("co", batch_cell_shape, &co_out));

    Tensor* h_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h", batch_cell_shape, &h_out));

    //printf("Inside %s:%d  %s()\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);
#if 0 //TMP tensors
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &icfo_tensor));
#endif
    const Device& device = ctx->eigen_device<Device>();

    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();

    int offset_r = input_size * cell_size * 4;
    const float *xt = get_tensor_ptr(x);
    const float *csp = get_tensor_ptr(cs_prev_tensor);
    const float *hp = get_tensor_ptr(h_prev_tensor);
    const float *w = get_tensor_ptr(w_tensor);
    const float *r = get_tensor_ptr(w_tensor)+offset_r;
    const float *b = get_tensor_ptr(b_tensor);
    float *cst = get_tensor_ptr(cs_out);
    float *ht = get_tensor_ptr(h_out);
    float *it = get_tensor_ptr(i_out);
    float *ft = get_tensor_ptr(f_out);
    float *ot = get_tensor_ptr(o_out);
    float *cit = get_tensor_ptr(ci_out);
    float *cot = get_tensor_ptr(co_out);

#if defined(_OPENMP)
    int nThreads = omp_get_max_threads(); /* number of threads */
#else
#ifndef DISABLE_EIGEN_THREADS
    const DeviceBase::CpuWorkerThreads* worker_threads =
      ctx->device()->tensorflow_cpu_worker_threads();

    int nThreads = worker_threads->num_threads;
#else
    int nThreads = 1; /* number of threads */
#endif
#endif

    if(xsmm_handle == nullptr || batch_size != cached_batch_size || input_size != cached_input_size || cell_size != cached_cell_size || timelen > cached_timelen || cached_num_threads != nThreads) {
      if(xsmm_handle != nullptr) {
        //printf("Destroying existing libxsmm handle New NCKT = (%d %d %d %d), old NCKT = (%d %d %d %d) \n", batch_size, input_size, cell_size, timelen, cached_batch_size, cached_input_size, cached_cell_size, cached_timelen);
        lstm_fwd_destroy( xsmm_handle );
        xsmm_handle = nullptr;
      }
      //printf("Creating new libxsmm handle NCKT = (%d %d %d %d) nThreads = %d\n", batch_size, input_size, cell_size, timelen, nThreads);
      xsmm_handle = lstm_fwd_create( batch_size, input_size, cell_size, timelen, nThreads, forget_bias_, 0,
          xt,
          csp,
          hp,
          w,
          r,
          b,
          cst,
          ht,
          it,
          ft,
          ot,
          cit,
          cot);

      cached_batch_size = batch_size;
      cached_input_size = input_size;
      cached_cell_size = cell_size;
      cached_timelen = timelen;
      cached_num_threads = nThreads;

      OP_REQUIRES(ctx, xsmm_handle != nullptr, errors::InvalidArgument("lstm_fwd_create)_ returned null Xsmm handle"));
    }
    else {
      //printf("Reusing existing libxsmm handle\n");
    }
    lstm_fwd_set_ptr( xsmm_handle,
                       forget_bias_,
                       timelen,
                       xt,
                       csp,
                       hp,
                       w,
                       r,
                       b,
                       cst,
                       ht,
                       it,
                       ft,
                       ot,
                       cit,
                       cot );

#if defined(_OPENMP)
#if 0
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      //printf("Thread %d calling lstm_fwd_execute_st OS TID = %d\n", tid, gettid());
      lstm_fwd_execute_st( xsmm_handle, tid );
    }
#else
    lstm_fwd_execute_omp( xsmm_handle );
#endif
#else
#ifndef DISABLE_EIGEN_THREADS
    BlockingCounter count(cached_num_threads);
    for (int i = 0; i < cached_num_threads; ++i) {
      worker_threads->workers->Schedule([=, &count]() {
          lstm_fwd_execute_st( xsmm_handle, i );
          count.DecrementCount();
          });
    }
    count.Wait();

#else
    lstm_fwd_execute_st( xsmm_handle, 0 );
#endif
#endif

#if 0  // Orig Code
    SliceHelper<Device, T> slicer(ctx);
    for (int64 t = 0; t < seq_len_max; ++t) {
      const Tensor x_tensor = slicer.InputSlice(*x, t, "x");
      const Tensor& cs_prev_tensor2 =
          t == 0 ? *cs_prev_tensor
                 : slicer.OutputSlice(cs_out, t - 1, "cs_prev");
      const Tensor& h_prev_tensor2 =
          t == 0 ? *h_prev_tensor : slicer.OutputSlice(h_out, t - 1, "h_prev");

      Tensor i_tensor = slicer.OutputSlice(i_out, t, "i_out");
      Tensor cs_tensor = slicer.OutputSlice(cs_out, t, "cs_out");
      Tensor f_tensor = slicer.OutputSlice(f_out, t, "f_out");
      Tensor o_tensor = slicer.OutputSlice(o_out, t, "o_out");
      Tensor ci_tensor = slicer.OutputSlice(ci_out, t, "ci_out");
      Tensor co_tensor = slicer.OutputSlice(co_out, t, "co_out");
      Tensor h_tensor = slicer.OutputSlice(h_out, t, "h_out");

      functor::LSTMBlockCellFprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                         cell_size)(
          ctx, device, forget_bias_, cell_clip_, use_peephole_,
          x_tensor.matrix<T>(), cs_prev_tensor2.matrix<T>(),
          h_prev_tensor2.matrix<T>(), w_tensor->matrix<T>(),
          wci_tensor->vec<T>(), wcf_tensor->vec<T>(), wco_tensor->vec<T>(),
          b_tensor->vec<T>(), xh_tensor.matrix<T>(), i_tensor.matrix<T>(),
          cs_tensor.matrix<T>(), f_tensor.matrix<T>(), o_tensor.matrix<T>(),
          ci_tensor.matrix<T>(), co_tensor.matrix<T>(), icfo_tensor.matrix<T>(),
          h_tensor.matrix<T>());
      slicer.FinishTimeStep();
    }

    if (seq_len_max < timelen) {
      Tensor cs_tensor = cs_out->Slice(seq_len_max, timelen);
      Tensor h_tensor = h_out->Slice(seq_len_max, timelen);

      functor::TensorUnalignedZero<Device, T>()(device,
                                                cs_tensor.unaligned_flat<T>());
      functor::TensorUnalignedZero<Device, T>()(device,
                                                h_tensor.unaligned_flat<T>());
    }
#endif
  }

 private:
  float forget_bias_;
  float cell_clip_;
  bool use_peephole_;
  bool use_residue_;
  bool use_dropout_;

  int cached_batch_size, cached_input_size, cached_cell_size, cached_timelen, cached_num_threads;
  void *xsmm_handle;

  void print_tensor_ptr(const char *name, const Tensor* t) {
    auto ptr = t->flat<T>();
    const T* p = ptr.data();
    int dims = t->dims();
    if(dims > 0) printf("  XsmmLSTM: %-10s: [%d", name, t->dim_size(0));
    else printf("  XsmmLSTM: %-10s: [", name);
    for(int i = 1; i < dims; i++) printf(", %d", t->dim_size(i));
    printf("] @%p (%lld)   %g\n", p, t->NumElements(), p[0]);
    //for(int i = 0; i < t->NumElements(); i++)
    //  printf("DUMP:    %-10s %6d %12g\n", name, i, p[i]);
  }

  T *get_tensor_ptr(Tensor* t) { return t->flat<T>().data(); }
  const T *get_tensor_ptr(const Tensor* t) { return t->flat<T>().data(); }

};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("XsmmFusedLSTM").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      XsmmFusedLSTMOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

template <typename Device, typename T, bool USE_CUBLAS>
class XsmmFusedLSTMGradOp : public OpKernel {
 public:
  explicit XsmmFusedLSTMGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), cached_batch_size(-2), cached_input_size(-2), cached_cell_size(-2), cached_timelen(-2), xsmm_handle(nullptr), cached_num_threads(-1) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_residue", &use_residue_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_dropout", &use_dropout_));
    OP_REQUIRES(ctx, use_peephole_ == false, errors::InvalidArgument("Peephole is not supported for XsmmLSTMCell"));
    printf("\nUsing XsmmFusedLSTMBwd:\n");
  }

#if 0
  bool UsesOmp() override {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
  }
#endif

  void Compute(OpKernelContext* ctx) override {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));
    const int64 timelen = x->dim_size(0);
    const int64 batch_size = x->dim_size(1);
    const int64 input_size = x->dim_size(2);

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));
    const int64 cell_size = w_tensor->dim_size(1) / 4;
    OP_REQUIRES(ctx, input_size + cell_size == w_tensor->dim_size(0),
                errors::InvalidArgument(
                    "w matrix rows don't match: ", input_size + cell_size,
                    " vs. ", w_tensor->dim_size(0)));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));
    OP_REQUIRES(
        ctx, cell_size == b_tensor->dim_size(0) / 4,
        errors::InvalidArgument("w and b cell_size don't match: ", cell_size,
                                " vs. ", b_tensor->dim_size(0)));

    const Tensor* i_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_out));

    const Tensor* cs_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_out));

    const Tensor* f_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_out));

    const Tensor* o_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_out));

    const Tensor* ci_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_out));

    const Tensor* co_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_out));

    const Tensor* h_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_out));

    const Tensor* cs_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad));

    const Tensor* h_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad));

    TensorShape batch_input_shape({timelen, batch_size, input_size});
    Tensor* x_grad;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("x_grad", batch_input_shape, &x_grad));

    Tensor* cs_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("cs_prev_grad", cs_prev_tensor->shape(),
                                        &cs_prev_grad_tensor));

    Tensor* h_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("h_prev_grad", h_prev_tensor->shape(),
                                        &h_prev_grad_tensor));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("w_grad", w_tensor->shape(), &w_grad_tensor));

    Tensor* wci_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wci_grad", wci_tensor->shape(),
                                             &wci_grad_tensor));

    Tensor* wcf_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wcf_grad", wcf_tensor->shape(),
                                             &wcf_grad_tensor));

    Tensor* wco_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("wco_grad", wco_tensor->shape(),
                                             &wco_grad_tensor));

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("b_grad", b_tensor->shape(), &b_grad_tensor));

    TensorShape batch_cell_shape({batch_size, cell_size});

    //printf("Inside %s:%d  %s()\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);
#if 0 // TMP tensors
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           xh_tensor.shape(), &xh_grad_tensor));

    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &di_tensor));

    Tensor dicfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &dicfo_tensor));

    Tensor cs_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &cs_grad_tensor));

    Tensor h_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           batch_cell_shape, &h_grad_tensor));

#endif
    const Device& device = ctx->eigen_device<Device>();

#if 0 // Orig Impl
    functor::TensorZero<Device, T>()(device, cs_grad_tensor.flat<T>());
    functor::TensorZero<Device, T>()(device, cs_prev_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, h_grad_tensor.flat<T>());
    functor::TensorZero<Device, T>()(device, h_prev_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, w_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wci_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wcf_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wco_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, b_grad_tensor->flat<T>());
#endif

    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();

    int offset_r = input_size * cell_size * 4;
    const float *xt = get_tensor_ptr(x);
    const float *csp = get_tensor_ptr(cs_prev_tensor);
    const float *hp = get_tensor_ptr(h_prev_tensor);
    const float *ht = get_tensor_ptr(h_out);
    const float *w = get_tensor_ptr(w_tensor);
    const float *r = get_tensor_ptr(w_tensor)+offset_r;
    const float *b = get_tensor_ptr(b_tensor);
    const float *it = get_tensor_ptr(i_out);
    const float *cst = get_tensor_ptr(cs_out);
    const float *ft = get_tensor_ptr(f_out);
    const float *ot = get_tensor_ptr(o_out);
    const float *cit = get_tensor_ptr(ci_out);
    const float *cot = get_tensor_ptr(co_out);
    const float *dcs = get_tensor_ptr(cs_grad);
    const float *dht = get_tensor_ptr(h_grad);
    float *dcspt = get_tensor_ptr(cs_prev_grad_tensor);
    float *dhpt = get_tensor_ptr(h_prev_grad_tensor);
    float *dxt = get_tensor_ptr(x_grad);
    float *dw = get_tensor_ptr(w_grad_tensor);
    float *dr = get_tensor_ptr(w_grad_tensor)+offset_r;
    float *db = get_tensor_ptr(b_grad_tensor);

#if defined(_OPENMP)
      int nThreads = omp_get_max_threads(); /* number of threads */
#else
#ifndef DISABLE_EIGEN_THREADS
      const DeviceBase::CpuWorkerThreads* worker_threads =
        ctx->device()->tensorflow_cpu_worker_threads();

      int nThreads = worker_threads->num_threads;
#else
      int nThreads = 1; /* number of threads */
#endif
#endif

    if(xsmm_handle == nullptr || batch_size != cached_batch_size || input_size != cached_input_size || cell_size != cached_cell_size || timelen > cached_timelen || cached_num_threads != nThreads) {
      if(xsmm_handle != nullptr) {
        //printf("Destroying existing libxsmm handle New NCKT = (%d %d %d %d), old NCKT = (%d %d %d %d) \n", batch_size, input_size, cell_size, timelen, cached_batch_size, cached_input_size, cached_cell_size, cached_timelen);
        lstm_bwd_destroy( xsmm_handle );
        xsmm_handle = nullptr;
      }
      //printf("Creating new libxsmm handle NCKT = (%d %d %d %d) nThreads = %d\n", batch_size, input_size, cell_size, timelen, nThreads);
      xsmm_handle = lstm_bwd_create( batch_size, input_size, cell_size, timelen, nThreads, 0, 0,
          xt,
          csp,
          hp,
          ht,
          w,
          r,
          cst,
          it,
          ft,
          ot,
          cit,
          cot,
          dcs,
          dht,
          dxt,
          dcspt,
          dhpt,
          dw,
          dr,
          db );

      cached_batch_size = batch_size;
      cached_input_size = input_size;
      cached_cell_size = cell_size;
      cached_timelen = timelen;
      cached_num_threads = nThreads;

      OP_REQUIRES(ctx, xsmm_handle != nullptr, errors::InvalidArgument("lstm_bwd_create)_ returned null Xsmm handle"));
    }
    else {
      //printf("Reusing existing libxsmm handle\n");
    }
    lstm_bwd_set_ptr( xsmm_handle, 0,
                       timelen,
                       xt,
                       csp,
                       hp,
                       ht,
                       w,
                       r,
                       cst,
                       it,
                       ft,
                       ot,
                       cit,
                       cot,
                       dcs,
                       dht,
                       dxt,
                       dcspt,
                       dhpt,
                       dw,
                       dr,
                       db );

#if defined(_OPENMP)
#if 0
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      //printf("Thread %d calling lstm_bwd_execute_st OS TID = %d\n", tid, gettid());
      lstm_bwd_execute_st( xsmm_handle, tid );
    }
#else
    lstm_bwd_execute_omp( xsmm_handle );
#endif
#else
#ifndef DISABLE_EIGEN_THREADS
    BlockingCounter count(cached_num_threads);
    for (int i = 0; i < cached_num_threads; ++i) {
      worker_threads->workers->Schedule([=, &count]() {
          lstm_bwd_execute_st( xsmm_handle, i );
          count.DecrementCount();
          });
    }
    count.Wait();

#else
    lstm_bwd_execute_st( xsmm_handle, 0 );
#endif
#endif

#if 0 // Orig Impl
    SliceHelper<Device, T> slicer(ctx);
    for (int64 t = seq_len_max - 1; t >= 0; --t) {
      const Tensor& x_tensor = slicer.InputSlice(*x, t, "x");
      const Tensor& cs_prev_tensor2 =
          t == 0 ? *cs_prev_tensor
                 : slicer.InputSlice(*cs_out, t - 1, "cs_prev");
      const Tensor& h_prev_tensor2 =
          t == 0 ? *h_prev_tensor : slicer.InputSlice(*h_out, t - 1, "h_prev");
      const Tensor& i_tensor = slicer.InputSlice(*i_out, t, "i_out");
      const Tensor& cs_tensor = slicer.InputSlice(*cs_out, t, "cs_out");
      const Tensor& f_tensor = slicer.InputSlice(*f_out, t, "f_out");
      const Tensor& o_tensor = slicer.InputSlice(*o_out, t, "o_out");
      const Tensor& ci_tensor = slicer.InputSlice(*ci_out, t, "ci_out");
      const Tensor& co_tensor = slicer.InputSlice(*co_out, t, "co_out");

      // Grab previous CS grad.
      const Tensor& const_cs_prev_grad_tensor = *cs_prev_grad_tensor;
      const Tensor const_cs_grad_slice =
          slicer.InputSlice(*cs_grad, t, "cs_grad");
      functor::TensorAdd<Device, T>()(
          device, const_cs_prev_grad_tensor.flat<T>(),
          const_cs_grad_slice.flat<T>(), cs_grad_tensor.flat<T>());

      // Combine previous h grad and h grad coming on top.
      const Tensor& const_h_prev_grad_tensor = *h_prev_grad_tensor;
      const Tensor const_h_grad_slice = slicer.InputSlice(*h_grad, t, "h_grad");
      functor::TensorAdd<Device, T>()(
          device, const_h_prev_grad_tensor.flat<T>(),
          const_h_grad_slice.flat<T>(), h_grad_tensor.flat<T>());

      const Tensor& const_cs_grad_tensor = cs_grad_tensor;
      const Tensor& const_h_grad_tensor = h_grad_tensor;

      Tensor x_grad_tensor = slicer.OutputSlice(x_grad, t, "x_grad");
      functor::BlockLSTMBprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                     cell_size)(
          ctx, device, use_peephole_, x_tensor.matrix<T>(),
          cs_prev_tensor2.matrix<T>(), h_prev_tensor2.matrix<T>(),
          w_tensor->matrix<T>(), wci_tensor->vec<T>(), wcf_tensor->vec<T>(),
          wco_tensor->vec<T>(), b_tensor->vec<T>(), xh_tensor.matrix<T>(),
          i_tensor.matrix<T>(), cs_tensor.matrix<T>(), f_tensor.matrix<T>(),
          o_tensor.matrix<T>(), ci_tensor.matrix<T>(), co_tensor.matrix<T>(),
          const_cs_grad_tensor.matrix<T>(), const_h_grad_tensor.matrix<T>(),
          do_tensor.matrix<T>(), dcs_tensor.matrix<T>(), dci_tensor.matrix<T>(),
          df_tensor.matrix<T>(), di_tensor.matrix<T>(),
          dicfo_tensor.matrix<T>(), cs_prev_grad_tensor->matrix<T>(),
          h_prev_grad_tensor->matrix<T>(), xh_grad_tensor.matrix<T>(),
          x_grad_tensor.matrix<T>(), w_grad_tensor->matrix<T>(),
          wci_grad_tensor->vec<T>(), wcf_grad_tensor->vec<T>(),
          wco_grad_tensor->vec<T>(), b_grad_tensor->vec<T>());
      slicer.FinishTimeStep();
    }

    if (seq_len_max < timelen) {
      Tensor x_grad_tensor = x_grad->Slice(seq_len_max, timelen);
      functor::TensorUnalignedZero<Device, T>()(
          device, x_grad_tensor.unaligned_flat<T>());
    }
#endif
  }

 private:
  bool use_peephole_;
  bool use_residue_;
  bool use_dropout_;

  int cached_batch_size, cached_input_size, cached_cell_size, cached_timelen, cached_num_threads;
  void *xsmm_handle;

  void print_tensor_ptr(const char *name, const Tensor* t) {
    auto ptr = t->flat<T>();
    const T* p = ptr.data();
    int dims = t->dims();
    if(dims > 0) printf("  XsmmLSTM: %-10s: [%d", name, t->dim_size(0));
    else printf("  XsmmLSTM: %-10s: [", name);
    for(int i = 1; i < dims; i++) printf(", %d", t->dim_size(i));
    printf("] @%p (%lld)   %g\n", p, t->NumElements(), p[0]);
    //for(int i = 0; i < t->NumElements(); i++)
    //  printf("DUMP:    %-10s %6d %12g\n", name, i, p[i]);
  }

  T *get_tensor_ptr(Tensor* t) { return t->flat<T>().data(); }
  const T *get_tensor_ptr(const Tensor* t) { return t->flat<T>().data(); }

};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("XsmmFusedLSTMGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      XsmmFusedLSTMGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
// REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

