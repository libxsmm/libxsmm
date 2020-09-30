/** \file mdarray.hpp
 *
 *  \brief Contains implementation of multidimensional array class.
 */

#ifndef __MDARRAY_HPP__
#define __MDARRAY_HPP__

#include <array>
#include <atomic>
#include <cassert>
#include <string>
#include <functional>
#include <initializer_list>
#include <memory>
#include <algorithm>
#include <cstring>
#include <signal.h>
#include <string>
#include <type_traits>
#include <vector>

#ifdef HAVE_CUDA
#include "GPU/cuda.hpp"
#endif

#if defined(HAVE_MKL) || defined(__MKL)
# include <mkl.h>
#elif defined(__CBLAS)
# include <cblas.h>
#else
# define CblasRowMajor 101
# define CblasColMajor 102
extern "C" void cblas_dger(int, int, int, double, const double*, int, const double*, int, double*, int);
#endif

#if !defined(CBLAS_LAYOUT)
# define CBLAS_LAYOUT int
#endif

#ifdef NDEBUG
#define mdarray_assert(condition__)
#else
#define mdarray_assert(condition__)                                 \
    {                                                               \
        if (!(condition__)) {                                       \
            printf("Assertion (%s) failed ", #condition__);         \
            printf("at line %i of file %s\n", __LINE__, __FILE__);  \
            printf("array label: %s\n", label_.c_str());            \
            int mdarray_assert_i_ = 0;                              \
            for (; mdarray_assert_i_ < N; mdarray_assert_i_++)      \
                printf("dim[%i].size = %llu\n", mdarray_assert_i_,  \
                    static_cast<unsigned long long>(                \
                    dims_[mdarray_assert_i_].size()));              \
            raise(SIGTERM);                                         \
            exit(-13);                                              \
        }                                                           \
    }
#endif

/// Type of the main processing unit.
enum device_t
{
    /// CPU device.
    CPU = 0,

    /// GPU device (with CUDA programming model).
    GPU = 1
};

/// Type of memory.
/** Various combinations of flags can be used. To check for any host memory
   (pinned or non-pinned): \code{.cpp} mem_type & memory_t::host ==
   memory_t::host \endcode To check for pinned memory: \code{.cpp} mem_type &
   memory_t::host_pinned == memory_t::host_pinned \endcode To check for device
   memory: \code{.cpp} mem_type & memory_t::device == memory_t::device \endcode
*/
enum class memory_t : unsigned int
{
    /// Nothing.
    none = 0b000,
    /// Host memory.
    host = 0b001,
    /// Pinned host memory. This is host memory + extra bit flag.
    host_pinned = 0b011,
    /// Device memory.
    device = 0b100
};

inline constexpr memory_t operator&(memory_t a__, memory_t b__)
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) &
                                 static_cast<unsigned int>(b__));
}

inline constexpr memory_t operator|(memory_t a__, memory_t b__)
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) |
                                 static_cast<unsigned int>(b__));
}

inline constexpr bool on_device(memory_t mem_type__)
{
    return (mem_type__ & memory_t::device) == memory_t::device ? true : false;
}

/// Index descriptor of mdarray.
class mdarray_index_descriptor
{
  private:
    /// Beginning of index.
    int64_t begin_{0};

    /// End of index.
    int64_t end_{-1};

    /// Size of index.
    size_t size_{0};

  public:
    /// Constructor of empty descriptor.
    mdarray_index_descriptor()
    {
    }

    /// Constructor for index range [0, size).
    mdarray_index_descriptor(size_t const size__)
        : begin_(0)
        , end_(size__ - 1)
        , size_(size__)
    {
    }

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(int64_t const begin__, int64_t const end__)
        : begin_(begin__)
        , end_(end__)
        , size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(std::pair<int, int> const range__)
        : begin_(range__.first)
        , end_(range__.second)
        , size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Return first index value.
    inline int64_t begin() const
    {
        return begin_;
    }

    /// Return last index value.
    inline int64_t end() const
    {
        return end_;
    }

    /// Return index size.
    inline size_t size() const
    {
        return size_;
    }
};


/// Base class of multidimensional array.
template <typename T, int N, CBLAS_LAYOUT format = CblasColMajor>
class mdarray_base
{
  protected:
    /// Optional array label.
    std::string label_;

    /// Unique pointer to the allocated memory.
    ///  std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_{nullptr};

    /// Raw pointer.
    T* raw_ptr_{nullptr};

    // layout Fortran by default
    CBLAS_LAYOUT layout_{CblasColMajor};

    // the table is allocated outside the class
    bool allocated_outside_cpu_{true};

    // the table is allocated outside the class
    bool allocated_outside_gpu_{true};

#ifdef __GPU
    /// Unique pointer to the allocated GPU memory.
    /// std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_device_{nullptr};

    /// Raw pointer to GPU memory
    T* raw_ptr_device_{nullptr};
#endif

    /// Array dimensions.
    std::array<mdarray_index_descriptor, N> dims_;

    /// List of offsets to compute the element location by dimension indices.
    std::array<int64_t, N> offsets_;

    /// leading dimension on CPU and GPUs (can be different because of alignment constraint)
    size_t ld_cpu_{0};
    size_t ld_gpu_{0};

    size_t raw_data_size_{0};

    void init_dimensions(std::array<mdarray_index_descriptor, N> const dims__)
    {
        dims_ = dims__;

        offsets_[0] = -dims_[0].begin();
        size_t ld{1};
        for (int i = 1; i < N; i++) {
            ld *= dims_[i - 1].size();
            offsets_[i] = ld;
            offsets_[0] -= ld * dims_[i].begin();
        }
        ld_cpu_ = dims_[0].size();

#ifdef HAVE_CUDA
        ni = dims_[0].size() / WARP_SIZE;
        lda_gpu_ = 32 * ( (dims_[0].size() % WARP_SIZE) != 0 + ni);
#endif
    }

  private:
    inline int64_t idx(std::array<int64_t, N> idx__) const
    {
#ifdef NDEBUG
        for (int d = 0; d < N; d++) {
            mdarray_assert(idx__[d] >= dims_[d].begin() && i0 <= dims_[d].end());
        }
#endif
        size_t i = offsets_[0] + idx__[0];
        for (int d = 0; d < idx__.size(); d++)
            i += idx__[d] * offsets_[d];
        mdarray_assert(/*i >= 0 &&*/ i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0) const
    {
        static_assert(N == 1, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        size_t i = offsets_[0] + i0;
        mdarray_assert(/*i >= 0 &&*/ i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1) const
    {
        static_assert(N == 2, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1];
        mdarray_assert(/*i >= 0 &&*/ i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1,
                       int64_t const i2) const
    {
        static_assert(N == 3, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2];
        mdarray_assert(/*i >= 0 &&*/ i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2,
                       int64_t const i3) const
    {
        static_assert(N == 4, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        mdarray_assert(i3 >= dims_[3].begin() && i3 <= dims_[3].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2] +
                   i3 * offsets_[3];
        mdarray_assert(/*i >= 0 &&*/ i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2,
                       int64_t const i3, int64_t const i4) const
    {
        static_assert(N == 5, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        mdarray_assert(i3 >= dims_[3].begin() && i3 <= dims_[3].end());
        mdarray_assert(i4 >= dims_[4].begin() && i4 <= dims_[4].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2] +
                   i3 * offsets_[3] + i4 * offsets_[4];
        mdarray_assert(/*i >= 0 &&*/ i < size());
        return i;
    }

    template <device_t pu>
    inline T* at_idx(int64_t const idx__)
    {
        switch (pu) {
            case CPU: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case GPU: {
#ifdef HAVE_CUDA
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                printf("error at line %i of file %s: not compiled with GPU support\n",
                       __LINE__, __FILE__);
                exit(0);
#endif
            }
        }
        return nullptr;
    }

    template <device_t pu>
    inline T const* at_idx(int64_t const idx__) const
    {
        switch (pu) {
            case CPU: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case GPU: {
#ifdef HAVE_CUDA
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                printf("error at line %i of file %s: not compiled with GPU support\n",
                       __LINE__, __FILE__);
                exit(0);
#endif
            }
        }
        return nullptr;
    }

    /// Copy constructor is forbidden
    mdarray_base(mdarray_base<T, N, format> const& src) = delete;

    /// Assignment operator is forbidden
    mdarray_base<T, N, format>&
    operator=(mdarray_base<T, N, format> const& src) = delete;

  public:
    /// Constructor of an empty array.
    mdarray_base()
    {
    }

    /// Destructor.
    ~mdarray_base()
    {
    }

    /// Move constructor
    mdarray_base(mdarray_base<T, N, format>&& src)
        : label_(src.label_)
        , //unique_ptr_(std::move(src.unique_ptr_)),
        raw_ptr_(src.raw_ptr_)
        , allocated_outside_cpu_(src.allocated_outside_cpu_)
#ifdef __GPU
        , allocated_outside_gpu_(src.allocated_outside_gpu_)
        ,
        //unique_ptr_device_(std::move(src.unique_ptr_device_)),
        raw_ptr_device_(src.raw_ptr_device_)
        , layout_(src.Layout_)
#endif
    {
        for (int i = 0; i < N; i++) {
            dims_[i]    = src.dims_[i];
            offsets_[i] = src.offsets_[i];
        }
        src.raw_ptr_ = nullptr;
#ifdef __GPU
        src.raw_ptr_device_ = nullptr;
#endif
    }

    /// Move assigment operator
    inline mdarray_base<T, N, format>&
    operator=(mdarray_base<T, N, format>&& src)
    {
        if (this != &src) {
            label_                 = src.label_;
            layout_                = src.layout_;
            raw_ptr_               = src.raw_ptr_;
            raw_data_size_         = src.raw_data_size_;
            allocated_outside_cpu_ = src.allocated_outside_cpu_;
            src.raw_ptr_           = nullptr;
#ifdef __GPU
            raw_ptr_device_        = src.raw_ptr_device_;
            src.raw_ptr_device_    = nullptr;
            allocated_outside_gpu_ = src.allocated_outside_gpu_;
#endif
            for (int i = 0; i < N; i++) {
                dims_[i]    = src.dims_[i];
                offsets_[i] = src.offsets_[i];
            }
        }
        return *this;
    }

    /// Allocate memory for array.
    void allocate(memory_t memory__)
    {
        if ((memory__ & memory_t::host) == memory_t::host) {
#if defined(_WIN32)
            raw_ptr_ = static_cast<T*>(_aligned_malloc(sizeof(T) * size<CPU>(), 256));
            if (raw_ptr_ == nullptr)
#else
            if (posix_memalign(reinterpret_cast<void**>(&raw_ptr_), 256, sizeof(T) * size<CPU>()) != 0)
#endif
            {
                printf("Allocation failed\n");
                std::abort();
            }

            allocated_outside_cpu_ = false;
        }

#ifdef __GPU
        if ((memory__ & memory_t::device) == memory_t::device) {
            cudaMalloc(&raw_prt_device_, sizeof(T) * size<GPU>());
            allocated_outside_gpu_ = false;
        }
#endif
    }

    void deallocate(memory_t memory__)
    {
        if ((memory__ & memory_t::host) == memory_t::host) {
            if ((raw_ptr_ != nullptr) && (!allocated_outside_cpu_)) {
#if defined(_WIN32)
                _aligned_free(raw_ptr_);
#else
                free(raw_ptr_);
#endif
                raw_ptr_ = nullptr;
            }
        }

#ifdef __GPU
        if ((memory__ & memory_t::device) == memory_t::device) {
            if ((raw_ptr_ != nullptr) && (!allocated_outside_gpu_)) {
                free(raw_ptr_device_);
                raw_ptr_device_ = nullptr;
            }
        }
#endif
    }

    void clear()
    {
        deallocate(memory_t::host);
#ifdef HAVE_CUDA
        deallocate(memory_t::device);
#endif
    }

    inline T& operator()(int64_t const i0)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0)];
    }

    inline T const& operator()(int64_t const i0) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1)];
        else
            return raw_ptr_[idx(i1, i0)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1)];
        else
            return raw_ptr_[idx(i1, i0)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1, i2)];
        else
            return raw_ptr_[idx(i2, i1, i0)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1,
                               int64_t const i2) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1, i2)];
        else
            return raw_ptr_[idx(i2, i1, i0)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2,
                         int64_t const i3)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1, i2, i3)];
        else
            return raw_ptr_[idx(i3, i2, i1, i0)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1,
                               int64_t const i2, int64_t const i3) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1, i2, i3)];
        else
            return raw_ptr_[idx(i3, i2, i1, i0)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2,
                         int64_t const i3, int64_t const i4)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1, i2, i3, i4)];
        else
            return raw_ptr_[idx(i4, i3, i2, i1, i0)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1,
                               int64_t const i2, int64_t const i3,
                               int64_t const i4) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        if (layout_ == CblasColMajor)
            return raw_ptr_[idx(i0, i1, i2, i3, i4)];
        else
            return raw_ptr_[idx(i4, i3, i2, i1, i0)];
    }

    inline T& operator()(std::array<int64_t, N> idx__)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(idx__)];
    }

    inline T& operator[](size_t const idx__)
    {
        mdarray_assert(/*idx__ >= 0 &&*/ idx__ < size());
        return raw_ptr_[idx__];
    }

    inline T const& operator[](size_t const idx__) const
    {
        assert(/*idx__ >= 0 &&*/ idx__ < size());
        return raw_ptr_[idx__];
    }

    template <device_t pu>
    inline T* at()
    {
        return at_idx<pu>(0);
    }

    template <device_t pu>
    inline T const* at() const
    {
        return at_idx<pu>(0);
    }

    template <device_t pu>
    inline T* at(int64_t const i0)
    {
        return at_idx<pu>(idx(i0));
    }

    template <device_t pu>
    inline T const* at(int64_t const i0) const
    {
        return at_idx<pu>(idx(i0));
    }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1)
    {
        if (layout_ == CblasColMajor)
            return at_idx<pu>(idx(i0, i1));
        else
            return at_idx<pu>(idx(i1, i0));
    }

    template <device_t pu>
    inline T const* at(int64_t const i0, int64_t const i1) const
    {
        if (layout_ == CblasColMajor)
            return at_idx<pu>(idx(i0, i1));
        else
            return at_idx<pu>(idx(i1, i0));
    }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1, int64_t const i2)
    {
        if (layout_ == CblasColMajor)
            return at_idx<pu>(idx(i0, i1, i2));
        else
            return at_idx<pu>(idx(i2, i1, i0));
    }

    template <device_t pu>
    inline T const* at(int64_t const i0, int64_t const i1, int64_t const i2) const
        {
            if (layout_ == CblasColMajor)
                return at_idx<pu>(idx(i0, i1, i2));
            else
                return at_idx<pu>(idx(i2, i1, i0));
        }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1, int64_t const i2,
                 int64_t const i3)
    {
        if (layout_ == CblasColMajor)
            return at_idx<pu>(idx(i0, i1, i2, i3));
        else
            return at_idx<pu>(idx(i3, i2, i1, i0));
    }

    template <device_t pu>
    inline T const* at(int64_t const i0, int64_t const i1, int64_t const i2,
                 int64_t const i3) const
        {
            if (layout_ == CblasColMajor)
                return at_idx<pu>(idx(i0, i1, i2, i3));
            else
                return at_idx<pu>(idx(i3, i2, i1, i0));
        }


    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1, int64_t const i2,
                 int64_t const i3, int64_t const i4)
    {
        if (layout_ == CblasColMajor)
            return at_idx<pu>(idx(i0, i1, i2, i3, i4));
        else
            return at_idx<pu>(idx(i4, i3, i2, i1, i0));
    }

    template <device_t pu>
    inline T* at(std::array<int64_t, N> const idx__)
    {

        if (layout_ == CblasRowMajor)
            std::reverse(std::begin(idx__), std::end(idx__));
        return at_idx<pu>(idx(idx__));
    }

    template <device_t pu =  CPU>
    /// Return total size (number of elements) of the array.
    inline size_t size() const
        {

            size_t size_{1};

            for (int i = 0; i < N; i++) {
                size_ *= dims_[i].size();
            }

            return size_;
        }

    /// Return size of particular dimension.
    inline size_t size(int i) const
    {
        mdarray_assert(i < N);
        if (layout_ == CblasRowMajor)
            return dims_[N - i - 1].size();
        else
            return dims_[i].size();
    }

    /// Return leading dimension size.
    inline uint32_t ld() const
    {
        mdarray_assert(dims_[0].size() < size_t(1 << 31));

        return (int32_t)dims_[0].size();
    }

    /// Compute hash of the array
    /** Example: printf("hash(h) : %16llX\n", h.hash()); */
    inline uint64_t hash(uint64_t h__ = 5381) const
    {
        for (size_t i = 0; i < size() * sizeof(T); i++) {
            h__ = ((h__ << 5) + h__) + ((unsigned char*)raw_ptr_)[i];
        }

        return h__;
    }

    /// Copy the content of the array to dest
    void operator>>(mdarray_base<T, N>& dest__) const
    {
        for (int i = 0; i < N; i++) {
            if (dest__.dims_[i].begin() != dims_[i].begin() ||
                dest__.dims_[i].end() != dims_[i].end()) {
                printf("error at line %i of file %s: array dimensions don't match\n",
                       __LINE__, __FILE__);
                raise(SIGTERM);
                exit(-1);
            }
        }
        std::memcpy(dest__.raw_ptr_, raw_ptr_, size() * sizeof(T));
    }

    /// Copy n elements starting from idx0.
    template <memory_t from__, memory_t to__>
    inline void copy(size_t idx0__, size_t n__, int stream_id__ = -1)
    {
#ifdef HAVE_CUDA
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);
        mdarray_assert(idx0__ + n__ <= size());

        if ((from__ & memory_t::host) == memory_t::host &&
            (to__ & memory_t::device) == memory_t::device) {
            if (stream_id__ == -1) {
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__);
            } else {
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__,
                            stream_id__);
            }
        }

        if ((from__ & memory_t::device) == memory_t::device &&
            (to__ & memory_t::host) == memory_t::host) {
            if (stream_id__ == -1) {
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__);
            } else {
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__,
                             stream_id__);
            }
        }
#else
        (void)idx0__; (void)n__; (void)stream_id__; /* unused */
#endif
    }

    template <memory_t from__, memory_t to__>
    inline void copy(size_t n__)
    {
        copy<from__, to__>(0, n__);
    }

    template <memory_t from__, memory_t to__>
    inline void async_copy(size_t n__, int stream_id__)
    {
        copy<from__, to__>(0, n__, stream_id__);
    }

    template <memory_t from__, memory_t to__>
    inline void copy()
    {
        copy<from__, to__>(0, size());
    }

    template <memory_t from__, memory_t to__>
    inline void async_copy(int stream_id__)
    {
        copy<from__, to__>(0, size(), stream_id__);
    }

    inline void retrieve(T *dst)
        {
            if (dst == this->at<CPU>())
                return;

            mdarray_assert(dst != nullptr);

            memcpy (dst,
                    this->at<CPU>(),
                    sizeof(T) * this->size());
        }

    inline void store(const T *src)
        {
            if (src == this->at< CPU>())
                return;
            mdarray_assert(src != nullptr);
            memcpy (this->at<CPU>(),
                    src,
                    sizeof(double) * this->size());
        }

    /// Zero n elements starting from idx0.
    template <memory_t mem_type__>
    inline void zero(size_t idx0__, size_t n__)
    {
        mdarray_assert(idx0__ + n__ <= size());
        if (((mem_type__ & memory_t::host) == memory_t::host) && n__) {
            mdarray_assert(raw_ptr_ != nullptr);
            std::memset(reinterpret_cast<void*>(&raw_ptr_[idx0__]), 0, n__ * sizeof(T));
        }
#ifdef HAVE_CUDA
        if (((mem_type__ & memory_t::device) == memory_t::device) && on_device() &&
            n__) {
            mdarray_assert(raw_ptr_device_ != nullptr);
            acc::zero(&raw_ptr_device_[idx0__], n__);
        }
#endif
    }

    template <memory_t mem_type__ = memory_t::host>
    inline void zero()
    {
        zero<mem_type__>(0, size());
    }

    inline bool on_device() const
    {
#ifdef HAVE_CUDA
        return (raw_ptr_device_ != nullptr);
#else
        return false;
#endif
    }
};

/// Multidimensional array with the column-major (Fortran) order.
template <typename T, int N, CBLAS_LAYOUT format = CblasColMajor>
class mdarray : public mdarray_base<T, N, format>
{
  public:
    mdarray()
    {
    }

    mdarray(std::array<int64_t, N> const& shape,
            memory_t memory__ = memory_t::host, std::string label__ = "")
    {
        this->label_  = label__;
        this->layout_ = format;
        this->init_dimensions(shape);
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            memory_t memory__ = memory_t::host, std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_  = label__;
        this->layout_ = format;
        this->init_dimensions({d0});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            memory_t memory__ = memory_t::host, std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_  = label__;
        this->layout_ = format;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1}});
        else
            this->init_dimensions({{d1, d0}});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            memory_t memory__ = memory_t::host, std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_  = label__;
        this->layout_ = format;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2}});
        else
            this->init_dimensions({{d2, d1, d0}});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            memory_t memory__ = memory_t::host, std::string label__ = "")
    {
        static_assert(N == 4, "wrong number of dimensions");

        this->label_  = label__;
        this->layout_ = format;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2, d3}});
        else
            this->init_dimensions({{d3, d2, d1, d0}});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            memory_t memory__ = memory_t::host, std::string label__ = "")
    {
        static_assert(N == 5, "wrong number of dimensions");

        this->label_  = label__;
        this->layout_ = format;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2, d3, d4}});
        else
            this->init_dimensions({{d4, d3, d2, d1, d0}});
        this->allocate(memory__);
    }

    mdarray(T* ptr__, std::array<int64_t, N> const& shape,
            std::string label__ = "")
    {
        this->layout_ = format;
        this->label_  = label__;
        this->init_dimensions(shape);
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__, mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__, T* ptr_device__, mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
#ifdef HAVE_CUDA
        this->raw_ptr_device_ = ptr_device__;
#else
        (void)ptr_device__; /* unused */
#endif
    }

    mdarray(T* ptr__, mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1, std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1}});
        else
            this->init_dimensions({{d1, d0}});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__, T* ptr_device__, mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1, std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1}});
        else
            this->init_dimensions({{d1, d0}});
        this->raw_ptr_ = ptr__;
#ifdef HAVE_CUDA
        this->raw_ptr_device_ = ptr_device__;
#else
        (void)ptr_device__; /* unused */
#endif
    }

    mdarray(T* ptr__, mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2, std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2}});
        else
            this->init_dimensions({{d2, d1, d0}});

        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__, T* ptr_device__, mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2, std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2}});
        else
            this->init_dimensions({{d2, d1, d0}});
        this->raw_ptr_ = ptr__;
#ifdef HAVE_CUDA
        this->raw_ptr_device_ = ptr_device__;
#else
        (void)ptr_device__; /* unused */
#endif
    }

    mdarray(T* ptr__, mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3, std::string label__ = "")
    {
        static_assert(N == 4, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2, d3}});
        else
            this->init_dimensions({{d3, d2, d1, d0}});

        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__, mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4, std::string label__ = "")
    {
        static_assert(N == 5, "wrong number of dimensions");
        this->layout_ = format;
        this->label_  = label__;
        if (this->layout_ == CblasColMajor)
            this->init_dimensions({{d0, d1, d2, d3, d4}});
        else
            this->init_dimensions({{d4, d3, d2, d1, d0}});
        this->raw_ptr_ = ptr__;
    }

    // mdarray<T, N, format>& operator=(std::function<T(int64_t)> f__)
    // {
    //     static_assert(N == 1, "wrong number of dimensions");

    //     for (int64_t i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end();
    //     i0++) {
    //         (*this)(i0) = f__(i0);
    //     }
    //     return *this;
    // }

    // mdarray<T, N, format>& operator=(std::function<T(int64_t, int64_t)> f__)
    // {
    //     static_assert(N == 2, "wrong number of dimensions");

    //     for (int64_t i1 = this->dims_[1].begin(); i1 <= this->dims_[1].end();
    //     i1++) {
    //         for (int64_t i0 = this->dims_[0].begin(); i0 <=
    //         this->dims_[0].end(); i0++) {
    //             (*this)(i0, i1) = f__(i0, i1);
    //         }
    //     }
    //     return *this;
    // }
};

// Alias for matrix
template <typename T>
using matrix = mdarray<T, 2>;

/// Serialize to std::ostream
template <typename T, int N, CBLAS_LAYOUT format>
std::ostream& operator<<(std::ostream& out, mdarray<T, N>& v)
{
    if (v.size()) {
        out << v[0];
        for (size_t i = 1; i < v.size(); i++) {
            out << std::string(" ") << v[i];
        }
    }
    return out;
}

#endif // __MDARRAY_HPP__

