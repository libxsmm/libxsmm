# ROOTDIR avoid abspath to match Makefile targets
ROOTDIR = $(subst //,$(NULL),$(dir $(firstword $(MAKEFILE_LIST)))/)
INCDIR = include
SCRDIR = scripts
TSTDIR = tests
BLDDIR = build
SRCDIR = src
OUTDIR = lib
BINDIR = bin
SPLDIR = samples
DOCDIR = documentation

# subdirectories (relative) to PREFIX (install targets)
PINCDIR ?= $(INCDIR)
PSRCDIR ?= $(PINCDIR)/libxsmm
POUTDIR ?= $(OUTDIR)
PPKGDIR ?= $(OUTDIR)
PMODDIR ?= $(OUTDIR)
PBINDIR ?= $(BINDIR)
PTSTDIR ?= $(TSTDIR)
PDOCDIR ?= share/libxsmm
LICFDIR ?= $(PDOCDIR)
LICFILE ?= LICENSE.md

# initial default flags: RPM_OPT_FLAGS are usually NULL
CFLAGS = $(RPM_OPT_FLAGS)
CXXFLAGS = $(RPM_OPT_FLAGS)
FCFLAGS = $(RPM_OPT_FLAGS)
DFLAGS = -DLIBXSMM_BUILD

# THRESHOLD problem size (M x N x K) determining when to use BLAS
# A value of zero (0) populates a default threshold
THRESHOLD ?= 0

# Generates M,N,K-combinations for each comma separated group e.g., "1, 2, 3" generates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified separately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# Enable thread-local cache of recently dispatched kernels either
# 0: "disable", 1: "enable", or small power-of-two number.
CACHE ?= 1

# Issue software prefetch instructions (see end of section
# https://github.com/hfp/libxsmm/#generator-driver)
# Use the enumerator 1...6, or the exact strategy
# name pfsigonly...AL2_BL2viaC.
# 0: no prefetch (nopf)
# 1: auto-select
# 2: pfsigonly
# 3: BL2viaC
# 4: curAL2
# 7: curAL2_BL2viaC
# 5: AL2
# 6: AL2_BL2viaC
PREFETCH ?= 1

# Preferred precision when registering statically generated code versions
# 0: SP and DP code versions to be registered
# 1: SP only
# 2: DP only
PRECISION ?= 0

# Specify the size of a cacheline (Bytes)
CACHELINE ?= 64

# Alpha argument of GEMM
# Supported: 1.0
ALPHA ?= 1
ifneq (1,$(ALPHA))
  $(info --------------------------------------------------------------------------------)
  $(error ALPHA needs to be 1)
endif

# Beta argument of GEMM
# Supported: 0.0, 1.0
# 0: C  = A * B
# 1: C += A * B
BETA ?= 1
ifneq (1,$(BETA))
ifneq (0,$(BETA))
  $(info --------------------------------------------------------------------------------)
  $(error BETA needs to be either 0 or 1)
endif
endif

# Determines if the library is thread-safe
THREADS ?= 1

# 0: shared libraries files suitable for dynamic linkage
# 1: library archives suitable for static linkage
STATIC ?= 1

# 0: link all dependencies as specified for the target
# 1: attempt to avoid dependencies if not referenced
ASNEEDED ?= 0

# 0: build according to the value of STATIC
# 1: build according to STATIC=0 and STATIC=1
SHARED ?= 0

# -1: support intercepted malloc (default: disabled at runtime)
#  0: disable intercepted malloc at compile-time
# >0: enable intercepted malloc
MALLOC ?= -1

# Determines the kind of routine called for intercepted GEMMs
# >=1 and odd : sequential and non-tiled (small problem sizes only)
# >=2 and even: parallelized and tiled (all problem sizes)
# >=3 and odd : GEMV is intercepted; small problem sizes
# >=4 and even: GEMV is intercepted; all problem sizes
# negative: BLAS provides DGEMM_BATCH and SGEMM_BATCH
# 0: disabled
WRAP ?= 1

# JIT backend is enabled by default
ifeq (,$(filter-out 0,$(PLATFORM)))
  JIT ?= 1
else # disabled if platform is forced
# enable: make PLATFORM=1 JIT=1
  JIT ?= 0
endif

# TRACE facility
INSTRUMENT ?= $(TRACE)

# target library for a broad range of systems
ifneq (0,$(JIT))
  SSE ?= 1
endif

ifneq (,$(MAXTARGET))
  DFLAGS += -DLIBXSMM_MAXTARGET=$(MAXTARGET)
endif

# Profiling JIT code using Linux Perf
# PERF=0: disabled (default)
# PERF=1: enabled (without JITDUMP)
# PERF=2: enabled (with JITDUMP)
#
# Additional support for jitdump
# JITDUMP=0: disabled (default)
# JITDUMP=1: enabled
# PERF=2: enabled
#
ifneq (,$(PERF))
ifneq (0,$(PERF))
ifneq (1,$(PERF))
  JITDUMP ?= 1
endif
endif
endif
JITDUMP ?= 0

ifneq (0,$(JITDUMP))
  PERF ?= 1
endif

PERF ?= 0
ifneq (0,$(PERF))
  SYM ?= 1
endif

# OpenMP is disabled by default and LIBXSMM is
# always agnostic wrt the threading runtime
OMP ?= 0

ifneq (,$(MKL))
ifneq (0,$(MKL))
  BLAS = $(MKL)
endif
endif

ifneq (1,$(CACHE))
  DFLAGS += -DLIBXSMM_CAPACITY_CACHE=$(CACHE)
endif

# disable lazy initialization and rely on ctor attribute
ifeq (0,$(INIT))
  DFLAGS += -DLIBXSMM_CTOR
endif

# Kind of documentation (internal key)
DOCEXT = pdf

# Timeout when downloading documentation parts
TIMEOUT = 30

# state to be excluded from tracking the (re-)build state
EXCLUDE_STATE = \
  DESTDIR PREFIX BINDIR CURDIR DOCDIR DOCEXT INCDIR LICFDIR OUTDIR TSTDIR TIMEOUT \
  PBINDIR PINCDIR POUTDIR PPKGDIR PMODDIR PSRCDIR PTSTDIR PDOCDIR SCRDIR SPLDIR \
  SRCDIR TEST VERSION_STRING DEPSTATIC ALIAS_% BLAS %_TARGET %ROOT MPSS KNC

# fixed .state file directory (included by source)
DIRSTATE = $(OUTDIR)/..

ifeq (,$(M)$(N)$(K))
ifneq (,$(filter 0,$(MNK) 0))
  EXCLUDE_STATE += PRECISION MNK M N K
endif
endif

# avoid to link with C++ standard library
FORCE_CXX = 0

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(BLDDIR))
IFLAGS += -I$(call quote,$(ROOTDIR)/$(SRCDIR))

# Version numbers according to interface (version.txt)
ifneq (,$(PYTHON))
  VERSION_MAJOR ?= $(shell $(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py 1)
  VERSION_MINOR ?= $(shell $(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py 2)
  VERSION_UPDATE ?= $(shell $(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py 3)
else
  $(info --------------------------------------------------------------------------------)
  $(error No Python interpreter found)
endif
VERSION_STRING ?= $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_UPDATE)
VERSION_API ?= $(shell $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py 0 $(VERSION_STRING))
VERSION_RELEASE ?= HEAD
VERSION_PACKAGE ?= 1

# explicitly target all objects
ifneq (,$(strip $(SSE)$(AVX)$(MIC)))
  TGT ?= 1
endif
TGT ?= 0

ifeq (0,$(BLAS))
ifeq (0,$(STATIC))
ifneq (0,$(LNKSOFT))
ifeq (Darwin,$(UNAME))
  LDFLAGS += $(call linkopt,-U,_dgemm_)
  LDFLAGS += $(call linkopt,-U,_sgemm_)
  LDFLAGS += $(call linkopt,-U,_dgemv_)
  LDFLAGS += $(call linkopt,-U,_sgemv_)
endif
endif
endif
endif

# target library for a broad range of systems
ifneq (0,$(JIT))
ifeq (file,$(origin AVX))
  AVX_STATIC = 0
endif
endif
AVX_STATIC ?= $(AVX)

ifeq (1,$(AVX_STATIC))
  GENTARGET = snb
else ifeq (2,$(AVX_STATIC))
  GENTARGET = hsw
else ifeq (3,$(AVX_STATIC))
  ifneq (0,$(MIC))
    ifeq (2,$(MIC))
      GENTARGET = knm
    else
      GENTARGET = knl
    endif
  else
    GENTARGET = skx
  endif
else ifneq (0,$(SSE))
  GENTARGET = wsm
else
  GENTARGET = noarch
endif

ifneq (Darwin,$(UNAME))
  GENGEMM = @$(ENV) \
    LD_LIBRARY_PATH="$(OUTDIR):$${LD_LIBRARY_PATH}" \
    PATH="$(OUTDIR):$${PATH}" \
  $(BINDIR)/libxsmm_gemm_generator
else # osx
  GENGEMM = @$(ENV) \
    DYLD_LIBRARY_PATH="$(OUTDIR):$${DYLD_LIBRARY_PATH}" \
    PATH="$(OUTDIR):$${PATH}" \
  $(BINDIR)/libxsmm_gemm_generator
endif

ifneq (,$(PYTHON))
  INDICES ?= $(shell $(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py -1 $(THRESHOLD) $(words $(MNK)) $(MNK) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
endif
NINDICES = $(words $(INDICES))

SRCFILES_KERNELS = $(patsubst %,$(BLDDIR)/mm_%.c,$(INDICES))
KRNOBJS_HST = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES))
KRNOBJS_MIC = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES))

HEADERS = $(wildcard $(ROOTDIR)/$(SRCDIR)/template/*.c) $(wildcard $(ROOTDIR)/$(SRCDIR)/*.h) \
          $(ROOTDIR)/$(SRCDIR)/libxsmm_hash.c \
          $(ROOTDIR)/include/libxsmm_blocked_gemm.h \
          $(ROOTDIR)/include/libxsmm_cpuid.h \
          $(ROOTDIR)/include/libxsmm_dnn.h \
          $(ROOTDIR)/include/libxsmm_dnn_tensor.h \
          $(ROOTDIR)/include/libxsmm_dnn_convolution.h \
          $(ROOTDIR)/include/libxsmm_dnn_fusedbatchnorm.h \
          $(ROOTDIR)/include/libxsmm_dnn_fusedgroupnorm.h \
          $(ROOTDIR)/include/libxsmm_dnn_pooling.h \
          $(ROOTDIR)/include/libxsmm_dnn_fullyconnected.h \
          $(ROOTDIR)/include/libxsmm_dnn_rnncell.h \
          $(ROOTDIR)/include/libxsmm_rng.h \
          $(ROOTDIR)/include/libxsmm_frontend.h \
          $(ROOTDIR)/include/libxsmm_fsspmdm.h \
          $(ROOTDIR)/include/libxsmm_generator.h \
          $(ROOTDIR)/include/libxsmm_intrinsics_x86.h \
          $(ROOTDIR)/include/libxsmm_macros.h \
          $(ROOTDIR)/include/libxsmm_malloc.h \
          $(ROOTDIR)/include/libxsmm_math.h \
          $(ROOTDIR)/include/libxsmm_memory.h \
          $(ROOTDIR)/include/libxsmm_mhd.h \
          $(ROOTDIR)/include/libxsmm_spmdm.h \
          $(ROOTDIR)/include/libxsmm_sync.h \
          $(ROOTDIR)/include/libxsmm_timer.h \
          $(ROOTDIR)/include/libxsmm_typedefs.h
SRCFILES_LIB = $(patsubst %,$(ROOTDIR)/$(SRCDIR)/%, \
          libxsmm_main.c libxsmm_memory.c libxsmm_malloc.c libxsmm_hash.c libxsmm_math.c \
          libxsmm_sync.c libxsmm_python.c libxsmm_mhd.c libxsmm_timer.c libxsmm_perf.c \
          libxsmm_gemm.c libxsmm_xcopy.c libxsmm_blocked_gemm.c libxsmm_spmdm.c libxsmm_fsspmdm.c libxsmm_rng.c\
          libxsmm_dnn.c libxsmm_dnn_tensor.c libxsmm_dnn_convolution.c  libxsmm_dnn_elementwise.c \
          libxsmm_dnn_rnncell.c libxsmm_dnn_rnncell_forward.c libxsmm_dnn_rnncell_backward_weight_update.c \
          libxsmm_dnn_fusedbatchnorm.c libxsmm_dnn_fusedbatchnorm_forward.c libxsmm_dnn_fusedbatchnorm_backward.c \
          libxsmm_dnn_fusedgroupnorm.c libxsmm_dnn_fusedgroupnorm_forward.c libxsmm_dnn_fusedgroupnorm_backward.c \
          libxsmm_dnn_pooling.c libxsmm_dnn_pooling_forward.c libxsmm_dnn_pooling_backward.c libxsmm_dnn_convolution_forward.c \
          libxsmm_dnn_fullyconnected.c libxsmm_dnn_fullyconnected_forward.c libxsmm_dnn_fullyconnected_backward_weight_update.c \
          libxsmm_dnn_convolution_backward.c libxsmm_dnn_convolution_weight_update.c)
SRCFILES_GEN_LIB = $(patsubst %,$(ROOTDIR)/$(SRCDIR)/%,$(notdir $(wildcard $(ROOTDIR)/$(SRCDIR)/generator_*.c)) \
          libxsmm_cpuid_x86.c libxsmm_generator.c libxsmm_trace.c)

SRCFILES_GEN_GEMM_BIN = $(patsubst %,$(ROOTDIR)/$(SRCDIR)/%,libxsmm_generator_gemm_driver.c)
OBJFILES_GEN_GEMM_BIN = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_GEMM_BIN))))
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_LIB))))
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/%.o,$(basename $(notdir $(SRCFILES_LIB)))) $(BLDDIR)/mic/generator_common.o
EXTOBJS_HST  = $(BLDDIR)/intel64/libxsmm_ext.o \
               $(BLDDIR)/intel64/libxsmm_ext_xcopy.o \
               $(BLDDIR)/intel64/libxsmm_ext_blocked_gemm.o \
               $(BLDDIR)/intel64/libxsmm_ext_gemm.o
EXTOBJS_MIC  = $(BLDDIR)/mic/libxsmm_ext.o \
               $(BLDDIR)/mic/libxsmm_ext_xcopy.o \
               $(BLDDIR)/mic/libxsmm_ext_blocked_gemm.o \
               $(BLDDIR)/mic/libxsmm_ext_gemm.o
NOBLAS_HST   = $(BLDDIR)/intel64/libxsmm_noblas.o
NOBLAS_MIC   = $(BLDDIR)/mic/libxsmm_noblas.o

# list of object might be "incomplete" if not all code gen. FLAGS are supplied with clean target!
OBJECTS = $(OBJFILES_GEN_LIB) $(OBJFILES_GEN_GEMM_BIN) $(OBJFILES_GEN_CONV_BIN) $(OBJFILES_HST) $(OBJFILES_MIC) \
          $(KRNOBJS_HST) $(KRNOBJS_MIC) $(EXTOBJS_HST) $(EXTOBJS_MIC) $(NOBLAS_HST) $(NOBLAS_MIC)
ifneq (,$(strip $(FC)))
  FTNOBJS = $(BLDDIR)/intel64/libxsmm-mod.o $(BLDDIR)/mic/libxsmm-mod.o
endif

MSGJITPROFILING = 0
ifneq (0,$(JIT))
ifneq (0,$(VTUNE))
ifeq (,$(filter Darwin,$(UNAME)))
  ifneq (0,$(PERF))
    DFLAGS += -DLIBXSMM_PERF
    ifneq (0,$(JITDUMP))
      DFLAGS += -DLIBXSMM_PERF_JITDUMP
    endif
  endif
  VTUNEROOT = $(shell env | grep VTUNE_AMPLIFIER | grep -m1 _DIR | cut -d= -f2-)
  ifeq (,$(VTUNEROOT))
    VTUNEROOT = $(EBROOTVTUNE)/vtune_amplifier
  endif
  ifneq (,$(wildcard $(VTUNEROOT)/lib64/libjitprofiling.$(SLIBEXT)))
    ifneq (0,$(SYM))
      LIBJITPROFILING = $(BLDDIR)/jitprofiling/libjitprofiling.$(SLIBEXT)
      OBJJITPROFILING = $(BLDDIR)/jitprofiling/*.o
      DFLAGS += -DLIBXSMM_VTUNE
      IFLAGS += -I$(call quote,$(VTUNEROOT)/include)
      ifneq (0,$(INTEL))
        CXXFLAGS += -diag-disable 271
        CFLAGS += -diag-disable 271
      endif
    endif
    MSGJITPROFILING = 1
  endif
endif
endif
endif

ifneq (,$(PYTHON))
information = \
	$(info ================================================================================) \
	$(info LIBXSMM $(shell $(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py) ($(UNAME))) \
	$(info --------------------------------------------------------------------------------) \
	$(info $(GINFO)) \
	$(info $(CINFO)) \
	$(if $(strip $(FC)),$(info $(FINFO)),$(NULL)) \
	$(if $(strip $(FC)),$(NULL), \
	$(if $(strip $(FC_VERSION)), \
	$(info Fortran Compiler $(FC_VERSION) is outdated!), \
	$(info Fortran Compiler is disabled or missing: no Fortran interface is built!))) \
	$(info --------------------------------------------------------------------------------)
endif

ifneq (,$(strip $(TEST)))
.PHONY: run-tests
run-tests: tests
endif

.PHONY: libxsmm
ifeq (0,$(COMPATIBLE))
ifeq (0,$(SHARED))
libxsmm: lib generator
else
libxsmm: libs generator
endif
else
ifeq (0,$(SHARED))
libxsmm: lib
else
libxsmm: libs
endif
endif
	$(information)
ifneq (,$(filter _0_,_$(LNKSOFT)_))
ifeq (0,$(DEPSTATIC))
	$(info Building a shared library requires to link against BLAS)
	$(info since a deferred choice is not implemented for this OS.)
	$(info --------------------------------------------------------------------------------)
endif
endif
ifneq (,$(filter _0_,_$(BLAS)_))
ifeq (,$(filter _0_,_$(NOBLAS)_))
	$(info BLAS dependency and fallback is removed!)
	$(info --------------------------------------------------------------------------------)
endif
else ifeq (, $(filter _0_,_$(LNKSOFT)_))
	$(info LIBXSMM is link-time agnostic with respect to a BLAS library!)
	$(info Forcing a specific library can take away a user's choice.)
	$(info If this was to solve linker errors (dgemm_, sgemm_, etc.),)
	$(info the BLAS library should go after LIBXSMM (link-line).)
	$(info --------------------------------------------------------------------------------)
endif
ifneq (2,$(INTRINSICS))
ifeq (0,$(AVX))
	$(info INTRINSICS=$(INTRINSICS) without setting AVX can reduce performance of certain code paths.)
else
	$(info INTRINSICS=$(INTRINSICS) limits LIBXSMM to AVX$(AVX) (and beyond).)
endif
ifeq (0,$(INTEL))
	$(info If adjusting INTRINSICS was necessary, reconsider an updated tool chain.)
else # Intel Compiler
	$(info Intel Compiler does not require adjusting INTRINSICS.)
endif
	$(info --------------------------------------------------------------------------------)
endif
ifneq (0,$(MSGJITPROFILING))
ifneq (,$(strip $(LIBJITPROFILING)))
	$(info Intel VTune Amplifier support has been incorporated.)
else
	$(info Intel VTune Amplifier support has been detected (enable with SYM=1).)
endif
	$(info --------------------------------------------------------------------------------)
endif

.PHONY: lib
lib: headers drytest lib_hst lib_mic

.PHONY: libs
libs: lib
ifneq (0,$(STATIC))
	@$(MAKE) --no-print-directory lib STATIC=0
else
	@$(MAKE) --no-print-directory lib STATIC=1
endif

.PHONY: all
all: libxsmm

.PHONY: realall
realall: all samples

.PHONY: headers
headers: cheader cheader_only fheader

.PHONY: header-only
header-only: cheader_only

.PHONY: header_only
header_only: header-only

.PHONY: interface
interface: headers module

.PHONY: winterface
winterface: headers sources

.PHONY: lib_mic
lib_mic: clib_mic flib_mic ext_mic noblas_mic

.PHONY: lib_hst
lib_hst: clib_hst flib_hst ext_hst noblas_hst

PREFETCH_UID = 0
PREFETCH_TYPE = 0
PREFETCH_SCHEME = nopf
ifneq (Windows_NT,$(UNAME)) # TODO: full support for Windows calling convention
  ifneq (0,$(shell echo $$((0 <= $(PREFETCH) && $(PREFETCH) <= 6))))
    PREFETCH_UID = $(PREFETCH)
  else ifneq (0,$(shell echo $$((0 > $(PREFETCH))))) # auto
    PREFETCH_UID = 1
  else ifeq (pfsigonly,$(PREFETCH))
    PREFETCH_UID = 2
  else ifeq (BL2viaC,$(PREFETCH))
    PREFETCH_UID = 3
  else ifeq (curAL2,$(PREFETCH))
    PREFETCH_UID = 4
  else ifeq (curAL2_BL2viaC,$(PREFETCH))
    PREFETCH_UID = 5
  else ifeq (AL2,$(PREFETCH))
    PREFETCH_UID = 6
  else ifeq (AL2_BL2viaC,$(PREFETCH))
    PREFETCH_UID = 7
  endif
  # Mapping build options to libxsmm_gemm_prefetch_type (see include/libxsmm_typedefs.h)
  ifeq (1,$(PREFETCH_UID))
    # Prefetch "auto" is a pseudo-strategy introduced by the frontend;
    # select "nopf" for statically generated code.
    PREFETCH_SCHEME = nopf
    PREFETCH_TYPE = -1
  else ifeq (2,$(PREFETCH_UID))
    PREFETCH_SCHEME = pfsigonly
    PREFETCH_TYPE = 1
  else ifeq (3,$(PREFETCH_UID))
    PREFETCH_SCHEME = BL2viaC
    PREFETCH_TYPE = 4
  else ifeq (4,$(PREFETCH_UID))
    PREFETCH_SCHEME = curAL2
    PREFETCH_TYPE = 8
  else ifeq (5,$(PREFETCH_UID))
    PREFETCH_SCHEME = curAL2_BL2viaC
    PREFETCH_TYPE = $(shell echo $$((4 | 8)))
  else ifeq (6,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL2
    PREFETCH_TYPE = 2
  else ifeq (7,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL2_BL2viaC
    PREFETCH_TYPE = $(shell echo $$((4 | 2)))
  endif
endif
ifeq (,$(PREFETCH_SCHEME_MIC)) # adopt host scheme
  PREFETCH_SCHEME_MIC = $(PREFETCH_SCHEME)
endif

# Mapping build options to libxsmm_gemm_flags (see include/libxsmm_typedefs.h)
#FLAGS = $(shell echo $$((((0==$(ALPHA))*4) | ((0>$(ALPHA))*8) | ((0==$(BETA))*16) | ((0>$(BETA))*32))))
FLAGS = 0

SUPPRESS_UNUSED_VARIABLE_WARNINGS = LIBXSMM_UNUSED(A); LIBXSMM_UNUSED(B); LIBXSMM_UNUSED(C);
ifneq (nopf,$(PREFETCH_SCHEME))
  #SUPPRESS_UNUSED_VARIABLE_WARNINGS += LIBXSMM_UNUSED(A_prefetch); LIBXSMM_UNUSED(B_prefetch);
  #SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXSMM_UNUSED(C_prefetch);~
  SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXSMM_UNUSED(A_prefetch); LIBXSMM_UNUSED(B_prefetch); LIBXSMM_UNUSED(C_prefetch);~
endif

# auto-clean the co-build
$(ROOTDIR)/$(SRCDIR)/template/libxsmm_config.h: $(ROOTDIR)/$(SCRDIR)/libxsmm_config.py $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py \
    $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc $(wildcard $(ROOTDIR)/.github/*) $(ROOTDIR)/version.txt
#ifneq (,$(filter-out 0 1 2 STATIC,$(words $(PRESTATE)) $(word 2,$(PRESTATE))))
ifneq (0,$(STATIC)) # static
	@rm -f $(OUTDIR)/libxsmm*.$(DLIBEXT) $(OUTDIR)/libxsmm*.$(DLIBEXT).*
else # shared/dynamic
	@rm -f $(OUTDIR)/libxsmm*.$(SLIBEXT) $(OUTDIR)/libxsmm*.$(SLIBEXT).*
endif
	@touch $@
#endif

.PHONY: config
config: $(INCDIR)/libxsmm_config.h
$(INCDIR)/libxsmm_config.h: $(INCDIR)/.make $(DIRSTATE)/.state $(ROOTDIR)/$(SRCDIR)/template/libxsmm_config.h
	$(information)
	$(info --- LIBXSMM build log)
	@if [ -e $(ROOTDIR)/.github/install.sh ]; then \
		$(ROOTDIR)/.github/install.sh; \
	fi
	@$(CP) $(filter $(ROOTDIR)/include/%.h,$(HEADERS)) $(INCDIR) 2>/dev/null || true
ifneq (,$(PYTHON))
	@$(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_config.py $(ROOTDIR)/$(SRCDIR)/template/libxsmm_config.h \
		$(MAKE_ILP64) $(OFFLOAD) $(CACHELINE) $(PRECISION) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(shell echo $$(($(THREADS)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(WRAP) $(MALLOC) $(INDICES) > $@
endif

.PHONY: cheader
cheader: $(INCDIR)/libxsmm.h
ifneq (,$(PYTHON))
$(INCDIR)/libxsmm.h: $(ROOTDIR)/$(SCRDIR)/libxsmm_interface.py \
                     $(ROOTDIR)/$(SRCDIR)/template/libxsmm.h \
                     $(INCDIR)/libxsmm_config.h $(HEADERS)
	@$(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_interface.py $(ROOTDIR)/$(SRCDIR)/template/libxsmm.h \
		$(PRECISION) $(PREFETCH_TYPE) $(INDICES) > $@
else
.PHONY: $(INCDIR)/libxsmm.h
endif

.PHONY: cheader_only
cheader_only: $(INCDIR)/libxsmm_source.h
$(INCDIR)/libxsmm_source.h: $(INCDIR)/.make $(ROOTDIR)/$(SCRDIR)/libxsmm_source.sh $(INCDIR)/libxsmm.h
	@$(ROOTDIR)/$(SCRDIR)/libxsmm_source.sh > $@

.PHONY: fheader
fheader: $(INCDIR)/libxsmm.f
ifneq (,$(PYTHON))
$(INCDIR)/libxsmm.f: $(ROOTDIR)/$(SCRDIR)/libxsmm_interface.py \
                     $(ROOTDIR)/$(SRCDIR)/template/libxsmm.f \
                     $(INCDIR)/libxsmm_config.h
	@$(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_interface.py $(ROOTDIR)/$(SRCDIR)/template/libxsmm.f \
		$(PRECISION) $(PREFETCH_TYPE) $(INDICES) | \
	$(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_config.py /dev/stdin \
		$(MAKE_ILP64) $(OFFLOAD) $(CACHELINE) $(PRECISION) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(shell echo $$(($(THREADS)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(WRAP) $(MALLOC) $(INDICES) | \
	sed "/ATTRIBUTES OFFLOAD:MIC/d" > $@
else
.PHONY: $(INCDIR)/libxsmm.f
endif

.PHONY: sources
sources: $(SRCFILES_KERNELS) $(BLDDIR)/libxsmm_dispatch.h
ifneq (,$(PYTHON))
$(BLDDIR)/libxsmm_dispatch.h: $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(SRCFILES_KERNELS) $(ROOTDIR)/$(SCRDIR)/libxsmm_dispatch.py
	@$(PYTHON) $(call quote,$(ROOTDIR)/$(SCRDIR)/libxsmm_dispatch.py) $(call qapath,$(DIRSTATE)/.state) $(PRECISION) $(THRESHOLD) $(INDICES) > $@
else
.PHONY: $(BLDDIR)/libxsmm_dispatch.h
endif

$(BLDDIR)/%.c: $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(BINDIR)/libxsmm_gemm_generator $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py $(ROOTDIR)/$(SCRDIR)/libxsmm_specialized.py
ifneq (,$(strip $(SRCFILES_KERNELS)))
	$(eval MVALUE := $(shell echo $(basename $(notdir $@)) | cut -d_ -f2))
	$(eval NVALUE := $(shell echo $(basename $(notdir $@)) | cut -d_ -f3))
	$(eval KVALUE := $(shell echo $(basename $(notdir $@)) | cut -d_ -f4))
	$(eval MNVALUE := $(MVALUE))
	$(eval NMVALUE := $(NVALUE))
	@echo "#include <libxsmm.h>" > $@
	@echo >> $@
ifeq (noarch,$(GENTARGET))
ifneq (,$(CTARGET))
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knl_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knl_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENGEMM) dense $@ libxsmm_s$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 knl $(PREFETCH_SCHEME) SP
	$(GENGEMM) dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 hsw $(PREFETCH_SCHEME) SP
	$(GENGEMM) dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 snb $(PREFETCH_SCHEME) SP
	$(GENGEMM) dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 wsm $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENGEMM) dense $@ libxsmm_d$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 knl $(PREFETCH_SCHEME) DP
	$(GENGEMM) dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 hsw $(PREFETCH_SCHEME) DP
	$(GENGEMM) dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 snb $(PREFETCH_SCHEME) DP
	$(GENGEMM) dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 wsm $(PREFETCH_SCHEME) DP
endif
endif # target
else # noarch
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENGEMM) dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 $(GENTARGET) $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENGEMM) dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 $(GENTARGET) $(PREFETCH_SCHEME) DP
endif
endif # noarch
	$(eval TMPFILE = $(shell $(MKTEMP) /tmp/.libxsmm_XXXXXX.mak))
	@cat $@ | sed \
		-e "s/void libxsmm_/LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_/" \
		-e "s/#ifndef NDEBUG/$(SUPPRESS_UNUSED_PREFETCH_WARNINGS)#ifdef LIBXSMM_NEVER_DEFINED/" \
		-e "s/#pragma message (\".*KERNEL COMPILATION ERROR in: \" __FILE__)/  $(SUPPRESS_UNUSED_VARIABLE_WARNINGS)/" \
		-e "/#error No kernel was compiled, lacking support for current architecture?/d" \
		-e "/#pragma message (\".*KERNEL COMPILATION WARNING: compiling ..* code on ..* or newer architecture: \" __FILE__)/d" \
		| tr "~" "\n" > $(TMPFILE)
ifneq (,$(PYTHON))
	@$(PYTHON) $(ROOTDIR)/$(SCRDIR)/libxsmm_specialized.py $(PRECISION) $(MVALUE) $(NVALUE) $(KVALUE) $(PREFETCH_TYPE) >> $(TMPFILE)
endif
	@$(MV) $(TMPFILE) $@
endif

define DEFINE_COMPILE_RULE
$(1): $(2) $(3) $(dir $(1))/.make
	@rm -f $(1)
	-$(CC) $(4) -c $(2) -o $(1)
	@if ! [ -e $(1) ]; then \
		echo "--------------------------------------------------------------"; \
		echo "In case of assembler error, perhaps the Binutils are outdated."; \
		echo "See https://github.com/hfp/libxsmm#outdated-binutils"; \
		echo "--------------------------------------------------------------"; \
		false; \
	fi
endef

EXTCFLAGS = -DLIBXSMM_BUILD_EXT
ifeq (0,$(OMP))
  ifeq (,$(filter environment% override command%,$(origin OMP)))
    EXTCFLAGS += $(OMPFLAG)
    EXTLDFLAGS += $(OMPLIB)
  endif
else # OpenMP
  DFLAGS += -DLIBXSMM_SYNC_OMP
endif

ifneq (0,$(MIC))
ifneq (0,$(MPSS))
$(foreach OBJ,$(OBJFILES_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(ROOTDIR)/$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h, -mmic \
  $(call applyif,$(GLIBC),libxsmm_malloc,$(OBJ),-DLIBXSMM_GLIBC) \
  $(DFLAGS) $(IFLAGS) $(CFLAGS))))
$(foreach OBJ,$(KRNOBJS_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(BLDDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, -mmic \
  $(DFLAGS) $(IFLAGS) $(CFLAGS))))
$(foreach OBJ,$(EXTOBJS_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(ROOTDIR)/$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, -mmic \
  $(DFLAGS) $(IFLAGS) $(EXTCFLAGS) $(CFLAGS))))
$(eval $(call DEFINE_COMPILE_RULE,$(NOBLAS_MIC),$(ROOTDIR)/$(SRCDIR)/libxsmm_ext.c,$(INCDIR)/libxsmm.h, -mmic \
  $(NOBLAS_CFLAGS) $(NOBLAS_FLAGS) $(NOBLAS_IFLAGS) $(DNOBLAS)))
endif
endif

# build rules that include target flags
$(eval $(call DEFINE_COMPILE_RULE,$(NOBLAS_HST),$(ROOTDIR)/$(SRCDIR)/libxsmm_ext.c,$(INCDIR)/libxsmm.h, \
  $(CTARGET) $(NOBLAS_CFLAGS) $(NOBLAS_FLAGS) $(NOBLAS_IFLAGS) $(DNOBLAS)))
$(foreach OBJ,$(OBJFILES_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTDIR)/$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h, \
  $(call applyif,$(GLIBC),libxsmm_malloc,$(OBJ),-DLIBXSMM_GLIBC) \
  $(DFLAGS) $(IFLAGS) $(CTARGET) $(CFLAGS))))
$(foreach OBJ,$(KRNOBJS_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(BLDDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(DFLAGS) $(IFLAGS) $(CTARGET) $(CFLAGS))))
$(foreach OBJ,$(EXTOBJS_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTDIR)/$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(DFLAGS) $(IFLAGS) $(CTARGET) $(EXTCFLAGS) $(CFLAGS))))

# build rules that by default include no target flags
ifneq (0,$(TGT))
  TGT_FLAGS ?= $(CTARGET)
endif
$(foreach OBJ,$(OBJFILES_GEN_LIB),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTDIR)/$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(DFLAGS) $(IFLAGS) $(TGT_FLAGS) $(CFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_GEMM_BIN),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTDIR)/$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(DFLAGS) $(IFLAGS) $(TGT_FLAGS) $(CFLAGS))))

.PHONY: compile_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
compile_mic:
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) -mmic -c $< -o $@
endif
endif

.PHONY: compile_hst
compile_hst:
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: module_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
module_mic: $(INCDIR)/mic/libxsmm.mod
$(BLDDIR)/mic/libxsmm-mod.o: $(BLDDIR)/mic/.make $(INCDIR)/mic/.make $(INCDIR)/libxsmm.f
	$(FC) $(DFLAGS) $(IFLAGS) $(FCMTFLAGS) $(FCFLAGS) -mmic -c $(INCDIR)/libxsmm.f -o $@ $(FMFLAGS) $(INCDIR)/mic
$(INCDIR)/mic/libxsmm.mod: $(BLDDIR)/mic/libxsmm-mod.o
	@if [ -e $(BLDDIR)/mic/LIBXSMM.mod ]; then $(CP) $(BLDDIR)/mic/LIBXSMM.mod $(INCDIR); fi
	@if [ -e $(BLDDIR)/mic/libxsmm.mod ]; then $(CP) $(BLDDIR)/mic/libxsmm.mod $(INCDIR); fi
	@if [ -e LIBXSMM.mod ]; then $(MV) LIBXSMM.mod $(INCDIR); fi
	@if [ -e libxsmm.mod ]; then $(MV) libxsmm.mod $(INCDIR); fi
	@touch $@
else
.PHONY: $(BLDDIR)/mic/libxsmm-mod.o
.PHONY: $(INCDIR)/mic/libxsmm.mod
endif
else
.PHONY: $(BLDDIR)/mic/libxsmm-mod.o
.PHONY: $(INCDIR)/mic/libxsmm.mod
endif
else
.PHONY: $(BLDDIR)/mic/libxsmm-mod.o
.PHONY: $(INCDIR)/mic/libxsmm.mod
endif

.PHONY: module_hst
ifneq (,$(strip $(FC)))
module_hst: $(INCDIR)/libxsmm.mod
$(BLDDIR)/intel64/libxsmm-mod.o: $(BLDDIR)/intel64/.make $(INCDIR)/libxsmm.f
	$(FC) $(DFLAGS) $(IFLAGS) $(FCMTFLAGS) $(FCFLAGS) $(FTARGET) -c $(INCDIR)/libxsmm.f -o $@ $(FMFLAGS) $(INCDIR)
$(INCDIR)/libxsmm.mod: $(BLDDIR)/intel64/libxsmm-mod.o
	@if [ -e $(BLDDIR)/intel64/LIBXSMM.mod ]; then $(CP) $(BLDDIR)/intel64/LIBXSMM.mod $(INCDIR); fi
	@if [ -e $(BLDDIR)/intel64/libxsmm.mod ]; then $(CP) $(BLDDIR)/intel64/libxsmm.mod $(INCDIR); fi
	@if [ -e LIBXSMM.mod ]; then $(MV) LIBXSMM.mod $(INCDIR); fi
	@if [ -e libxsmm.mod ]; then $(MV) libxsmm.mod $(INCDIR); fi
	@touch $@
else
.PHONY: $(BLDDIR)/intel64/libxsmm-mod.o
.PHONY: $(INCDIR)/libxsmm.mod
endif

.PHONY: module
module: module_hst module_mic

.PHONY: build_generator_lib
build_generator_lib: $(OUTDIR)/libxsmmgen.$(LIBEXT)
$(OUTDIR)/libxsmmgen.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_GEN_LIB) $(OUTDIR)/module
ifeq (0,$(STATIC))
	$(LIB_LD) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(OBJFILES_GEN_LIB) $(call cleanld,$(NOBLAS_LDFLAGS) $(NOBLAS_CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(OBJFILES_GEN_LIB)
endif

.PHONY: generator
generator: $(BINDIR)/libxsmm_gemm_generator
$(BINDIR)/libxsmm_gemm_generator: $(BINDIR)/.make $(OBJFILES_GEN_GEMM_BIN) $(OUTDIR)/libxsmmgen.$(LIBEXT)
	$(LD) -o $@ $(OBJFILES_GEN_GEMM_BIN) $(call abslib,$(OUTDIR)/libxsmmgen.$(ILIBEXT)) \
		$(call cleanld,$(NOBLAS_LDFLAGS) $(NOBLAS_CLDFLAGS))

ifneq (,$(strip $(LIBJITPROFILING)))
$(LIBJITPROFILING): $(BLDDIR)/jitprofiling/.make
	@$(CP) $(VTUNEROOT)/lib64/libjitprofiling.$(SLIBEXT) $(BLDDIR)/jitprofiling
	@cd $(BLDDIR)/jitprofiling; $(AR) x libjitprofiling.$(SLIBEXT)
endif

.PHONY: clib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
clib_mic: $(OUTDIR)/mic/libxsmm.$(LIBEXT)
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OUTDIR)/mic/.make $(OBJFILES_MIC) $(KRNOBJS_MIC)
ifeq (0,$(STATIC))
	$(LIB_LD) -mmic $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(OBJFILES_MIC) $(KRNOBJS_MIC) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(OBJFILES_MIC) $(KRNOBJS_MIC)
endif
endif
endif

.PHONY: clib_hst
clib_hst: $(OUTDIR)/libxsmm.pc
$(OUTDIR)/libxsmm.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(LIBJITPROFILING)
ifeq (0,$(STATIC))
	$(LIB_LD) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(LIBJITPROFILING) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(OBJJITPROFILING)
endif

.PHONY: flib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
flib_mic: $(OUTDIR)/mic/libxsmmf.$(LIBEXT)
$(OUTDIR)/mic/libxsmmf.$(LIBEXT): $(INCDIR)/mic/libxsmm.mod $(OUTDIR)/mic/libxsmm.$(LIBEXT)
ifeq (0,$(STATIC))
	$(LIB_FLD) -mmic $(FCMTFLAGS) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(BLDDIR)/mic/libxsmm-mod.o $(call abslib,$(OUTDIR)/mic/libxsmm.$(ILIBEXT)) $(call cleanld,$(LDFLAGS) $(FLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(BLDDIR)/mic/libxsmm-mod.o
endif
else
.PHONY: $(OUTDIR)/mic/libxsmmf.$(LIBEXT)
endif
endif
endif

.PHONY: flib_hst
ifneq (,$(strip $(FC)))
flib_hst: $(OUTDIR)/libxsmmf.pc
$(OUTDIR)/libxsmmf.$(LIBEXT): $(INCDIR)/libxsmm.mod $(OUTDIR)/libxsmm.$(LIBEXT)
ifeq (0,$(STATIC))
	$(LIB_FLD) $(FCMTFLAGS) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(BLDDIR)/intel64/libxsmm-mod.o $(call abslib,$(OUTDIR)/libxsmm.$(ILIBEXT)) $(call cleanld,$(LDFLAGS) $(FLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(BLDDIR)/intel64/libxsmm-mod.o
endif
else
.PHONY: $(OUTDIR)/libxsmmf.pc
endif

.PHONY: ext_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ext_mic: $(OUTDIR)/mic/libxsmmext.$(LIBEXT)
$(OUTDIR)/mic/libxsmmext.$(LIBEXT): $(EXTOBJS_MIC) $(OUTDIR)/mic/libxsmm.$(LIBEXT)
ifeq (0,$(STATIC))
	$(LIB_LD) -mmic $(EXTLDFLAGS) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(EXTOBJS_MIC) $(call abslib,$(OUTDIR)/mic/libxsmm.$(ILIBEXT)) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(EXTOBJS_MIC)
endif
endif
endif

.PHONY: ext_hst
ext_hst: $(OUTDIR)/libxsmmext.pc
$(OUTDIR)/libxsmmext.$(LIBEXT): $(OUTDIR)/libxsmm.$(LIBEXT) $(EXTOBJS_HST)
ifeq (0,$(STATIC))
	$(LIB_LD) $(EXTLDFLAGS) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(EXTOBJS_HST) $(call abslib,$(OUTDIR)/libxsmm.$(ILIBEXT)) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(EXTOBJS_HST)
endif

.PHONY: noblas_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
noblas_mic: $(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT)
$(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT): $(NOBLAS_MIC)
ifeq (0,$(STATIC))
	$(LIB_LD) -mmic $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(NOBLAS_MIC) $(call cleanld,$(NOBLAS_LDFLAGS) $(NOBLAS_CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(NOBLAS_MIC)
endif
endif
endif

.PHONY: noblas_hst
noblas_hst: $(OUTDIR)/libxsmmnoblas.pc
$(OUTDIR)/libxsmmnoblas.$(LIBEXT): $(NOBLAS_HST)
ifeq (0,$(STATIC))
	$(LIB_LD) $(call solink,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(NOBLAS_HST) $(call cleanld,$(NOBLAS_LDFLAGS) $(NOBLAS_CLDFLAGS))
else # static
	@rm -f $@
	$(AR) -rs $@ $(NOBLAS_HST)
endif

# use dir not qdir to avoid quotes; also $(ROOTDIR)/$(SPLDIR) is relative
DIRS_SAMPLES = $(dir $(shell find $(ROOTDIR)/$(SPLDIR) -type f -name Makefile \
	| grep -v /deeplearning/tf_lstm_ops/ \
	| grep -v /deeplearning/gxm/ \
	| grep -v /edge/repro/ \
	| grep -v /packed/ \
	| grep -v /pyfr/ \
	$(NULL)))

.PHONY: samples $(DIRS_SAMPLES)
samples: $(DIRS_SAMPLES)
$(DIRS_SAMPLES): lib_hst
	@$(FLOCK) $@ "$(MAKE) DEPSTATIC=$(STATIC)"

.PHONY: cp2k cp2k_mic
cp2k: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"
cp2k_mic: lib_mic
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

.PHONY: wrap wrap_mic
wrap: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/utilities/wrap "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) TRACE=0"
wrap_mic: lib_mic
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/utilities/wrap "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1 TRACE=0"

.PHONY: nek nek_mic
nek: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"
nek_mic: lib_mic
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

.PHONY: smm smm_mic
smm: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/smm "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"
smm_mic: lib_mic
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/smm "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

# added for specfem sample
# will need option: make MNK="5 25" ..
.PHONY: specfem specfem_mic
specfem: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/specfem "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"
specfem_mic: lib_mic
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/specfem "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

.PHONY: drytest
drytest: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.sh $(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.sh \
	$(ROOTDIR)/$(SPLDIR)/nek/axhm-perf.sh $(ROOTDIR)/$(SPLDIR)/nek/grad-perf.sh $(ROOTDIR)/$(SPLDIR)/nek/rstr-perf.sh

$(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.sh: $(ROOTDIR)/$(SPLDIR)/cp2k/.make $(ROOTDIR)/Makefile
	@echo "#!/usr/bin/env sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=cp2k-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23 4_6_9 13_5_7 24_3_36\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  SIZE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "else" >> $@
	@echo "  SIZE=0" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w | tr -d ' ')" >> $@
	@echo "for RUN in \$${RUNS}; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 echo -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/cp2k-dbcsr.sh \$${MVALUE} \$${SIZE} 0 \$${NVALUE} \$${KVALUE} >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    echo \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    echo \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.sh: $(ROOTDIR)/$(SPLDIR)/smm/.make $(ROOTDIR)/Makefile
	@echo "#!/usr/bin/env sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/smmf-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23 4_6_9 13_5_7 24_3_36\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w | tr -d ' ')" >> $@
	@echo "for RUN in \$${RUNS}; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 echo -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/smm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    echo \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    echo \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(ROOTDIR)/$(SPLDIR)/nek/axhm-perf.sh: $(ROOTDIR)/$(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/usr/bin/env sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/axhm-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_6_9 8_8_8 13_13_13 16_8_13\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w | tr -d ' ')" >> $@
	@echo "for RUN in \$${RUNS}; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 echo -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/axhm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    echo \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    echo \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(ROOTDIR)/$(SPLDIR)/nek/grad-perf.sh: $(ROOTDIR)/$(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/usr/bin/env sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/grad-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_6_9 8_8_8 13_13_13 16_8_13\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w | tr -d ' ')" >> $@
	@echo "for RUN in \$${RUNS}; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 echo -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/grad.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    echo \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    echo \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(ROOTDIR)/$(SPLDIR)/nek/rstr-perf.sh: $(ROOTDIR)/$(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/usr/bin/env sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/rstr-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
	@echo "RUNT=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_4_4 8_8_8\"" >> $@
	@echo "RUNT=\"7_7_7 10_10_10\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NRUNS=\$$(echo \$${RUNS} | wc -w | tr -d ' ')" >> $@
	@echo "NRUNT=\$$(echo \$${RUNT} | wc -w | tr -d ' ')" >> $@
	@echo "NMAX=\$$((NRUNS*NRUNT))" >> $@
	@echo "for RUN1 in \$${RUNS}; do" >> $@
	@echo "  for RUN2 in \$${RUNT}; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN1} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN1} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN1} | cut -d_ -f3)" >> $@
	@echo "  MMVALUE=\$$(echo \$${RUN2} | cut -d_ -f1)" >> $@
	@echo "  NNVALUE=\$$(echo \$${RUN2} | cut -d_ -f2)" >> $@
	@echo "  KKVALUE=\$$(echo \$${RUN2} | cut -d_ -f3)" >> $@
	@echo "  >&2 echo -n \"\$${NRUN} of \$${NMAX} (MNK=\$${MVALUE}x\$${NVALUE}x\$${KVALUE} MNK2=\$${MMVALUE}x\$${NNVALUE}x\$${KKVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/rstr.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${MMVALUE} \$${NNVALUE} \$${KKVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    echo \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    echo \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

.PHONY: test
test: tests

.PHONY: perf
perf: perf-cp2k

.PHONY: test-all
test-all: tests test-cp2k test-smm test-nek test-wrap

.PHONY: build-tests
build-tests: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(TSTDIR) "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"

.PHONY: tests
tests: lib_hst
	@$(FLOCK) $(ROOTDIR)/$(TSTDIR) "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) test"

.PHONY: cpp-test
cpp-test: test-cpp

.PHONY: test-cpp
test-cpp: $(INCDIR)/libxsmm_source.h
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) TRACE=0 \
		ECXXFLAGS='-DUSE_HEADER_ONLY $(ECXXFLAGS)' clean compile"

.PHONY: test-cp2k
test-cp2k: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-test.txt
$(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-test.txt: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.sh lib_hst cp2k
	@$(FLOCK) $(call qdir,$@) "./cp2k-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE) * 128)))"

.PHONY: perf-cp2k
perf-cp2k: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.txt
$(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.txt: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.sh lib_hst cp2k
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/cp2k "./cp2k-perf.sh $(call qndir,$@)"

.PHONY: test-wrap
test-wrap: wrap
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/utilities/wrap "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) TRACE=0 test"

.PHONY: test-smm
ifneq (,$(strip $(FC)))
test-smm: $(ROOTDIR)/$(SPLDIR)/smm/smm-test.txt
$(ROOTDIR)/$(SPLDIR)/smm/smm-test.txt: $(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.sh lib_hst smm
	@$(FLOCK) $(call qdir,$@) "./smmf-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE) * -128)))"
endif

.PHONY: perf-smm
ifneq (,$(strip $(FC)))
perf-smm: $(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.txt
$(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.txt: $(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.sh lib_hst smm
	@$(FLOCK) $(call qdir,$@) "./smmf-perf.sh $(call qndir,$@)"
endif

.PHONY: test-nek
ifneq (,$(strip $(FC)))
test-nek: \
	$(ROOTDIR)/$(SPLDIR)/nek/axhm-perf.txt \
	$(ROOTDIR)/$(SPLDIR)/nek/grad-perf.txt \
	$(ROOTDIR)/$(SPLDIR)/nek/rstr-perf.txt
$(ROOTDIR)/$(SPLDIR)/nek/axhm-perf.txt: $(ROOTDIR)/$(SPLDIR)/nek/axhm-perf.sh lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) axhm"
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "./axhm-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE) * -128)))"
$(ROOTDIR)/$(SPLDIR)/nek/grad-perf.txt: $(ROOTDIR)/$(SPLDIR)/nek/grad-perf.sh lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) grad"
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "./grad-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE) * -128)))"
$(ROOTDIR)/$(SPLDIR)/nek/rstr-perf.txt: $(ROOTDIR)/$(SPLDIR)/nek/rstr-perf.sh lib_hst
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) rstr"
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/nek "./rstr-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE) * -128)))"
endif

$(DOCDIR)/index.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/README.md
	@sed $(ROOTDIR)/README.md \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/\[\[..*\](..*)\]//g' \
		-e "s/](${DOCDIR}\//](/g" \
		-e 'N;/^\n$$/d;P;D' \
		> $@

$(DOCDIR)/libxsmm.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/documentation/index.md \
$(ROOTDIR)/documentation/libxsmm_mm.md $(ROOTDIR)/documentation/libxsmm_dl.md $(ROOTDIR)/documentation/libxsmm_aux.md \
$(ROOTDIR)/documentation/libxsmm_prof.md $(ROOTDIR)/documentation/libxsmm_tune.md $(ROOTDIR)/documentation/libxsmm_be.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/documentation/.libxsmm_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@cd $(ROOTDIR)/documentation && ( \
		iconv -t utf-8 index.md && echo && \
		echo "# LIBXSMM Domains" && \
		iconv -t utf-8 libxsmm_mm.md && echo && \
		iconv -t utf-8 libxsmm_dl.md && echo && \
		iconv -t utf-8 libxsmm_aux.md && echo && \
		iconv -t utf-8 libxsmm_prof.md && echo && \
		iconv -t utf-8 libxsmm_tune.md && echo && \
		iconv -t utf-8 libxsmm_be.md && echo && \
		echo "# Appendix" && \
		echo "## Compatibility" && \
		wget -T $(TIMEOUT) -q -O - https://raw.githubusercontent.com/wiki/hfp/libxsmm/Compatibility.md 2>/dev/null && echo && \
		echo "## Validation" && \
		wget -T $(TIMEOUT) -q -O - https://raw.githubusercontent.com/wiki/hfp/libxsmm/Validation.md 2>/dev/null; ) \
	| sed \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(call qndir,$(TMPFILE)) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSMM Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(call qndir,$@)
	@rm $(TMPFILE)

$(DOCDIR)/libxsmm_samples.md: $(ROOTDIR)/Makefile $(ROOTDIR)/$(SPLDIR)/*/README.md $(ROOTDIR)/$(SPLDIR)/deeplearning/*/README.md $(ROOTDIR)/$(SPLDIR)/utilities/*/README.md
	@cat $(ROOTDIR)/$(SPLDIR)/*/README.md $(ROOTDIR)/$(SPLDIR)/deeplearning/*/README.md $(ROOTDIR)/$(SPLDIR)/utilities/*/README.md \
	| sed \
		-e 's/^#/##/' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
		-e '1s/^/# [LIBXSMM Samples](https:\/\/github.com\/hfp\/libxsmm\/raw\/master\/documentation\/libxsmm_samples.pdf)\n\n/' \
		> $@

$(DOCDIR)/libxsmm_samples.$(DOCEXT): $(ROOTDIR)/documentation/libxsmm_samples.md
	$(eval TMPFILE = $(shell $(MKTEMP) .libxsmm_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@iconv -t utf-8 $(ROOTDIR)/documentation/libxsmm_samples.md \
	| pandoc \
		--template=$(TMPFILE) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSMM Sample Code Summary" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE)

$(DOCDIR)/tensorflow.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/documentation/tensorflow.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/documentation/.libxsmm_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@cd $(ROOTDIR)/documentation && iconv -t utf-8 tensorflow.md \
	| sed \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(call qndir,$(TMPFILE)) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="TensorFlow with LIBXSMM" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(call qndir,$@)
	@rm $(TMPFILE)

.PHONY: documentation
documentation: \
$(DOCDIR)/libxsmm.$(DOCEXT) \
$(DOCDIR)/libxsmm_samples.$(DOCEXT) \
$(DOCDIR)/tensorflow.$(DOCEXT)

.PHONY: mkdocs
mkdocs: $(ROOTDIR)/documentation/index.md $(ROOTDIR)/documentation/libxsmm_samples.md
	@mkdocs build --clean
	@mkdocs serve

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(HEREDIR))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(FTNOBJS) $(SRCFILES_KERNELS) $(BLDDIR)/libxsmm_dispatch.h
	@rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif
	@find . -type f \( -name .make -or -name .state \) -exec rm {} \;
	@rm -f $(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.pyc
	@rm -rf $(ROOTDIR)/$(SCRDIR)/__pycache__

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(HEREDIR))
	@rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@rm -f $(OUTDIR)/libxsmm.$(LIBEXT)* $(OUTDIR)/mic/libxsmm.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmf.$(LIBEXT)* $(OUTDIR)/mic/libxsmmf.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmext.$(LIBEXT)* $(OUTDIR)/mic/libxsmmext.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmnoblas.$(LIBEXT)* $(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmgen.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmm*.pc
endif
ifneq ($(call qapath,$(BINDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BINDIR)),$(HEREDIR))
	@rm -rf $(BINDIR)
endif
endif
ifneq (,$(wildcard $(BINDIR))) # still exists
	@rm -f $(BINDIR)/libxsmm_*_generator
endif
	@rm -f $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(ROOTDIR)/$(SPLDIR)/smm/smmf-perf.sh
	@rm -f $(ROOTDIR)/$(SPLDIR)/nek/grad-perf.sh
	@rm -f $(ROOTDIR)/$(SPLDIR)/nek/axhm-perf.sh
	@rm -f $(ROOTDIR)/$(SPLDIR)/nek/rstr-perf.sh
	@rm -f $(INCDIR)/libxsmm_config.h
	@rm -f $(INCDIR)/libxsmm.modmic
	@rm -f $(INCDIR)/libxsmm.mod
	@rm -f $(INCDIR)/libxsmm.f
	@rm -f $(INCDIR)/libxsmm.h

.PHONY: clean-all
clean-all: clean
	@find $(ROOTDIR) -type f -name Makefile -exec $(FLOCK) {} \
		"$(MAKE) --no-print-directory clean" \; 2>/dev/null || true

.PHONY: realclean-all
realclean-all: realclean
	@find $(ROOTDIR) -type f -name Makefile -exec $(FLOCK) {} \
		"$(MAKE) --no-print-directory realclean" \; 2>/dev/null || true

.PHONY: distclean
distclean: realclean-all
	@rm -rf libxsmm*

# keep original prefix (:)
ALIAS_PREFIX := $(PREFIX)

# DESTDIR is used as prefix of PREFIX
ifneq (,$(strip $(DESTDIR)))
  override PREFIX := $(call qapath,$(DESTDIR)/$(PREFIX))
endif
# fall-back
ifeq (,$(strip $(PREFIX)))
  override PREFIX := $(HEREDIR)
endif

# setup maintainer-layout
ifeq (,$(strip $(ALIAS_PREFIX)))
  override ALIAS_PREFIX := $(PREFIX)
endif
ifneq ($(ALIAS_PREFIX),$(PREFIX))
  PPKGDIR = libdata/pkgconfig
  PMODDIR = share/modules
endif

.PHONY: install-minimal
install-minimal: libxsmm
ifneq ($(PREFIX),$(ABSDIR))
	@mkdir -p $(PREFIX)/$(POUTDIR) $(PREFIX)/$(PBINDIR) $(PREFIX)/$(PINCDIR) $(PREFIX)/$(PSRCDIR)
	@echo
	@echo "LIBXSMM installing libraries..."
	@$(CP) -va $(OUTDIR)/libxsmmnoblas.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxsmmnoblas.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -va $(OUTDIR)/libxsmmgen.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxsmmgen.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -va $(OUTDIR)/libxsmmext.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxsmmext.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -va $(OUTDIR)/libxsmmf.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxsmmf.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -va $(OUTDIR)/libxsmm.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxsmm.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@if [ -e $(OUTDIR)/mic/libxsmmnoblas.$(DLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -va $(OUTDIR)/mic/libxsmmnoblas.$(DLIBEXT)* $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmnoblas.$(SLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsmmnoblas.$(SLIBEXT) $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmext.$(DLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -va $(OUTDIR)/mic/libxsmmext.$(DLIBEXT)* $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmext.$(SLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsmmext.$(SLIBEXT) $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmf.$(DLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -va $(OUTDIR)/mic/libxsmmf.$(DLIBEXT)* $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmf.$(SLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsmmf.$(SLIBEXT) $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmm.$(DLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -va $(OUTDIR)/mic/libxsmm.$(DLIBEXT)* $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmm.$(SLIBEXT) ]; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsmm.$(SLIBEXT) $(PREFIX)/$(POUTDIR)/mic; \
	fi
	@echo
	@echo "LIBXSMM installing pkg-config and module files..."
	@mkdir -p $(PREFIX)/$(PPKGDIR)
	@$(CP) -v $(OUTDIR)/*.pc $(PREFIX)/$(PPKGDIR) 2>/dev/null || true
	@mkdir -p $(PREFIX)/$(PMODDIR)
	@$(CP) -v $(OUTDIR)/module $(PREFIX)/$(PMODDIR)/libxsmm 2>/dev/null || true
	@echo
	@echo "LIBXSMM installing stand-alone generators..."
	@$(CP) -v $(BINDIR)/libxsmm_*_generator $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@echo
	@echo "LIBXSMM installing interface..."
	@$(CP) -v $(INCDIR)/libxsmm*.h $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v $(INCDIR)/libxsmm.f $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v $(INCDIR)/*.mod* $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@echo
	@echo "LIBXSMM installing header-only..."
	@$(CP) -r $(ROOTDIR)/$(SRCDIR)/* $(PREFIX)/$(PSRCDIR) >/dev/null 2>/dev/null || true
endif

.PHONY: install
install: install-minimal
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXSMM installing documentation..."
	@mkdir -p $(PREFIX)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/$(DOCDIR)/*.pdf $(PREFIX)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/$(DOCDIR)/*.md $(PREFIX)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/SECURITY.md $(PREFIX)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/version.txt $(PREFIX)/$(PDOCDIR)
	@sed "s/^\"//;s/\\\n\"$$//;/STATIC=/d" $(DIRSTATE)/.state > $(PREFIX)/$(PDOCDIR)/build.txt 2>/dev/null || true
	@mkdir -p $(PREFIX)/$(LICFDIR)
ifneq ($(call qapath,$(PREFIX)/$(PDOCDIR)/LICENSE.md),$(call qapath,$(PREFIX)/$(LICFDIR)/$(LICFILE)))
	@$(MV) $(PREFIX)/$(PDOCDIR)/LICENSE.md $(PREFIX)/$(LICFDIR)/$(LICFILE)
endif
endif

.PHONY: install-all
install-all: install samples
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXSMM installing samples..."
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/cp2k/,cp2k cp2k.sh cp2k-perf* cp2k-plot.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/wrap/,dgemm-blas dgemm-blas.sh dgemm-wrap dgemm-wrap.sh wrap-test.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/dispatch/,dispatch dispatch.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/nek/,axhm grad rstr *.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/smm/,smm smm.sh smm-perf* smmf-perf.sh smm-plot.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/smm/,specialized specialized.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/smm/,dispatched dispatched.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/smm/,inlined inlined.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/smm/,blas blas.sh) $(PREFIX)/$(PBINDIR) 2>/dev/null || true
endif

.PHONY: install-dev
install-dev: install-all build-tests
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXSMM installing tests..."
	@mkdir -p $(PREFIX)/$(PTSTDIR)
	@$(CP) -v $(basename $(wildcard $(ROOTDIR)/$(TSTDIR)/*.c)) $(PREFIX)/$(PTSTDIR) 2>/dev/null || true
endif

.PHONY: install-artifacts
install-artifacts: install-dev
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXSMM installing artifacts..."
	@mkdir -p $(PREFIX)/$(PDOCDIR)/artifacts
	@$(CP) -v $(DIRSTATE)/.state $(PREFIX)/$(PDOCDIR)/artifacts/make.txt
endif

ifeq (Windows_NT,$(UNAME))
  ALIAS_PRIVLIBS = $(call ldlib,$(LD),$(SLDFLAGS),dbghelp)
else ifneq (Darwin,$(UNAME))
  ifneq (FreeBSD,$(UNAME))
    ALIAS_PRIVLIBS = $(LIBPTHREAD) $(LIBRT) $(LIBDL) $(LIBM) $(LIBC)
  else
    ALIAS_PRIVLIBS = $(LIBDL) $(LIBM) $(LIBC)
  endif
endif
ifneq (Darwin,$(UNAME))
  ALIAS_PRIVLIBS_EXT = -fopenmp
endif

ALIAS_INCLUDEDIR = $(subst $$$$,$(if $(findstring $$$$/,$$$$$(PINCDIR)),,\$${prefix}/),$(subst $$$$$(ALIAS_PREFIX),\$${prefix},$$$$$(PINCDIR)))
ALIAS_LIBDIR = $(subst $$$$,$(if $(findstring $$$$/,$$$$$(POUTDIR)),,\$${prefix}/),$(subst $$$$$(ALIAS_PREFIX),\$${prefix},$$$$$(POUTDIR)))

$(OUTDIR)/libxsmm.pc: $(OUTDIR)/libxsmm.$(LIBEXT)
	@echo "Name: libxsmm" > $@
	@echo "Description: Matrix operations and deep learning primitives" >> $@
	@echo "URL: https://github.com/hfp/libxsmm" >> $@
	@echo "Version: $(VERSION_STRING)" >> $@
	@echo >> $@
	@echo "prefix=$(ALIAS_PREFIX)" >> $@
	@echo "includedir=$(ALIAS_INCLUDEDIR)" >> $@
	@echo "libdir=$(ALIAS_LIBDIR)" >> $@
	@echo >> $@
	@echo "Cflags: -I\$${includedir}" >> $@
ifneq (,$(ALIAS_PRIVLIBS))
	@if [ -e $(OUTDIR)/libxsmm.$(DLIBEXT) ]; then \
		echo "Libs: -L\$${libdir} -lxsmm" >> $@; \
		echo "Libs.private: $(ALIAS_PRIVLIBS)" >> $@; \
	else \
		echo "Libs: -L\$${libdir} -lxsmm $(ALIAS_PRIVLIBS)" >> $@; \
	fi
else # no private libraries
	@echo "Libs: -L\$${libdir} -lxsmm" >> $@
endif

$(OUTDIR)/libxsmmf.pc: $(OUTDIR)/libxsmmf.$(LIBEXT)
	@echo "Name: libxsmm/f" > $@
	@echo "Description: LIBXSMM for Fortran" >> $@
	@echo "URL: https://github.com/hfp/libxsmm" >> $@
	@echo "Version: $(VERSION_STRING)" >> $@
	@echo >> $@
	@echo "prefix=$(ALIAS_PREFIX)" >> $@
	@echo "includedir=$(ALIAS_INCLUDEDIR)" >> $@
	@echo "libdir=$(ALIAS_LIBDIR)" >> $@
	@echo >> $@
	@echo "Requires: libxsmm" >> $@
	@echo "Cflags: -I\$${includedir}" >> $@
	@echo "Libs: -L\$${libdir} -lxsmmf" >> $@

$(OUTDIR)/libxsmmext.pc: $(OUTDIR)/libxsmmext.$(LIBEXT)
	@echo "Name: libxsmm/ext" > $@
	@echo "Description: LIBXSMM/multithreaded for OpenMP" >> $@
	@echo "URL: https://github.com/hfp/libxsmm" >> $@
	@echo "Version: $(VERSION_STRING)" >> $@
	@echo >> $@
	@echo "prefix=$(ALIAS_PREFIX)" >> $@
	@echo "includedir=$(ALIAS_INCLUDEDIR)" >> $@
	@echo "libdir=$(ALIAS_LIBDIR)" >> $@
	@echo >> $@
	@echo "Requires: libxsmm" >> $@
	@echo "Cflags: -I\$${includedir}" >> $@
ifneq (,$(ALIAS_PRIVLIBS_EXT))
	@if [ -e $(OUTDIR)/libxsmmext.$(DLIBEXT) ]; then \
		echo "Libs: -L\$${libdir} -lxsmmext" >> $@; \
		echo "Libs.private: $(ALIAS_PRIVLIBS_EXT)" >> $@; \
	else \
		echo "Libs: -L\$${libdir} -lxsmmext $(ALIAS_PRIVLIBS_EXT)" >> $@; \
	fi
else # no private libraries
	@echo "Libs: -L\$${libdir} -lxsmmext" >> $@
endif

$(OUTDIR)/libxsmmnoblas.pc: $(OUTDIR)/libxsmmnoblas.$(LIBEXT)
	@echo "Name: libxsmm/noblas" > $@
	@echo "Description: LIBXSMM substituted LAPACK/BLAS dependency" >> $@
	@echo "URL: https://github.com/hfp/libxsmm" >> $@
	@echo "Version: $(VERSION_STRING)" >> $@
	@echo >> $@
	@echo "prefix=$(ALIAS_PREFIX)" >> $@
	@echo "includedir=$(ALIAS_INCLUDEDIR)" >> $@
	@echo "libdir=$(ALIAS_LIBDIR)" >> $@
	@echo >> $@
	@echo "Requires: libxsmm" >> $@
	@echo "Cflags: -I\$${includedir}" >> $@
	@echo "Libs: -L\$${libdir} -lxsmmnoblas" >> $@

$(OUTDIR)/module: $(OUTDIR)/.make $(INCDIR)/libxsmm.h
	@echo "#%Module1.0" > $@
	@echo >> $@
	@echo "module-whatis \"LIBXSMM $(VERSION_STRING)\"" >> $@
	@echo >> $@
	@echo "set PREFIX \"$(ALIAS_PREFIX)\"" >> $@
	@echo "prepend-path PATH \"\$$PREFIX/bin\"" >> $@
	@echo "prepend-path LD_LIBRARY_PATH \"\$$PREFIX/lib\"" >> $@
	@echo >> $@
	@echo "prepend-path PKG_CONFIG_PATH \"\$$PREFIX/lib\"" >> $@
	@echo "prepend-path LIBRARY_PATH \"\$$PREFIX/lib\"" >> $@
	@echo "prepend-path CPATH \"\$$PREFIX/include\"" >> $@

.PHONY: deb
deb:
	@if [ "" != "$$(command -v git)" ]; then \
		VERSION_ARCHIVE=$$(git describe --tags --abbrev=0 2>/dev/null); \
		VERSION_ARCHIVE_SONAME=$$($(ROOTDIR)/$(SCRDIR)/libxsmm_utilities.py 0 $${VERSION_ARCHIVE}); \
	fi; \
	if [ "" != "$${VERSION_ARCHIVE}" ] && [ "" != "$${VERSION_ARCHIVE_SONAME}" ]; then \
		ARCHIVE_AUTHOR_NAME="$$(git config user.name)"; \
		ARCHIVE_AUTHOR_MAIL="$$(git config user.email)"; \
		ARCHIVE_NAME=libxsmm$${VERSION_ARCHIVE_SONAME}; \
		ARCHIVE_DATE="$$(LANG=C date -R)"; \
		if [ "" != "$${ARCHIVE_AUTHOR_NAME}" ] && [ "" != "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
			ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME} <$${ARCHIVE_AUTHOR_MAIL}>"; \
		else \
			echo "Warning: Please git-config user.name and user.email!"; \
			if [ "" != "$${ARCHIVE_AUTHOR_NAME}" ] || [ "" != "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
				ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME}$${ARCHIVE_AUTHOR_MAIL}"; \
			fi \
		fi; \
		if ! [ -e $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz ]; then \
			git archive --prefix $${ARCHIVE_NAME}-$${VERSION_ARCHIVE}/ \
				-o $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz $(VERSION_RELEASE); \
		fi; \
		tar xf $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz; \
		cd $${ARCHIVE_NAME}-$${VERSION_ARCHIVE}; \
		mkdir -p debian/source; cd debian/source; \
		echo "3.0 (quilt)" > format; \
		cd ..; \
		echo "Source: $${ARCHIVE_NAME}" > control; \
		echo "Section: libs" >> control; \
		echo "Homepage: https://github.com/hfp/libxsmm" >> control; \
		echo "Vcs-Git: https://github.com/hfp/libxsmm/libxsmm.git" >> control; \
		echo "Maintainer: $${ARCHIVE_AUTHOR}" >> control; \
		echo "Priority: optional" >> control; \
		echo "Build-Depends: debhelper (>= 9)" >> control; \
		echo "Standards-Version: 3.9.8" >> control; \
		echo >> control; \
		echo "Package: $${ARCHIVE_NAME}" >> control; \
		echo "Section: libs" >> control; \
		echo "Architecture: amd64" >> control; \
		echo "Depends: \$${shlibs:Depends}, \$${misc:Depends}" >> control; \
		echo "Description: Matrix operations and deep learning primitives" >> control; \
		wget -T $(TIMEOUT) -qO- https://api.github.com/repos/hfp/libxsmm \
		| sed -n 's/ *\"description\": \"\(..*\)\".*/\1/p' \
		| fold -s -w 79 | sed -e 's/^/ /' -e 's/[[:space:]][[:space:]]*$$//' >> control; \
		echo "$${ARCHIVE_NAME} ($${VERSION_ARCHIVE}-$(VERSION_PACKAGE)) UNRELEASED; urgency=low" > changelog; \
		echo >> changelog; \
		wget -T $(TIMEOUT) -qO- https://api.github.com/repos/hfp/libxsmm/releases/tags/$${VERSION_ARCHIVE} \
		| sed -n 's/ *\"body\": \"\(..*\)\".*/\1/p' \
		| sed -e 's/\\r\\n/\n/g' -e 's/\\"/"/g' -e 's/\[\([^]]*\)\]([^)]*)/\1/g' \
		| sed -n 's/^\* \(..*\)/\* \1/p' \
		| fold -s -w 78 | sed -e 's/^/  /g' -e 's/^  \* /\* /' -e 's/^/  /' -e 's/[[:space:]][[:space:]]*$$//' >> changelog; \
		echo >> changelog; \
		echo " -- $${ARCHIVE_AUTHOR}  $${ARCHIVE_DATE}" >> changelog; \
		echo "#!/usr/bin/make -f" > rules; \
		echo "export DH_VERBOSE = 1" >> rules; \
		echo >> rules; \
		echo "%:" >> rules; \
		$$(which echo) -e "\tdh \$$@" >> rules; \
		echo >> rules; \
		echo "override_dh_auto_install:" >> rules; \
		$$(which echo) -e "\tdh_auto_install -- prefix=/usr" >> rules; \
		echo >> rules; \
		echo "9" > compat; \
		$(CP) ../LICENSE.md copyright; \
		rm -f ../$(TSTDIR)/mhd_test.mhd; \
		chmod +x rules; \
		debuild \
			-e PREFIX=debian/$${ARCHIVE_NAME}/usr \
			-e PDOCDIR=share/doc/$${ARCHIVE_NAME} \
			-e LICFILE=copyright \
			-e LICFDIR=../.. \
			-e SONAMELNK=1 \
			-e SHARED=1 \
			-e SYM=1 \
			-us -uc; \
	else \
		echo "Error: Git is unavailable or make-deb runs outside of cloned repository!"; \
	fi

