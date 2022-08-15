ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 2
OMP = 1
SYM = 1

BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

ifneq (,$(strip $(wildcard $(LIBNAME).$(SLIBEXT))))
  DEPSTATIC = 1
else
  DEPSTATIC = 0
endif

ifeq (Darwin,$(UNAME))
  XWRAP ?= 0
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
           $(DEPDIR)/include/libxsmm_source.h
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CXXSRCS := $(wildcard $(SRCDIR)/*.cxx)
CCXSRCS := $(wildcard $(SRCDIR)/*.cc)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
ifneq (,$(strip $(FC)))
FXXSRCS := $(wildcard $(SRCDIR)/*.f)
F77SRCS := $(wildcard $(SRCDIR)/*.F)
F90SRCS := $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)
FXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FXXSRCS:.f=-f.o)))
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90OBJS:.F90=-f90.o)))
endif
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
FTNSRCS := $(FXXSRCS) $(F77SRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F77OBJS) $(F90OBJS)
XFILES := \
  $(OUTDIR)/dgemm-blas $(OUTDIR)/dgemm-wrap \
  $(OUTDIR)/dgemv-blas $(OUTDIR)/dgemv-wrap

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

ifneq (0,$(XWRAP))
  ifneq (2,$(XWRAP))
    WRAP_GEMM = -Wl,--wrap=dgemm_,--wrap=sgemm_
    WRAP_GEMV = -Wl,--wrap=dgemv_,--wrap=sgemv_
  else
    WRAP_GEMM = -Wl,--wrap=dgemm_
    WRAP_GEMV = -Wl,--wrap=dgemv_
  endif
  DFLAGS += -DWRAP
endif

ifeq (0,$(NOBLAS))
$(OUTDIR)/dgemm-blas: $(BLDDIR)/dgemm-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (Darwin,$(UNAME))
ifneq (0,$(XWRAP))
	$(info ================================================================================)
	$(info The static link-time wrapper mechanism is not supported under macOS!)
	$(info ================================================================================)
endif
endif
else
.PHONY: $(OUTDIR)/dgemm-blas
endif

ifeq (0,$(NOBLAS))
$(OUTDIR)/dgemv-blas: $(BLDDIR)/dgemv-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (Darwin,$(UNAME))
ifneq (0,$(XWRAP))
	$(info ================================================================================)
	$(info The static link-time wrapper mechanism is not supported under macOS!)
	$(info ================================================================================)
endif
endif
else
.PHONY: $(OUTDIR)/dgemv-blas
endif

ifneq (0,$(XWRAP))
$(OUTDIR)/dgemm-wrap: $(BLDDIR)/dgemm-c.o $(OUTDIR)/.make
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
endif
else
.PHONY: $(OUTDIR)/dgemm-wrap
endif

ifneq (0,$(XWRAP))
$(OUTDIR)/dgemv-wrap: $(BLDDIR)/dgemv-c.o $(OUTDIR)/.make
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMV)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMV) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMV)
endif
else
.PHONY: $(OUTDIR)/dgemv-wrap
endif

.PHONY: test
test: $(OUTDIR)/.make $(OUTDIR)/wrap-test.sh $(XFILES)
	@$(OUTDIR)/wrap-test.sh dgemm $(shell echo $$(($(TESTSIZE) * 1000)))
	@$(OUTDIR)/wrap-test.sh dgemm 350  16  20 350  35 350  1 0.0
	@$(OUTDIR)/wrap-test.sh dgemm 200 200 200 256 256 256  1 0.0
	@$(OUTDIR)/wrap-test.sh dgemm  24  23  21  32  32  32 -1 0.5
	@$(OUTDIR)/wrap-test.sh dgemv $(shell echo $$(($(TESTSIZE) * 1000)))
	@$(OUTDIR)/wrap-test.sh dgemv 350  20 350 1 1 1 0
	@$(OUTDIR)/wrap-test.sh dgemv 200 200 256 1 1 1 0
	@$(OUTDIR)/wrap-test.sh dgemv  24  21  32 2 2 1 1

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* fit.log *.dat
	@-rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif
	@-rm -f .make .state

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(OUTDIR)/libxsmm.$(DLIBEXT) $(OUTDIR)/*.stackdump
	@-rm -f $(XFILES) $(MODULES)
endif

