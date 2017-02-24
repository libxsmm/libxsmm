# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(MAKE_PARALLEL))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = $(ROOTDIR)
INCDIR = .
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(DEPDIR)/include

BLAS = 2
OMP = 1
SYM = 1

BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

ifneq (,$(strip $(wildcard $(LIBNAME).$(SLIBEXT))))
  DEPSTATIC = 1
else
  DEPSTATIC = 0
endif

XWRAP ?= 0
ifneq (Darwin,$(UNAME))
ifneq (0,$(DEPSTATIC))
  XWRAP = $(WRAP)
endif
endif

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh)
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CXXSRCS := $(wildcard $(SRCDIR)/*.cxx)
CCXSRCS := $(wildcard $(SRCDIR)/*.cx)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
FXXSRCS := $(wildcard $(SRCDIR)/*.f)
F77SRCS := $(wildcard $(SRCDIR)/*.F)
F90SRCS := $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
FXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FXXSRCS:.f=-f.o)))
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90OBJS:.F90=-f90.o)))
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
FTNSRCS := $(FXXSRCS) $(F77SRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F77OBJS) $(F90OBJS)
XFILES := $(OUTDIR)/dgemm-blas $(OUTDIR)/dgemm-wrap

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

$(OUTDIR)/dgemm-blas: $(BLDDIR)/dgemm-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
ifeq (Darwin,$(UNAME))
ifneq (0,$(XWRAP))
	$(info ================================================================================)
	$(info The static link-time wrapper mechanism is not supported under OS X!)
	$(info ================================================================================)
endif
endif

ifneq (0,$(XWRAP))
ifneq (2,$(XWRAP))
  WRAP_FLAG = -Wl,--wrap=dgemm_,--wrap=sgemm_
else
  WRAP_FLAG = -Wl,--wrap=dgemm_
endif
endif

ifneq (0,$(XWRAP))
$(OUTDIR)/dgemm-wrap: $(BLDDIR)/dgemm-c.o $(OUTDIR)/.make $(LIBNAME).$(LIBEXT)
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_FLAG)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_FLAG) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_FLAG)
endif
else
.PHONY: $(OUTDIR)/dgemm-wrap
endif

.PHONY: test
test: $(OUTDIR)/.make $(OUTDIR)/dgemm-test.sh $(XFILES)
	@bash $(OUTDIR)/dgemm-test.sh $(shell echo $$(($(TESTSIZE) * 1000)))
	@bash $(OUTDIR)/dgemm-test.sh 350 16 20 350 35 350 1 0
	@bash $(OUTDIR)/dgemm-test.sh 200 200 200 256 256 256 1 0
	@bash $(OUTDIR)/dgemm-test.sh 24 23 21 32 32 32 -1 0.5

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* fit.log *.dat *.mhd
	@rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif
	@rm -f .make .state

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@rm -f $(OUTDIR)/libxsmm.$(DLIBEXT) $(OUTDIR)/*.stackdump
	@rm -f $(XFILES) $(MODULES)
endif

