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

BLAS = 1
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

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
XFILES := $(OUTDIR)/edge_vol_int
#XFILES := $(OUTDIR)/edge_vol_int $(OUTDIR)/asparse_soa_f64 $(OUTDIR)/bsparse_soa_f64 $(OUTDIR)/asparse_soa_f32 $(OUTDIR)/bsparse_soa_f32

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

$(OUTDIR)/edge_vol_int: $(OUTDIR)/.make $(BLDDIR)/edge_vol_int-c.o $(BLDDIR)/edge_proxy_common-c.o $(LIBNAME).$(LIBEXT)
	$(LD) -o $@ $(BLDDIR)/edge_vol_int-c.o $(BLDDIR)/edge_proxy_common-c.o $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

#$(OUTDIR)/asparse_soa_f64: $(OUTDIR)/.make $(BLDDIR)/asparse_soa-c.o $(LIBNAME).$(LIBEXT)
#	$(LD) -o $@ $(BLDDIR)/asparse_soa-c.o $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
#
#$(OUTDIR)/bsparse_soa_f64: $(OUTDIR)/.make $(BLDDIR)/bsparse_soa-c.o $(LIBNAME).$(LIBEXT)
#	$(LD) -o $@ $(BLDDIR)/bsparse_soa-c.o $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
#
#$(OUTDIR)/asparse_soa_f32: $(OUTDIR)/.make $(BLDDIR)/asparse_soa-c-f32.o $(LIBNAME).$(LIBEXT)
#	$(LD) -o $@ $(BLDDIR)/asparse_soa-c-f32.o $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
#
#$(OUTDIR)/bsparse_soa_f32: $(OUTDIR)/.make $(BLDDIR)/bsparse_soa-c-f32.o $(LIBNAME).$(LIBEXT)
#	$(LD) -o $@ $(BLDDIR)/bsparse_soa-c-f32.o $(call abslib,$(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-cpp-f32.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(CXXFLAGS) $(DFLAGS) -D__EDGE_EXECUTE_F32__ $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c-f32.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) -D__EDGE_EXECUTE_F32__ $(IFLAGS) $(CTARGET) -c $< -o $@

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

