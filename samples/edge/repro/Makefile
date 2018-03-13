ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../../..
SRCDIR = $(ROOTDIR)/src
INCDIR = .
BLDDIR = build
OUTDIR = bin

# settings for reproducer
EDGE_EQUATION  ?= elastic
EDGE_ELEMENT   ?= tet4
EDGE_KERNEL    ?= xsmm
EDGE_PARALLEL  ?= omp
EDGE_ARCH      ?= avx512
EDGE_ORDER     ?= 5
EDGE_PRECISION ?= 32
EDGE_CFR       ?= 16
EDGE_CONFIG    := $(EDGE_ORDER)_$(EDGE_PRECISION)_$(EDGE_CFR)
EDGE_VALID     ?= 0

ifeq ($(EDGE_PARALLEL), omp)
EDGE_FLAGS = -DPP_USE_OMP
else
$(error EDGE_PARALLEL=$(EDGE_PARALLEL) unsupported)
endif

ifeq ($(EDGE_EQUATION), elastic)
EDGE_FLAGS += -DPP_T_EQUATIONS_ELASTIC
else
$(error EDGE_EQUATION=$(EDGE_EQUATION) unsupported)
endif

ifeq ($(EDGE_ELEMENT), tet4)
EDGE_FLAGS += -DPP_T_ELEMENTS_TET4
else
$(error EDGE_ELEMENT=$(EDGE_ELEMENT) unsupported)
endif

ifeq ($(EDGE_KERNEL), xsmm)
EDGE_FLAGS += -DPP_T_KERNELS_XSMM
else
$(error EDGE_KERNEL=$(EDGE_KERNEL) unsupported)
endif

EDGE_FLAGS += -DPP_ORDER=$(EDGE_ORDER) -DPP_PRECISION=$(EDGE_PRECISION) -DPP_N_CRUNS=$(EDGE_CFR)

ifeq ($(EDGE_VALID), 1)
EDGE_FLAGS += -DPP_REPRODUCER_VALIDATE
endif

# settings for compiler
CXXFLAGS = $(EDGE_CXXFLAGS) $(EDGE_FLAGS)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(DEPDIR)/include

# settings for libxsmm
BLAS = 1
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh)
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))

SOURCES := $(CPPSRCS) #$(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) #$(CXXOBJS) $(CCXOBJS) $(COBJCTS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))

EXFILES := $(OUTDIR)/local_$(EDGE_CONFIG) \
           $(OUTDIR)/neigh_$(EDGE_CONFIG) \
           $(OUTDIR)/combine_$(EDGE_CONFIG)
TESTEXS := $(OUTDIR)/test_local_$(EDGE_CONFIG) \
           $(OUTDIR)/test_neigh_$(EDGE_CONFIG) \
           $(OUTDIR)/test_combine_$(EDGE_CONFIG) #\
           $(OUTDIR)/test_helper

ifeq ($(EDGE_ARCH), avx512)
ifeq ($(CXX), icpc)
CTARGET = -xCOMMON-AVX512
else
CTARGET = -mavx512f -mavx512cd
endif
endif


.PHONY: all
all: $(EXFILES) test

test: $(TESTEXS)

test_helper: $(OUTDIR)/test_helper

local: $(OUTDIR)/local_$(EDGE_CONFIG)
neigh: $(OUTDIR)/neigh_$(EDGE_CONFIG)
combine: $(OUTDIR)/combine_$(EDGE_CONFIG)

.PHONY: compile
compile: $(OBJECTS)



$(OUTDIR)/local_$(EDGE_CONFIG): $(OUTDIR)/.make $(BLDDIR)/edge_local_$(EDGE_CONFIG).o $(LIBDEP)
	$(XLD) -o $@ $(BLDDIR)/edge_local_$(EDGE_CONFIG).o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/neigh_$(EDGE_CONFIG): $(OUTDIR)/.make $(BLDDIR)/edge_neigh_$(EDGE_CONFIG).o $(LIBDEP)
	$(XLD) -o $@ $(BLDDIR)/edge_neigh_$(EDGE_CONFIG).o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/combine_$(EDGE_CONFIG): $(OUTDIR)/.make $(BLDDIR)/edge_combine_$(EDGE_CONFIG).o $(LIBDEP)
	$(XLD) -o $@ $(BLDDIR)/edge_combine_$(EDGE_CONFIG).o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)


$(OUTDIR)/test_local_$(EDGE_CONFIG): $(OUTDIR)/.make $(BLDDIR)/edge_local.test_$(EDGE_CONFIG).o $(LIBDEP)
	$(XLD) -o $@ $(BLDDIR)/edge_local.test_$(EDGE_CONFIG).o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/test_neigh_$(EDGE_CONFIG): $(OUTDIR)/.make $(BLDDIR)/edge_neigh.test_$(EDGE_CONFIG).o $(LIBDEP)
	$(XLD) -o $@ $(BLDDIR)/edge_neigh.test_$(EDGE_CONFIG).o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/test_combine_$(EDGE_CONFIG): $(OUTDIR)/.make $(BLDDIR)/edge_combine.test_$(EDGE_CONFIG).o $(LIBDEP)
	$(XLD) -o $@ $(BLDDIR)/edge_combine.test_$(EDGE_CONFIG).o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

# $(OUTDIR)/test_helper: $(SRCDIR)/edge_helper.test.cpp .state $(OUTDIR)/.make $(SRCDIR)/edge_helper.hpp $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBDEP)
# 	$(CXX) $(EDGE_CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $(SRCDIR)/edge_helper.test.cpp -o $(BLDDIR)/edge_helper.test.o
# 	$(XLD) -o $@ $(BLDDIR)/edge_helper.test.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%_$(EDGE_CONFIG).o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* fit.log *.dat
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
	@rm -f $(EXFILES) $(MODULES)
endif
