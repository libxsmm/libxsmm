ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
TSTDIR = ./kernel_test
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

OMP = 1
SYM = 1

# explore AVX/ARCH=native
SSE = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

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
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS)))
XFILES := $(OUTDIR)/eltwise_unary_reduce $(OUTDIR)/eltwise_unary_gather_scatter \
          $(OUTDIR)/eltwise_opreduce_idxvecs $(OUTDIR)/eltwise_binary_simple \
          $(OUTDIR)/eltwise_unary_dropout $(OUTDIR)/eltwise_unary_transform \
          $(OUTDIR)/eltwise_unary_simple $(OUTDIR)/eltwise_unary_relu \
          $(OUTDIR)/eltwise_unary_quantization

.PHONY: all
all: $(XFILES) scripts

.PHONY: compile
compile: $(OBJECTS)

.PHONY: scripts
scripts: $(TSTDIR)/.generate_opreduce_test_scripts $(TSTDIR)/.generate_unary_simple_test_scripts $(TSTDIR)/.generate_unary_transform_test_scripts $(TSTDIR)/.generate_unary_reduce_test_scripts $(TSTDIR)/.generate_unary_relu_test_scripts $(TSTDIR)/.generate_unary_dropout_test_scripts $(TSTDIR)/.generate_unary_gather_scatter_test_scripts $(TSTDIR)/.generate_binary_test_scripts

$(TSTDIR)/.generate_opreduce_test_scripts: $(TSTDIR)/generate_opreduce_test_scripts.sh Makefile
	$(info Creating opreduce test scripts...)
	@cd $(TSTDIR) && ./generate_opreduce_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_unary_simple_test_scripts: $(TSTDIR)/generate_unary_simple_test_scripts.sh Makefile
	$(info Creating unary simple test scripts...)
	@cd $(TSTDIR) && ./generate_unary_simple_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_unary_transform_test_scripts: $(TSTDIR)/generate_unary_transform_test_scripts.sh Makefile
	$(info Creating unary transform test scripts...)
	@cd $(TSTDIR) && ./generate_unary_transform_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_unary_reduce_test_scripts: $(TSTDIR)/generate_unary_reduce_test_scripts.sh Makefile
	$(info Creating unary reduce test scripts...)
	@cd $(TSTDIR) && ./generate_unary_reduce_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_unary_relu_test_scripts: $(TSTDIR)/generate_unary_relu_test_scripts.sh Makefile
	$(info Creating unary relu test scripts...)
	@cd $(TSTDIR) && ./generate_unary_relu_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_unary_dropout_test_scripts: $(TSTDIR)/generate_unary_dropout_test_scripts.sh Makefile
	$(info Creating unary dropout test scripts...)
	@cd $(TSTDIR) && ./generate_unary_dropout_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_unary_gather_scatter_test_scripts: $(TSTDIR)/generate_unary_gather_scatter_test_scripts.sh Makefile
	$(info Creating unary gather scatter test scripts...)
	@cd $(TSTDIR) && ./generate_unary_gather_scatter_test_scripts.sh >/dev/null
	@-touch $@

$(TSTDIR)/.generate_binary_test_scripts: $(TSTDIR)/generate_binary_test_scripts.sh Makefile
	$(info Creating binary test scripts...)
	@cd $(TSTDIR) && ./generate_binary_test_scripts.sh >/dev/null
	@-touch $@

$(OUTDIR)/eltwise_unary_reduce: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_reduce-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_reduce-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_unary_gather_scatter: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_gather_scatter-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_gather_scatter-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_opreduce_idxvecs: $(OUTDIR)/.make $(BLDDIR)/eltwise_opreduce_idxvecs-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_opreduce_idxvecs-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_binary_simple: $(OUTDIR)/.make $(BLDDIR)/eltwise_binary_simple-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_binary_simple-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_unary_simple: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_simple-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_simple-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_unary_relu: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_relu-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_relu-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_unary_dropout: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_dropout-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_dropout-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_unary_transform: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_transform-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_transform-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_unary_quantization: $(OUTDIR)/.make $(BLDDIR)/eltwise_unary_quantization-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_unary_quantization-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* *.dat *.log
	@-rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif

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
	@-rm -f $(TSTDIR)/.generate_opreduce_test_scripts
	@-rm -f $(TSTDIR)/.generate_unary_simple_test_scripts
	@-rm -f $(TSTDIR)/.generate_unary_transform_test_scripts
	@-rm -f $(TSTDIR)/.generate_unary_reduce_test_scripts
	@-rm -f $(TSTDIR)/.generate_unary_relu_test_scripts
	@-rm -f $(TSTDIR)/.generate_unary_dropout_test_scripts
	@-rm -f $(TSTDIR)/.generate_unary_gather_scatter_test_scripts
	@-rm -f $(TSTDIR)/.generate_binary_test_scripts
	@-rm -f $(TSTDIR)/opreduce_op_*.sh
	@-rm -f $(TSTDIR)/unary_*.sh
	@-rm -f $(TSTDIR)/binary_*.sh

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
