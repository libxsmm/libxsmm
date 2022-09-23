ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../../../../../
DEP2DIR = ../../
SRCDIR = .
INCDIR = ../../include
INC2DIR = $(DEP2DIR)/include
SRC2DIR = $(DEP2DIR)/src
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 0
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
           $(DEPDIR)/include/libxsmm_source.h
HEADERS2 := $(wildcard $(INC2DIR)/*.h) $(wildcard $(INC2DIR)/*.hpp) $(wildcard $(INC2DIR)/*.hxx) $(wildcard $(INC2DIR)/*.hh) \
            $(wildcard $(SRC2DIR)/*.h) $(wildcard $(SRC2DIR)/*.hpp) $(wildcard $(SRC2DIR)/*.hxx) $(wildcard $(SRC2DIR)/*.hh)
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CXXSRCS := $(wildcard $(SRCDIR)/*.cxx)
CCXSRCS := $(wildcard $(SRCDIR)/*.cc)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
CPPSRCS2 := $(wildcard $(SRC2DIR)/*.cpp)
CXXSRCS2 := $(wildcard $(SRC2DIR)/*.cxx)
CCXSRCS2 := $(wildcard $(SRC2DIR)/*.cc)
CSOURCS2 := $(wildcard $(SRC2DIR)/*.c)
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
SOURCES2 := $(CPPSRCS2) $(CXXSRCS2) $(CCXSRCS2) $(CSOURCS2)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
FTNSRCS := $(FXXSRCS) $(F77SRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F77OBJS) $(F90OBJS)
XFILES := $(OUTDIR)/mlp_mpi_example_f32 $(OUTDIR)/mlp_mpi_ovlp_example_f32
DNNLIB := $(DEP2DIR)/lib/libxsmm_dnn.a

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

$(DNNLIB): $(HEADERS2) $(SOURCES2)
	$(MAKE) -j8 -C $(DEP2DIR)

$(OUTDIR)/mlp_mpi_example_f32: $(OUTDIR)/.make $(BLDDIR)/mlp_mpi_example_f32-c.o $(LIBDEP) $(DNNLIB)
	$(LD) -o $@ $(BLDDIR)/mlp_mpi_example_f32-c.o $(call cleanld,$(MAINLIB) $(DNNLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS))

$(OUTDIR)/mlp_mpi_ovlp_example_f32: $(OUTDIR)/.make $(BLDDIR)/mlp_mpi_ovlp_example_f32-c.o $(LIBDEP) $(DNNLIB)
	$(LD) -o $@ $(BLDDIR)/mlp_mpi_ovlp_example_f32-c.o $(call cleanld,$(MAINLIB) $(DNNLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS))

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
