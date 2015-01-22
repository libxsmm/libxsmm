
# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise 
ROW_MAJOR ?= 1

# M, N, K values of the generated matrices
INDICES_M ?= $(shell seq 1 5)
INDICES_N ?= $(shell seq 1 5)
INDICES_K ?= $(shell seq 1 5)

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Use aligned Store and/or aligned Load instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

# THRESHOLD problem size (M x N x K); determines when to use BLAS 
THRESHOLD ?= $(shell echo $$((24 * 24 * 24)))

ROOTDIR ?= .
SCRDIR = $(ROOTDIR)/scripts
OBJDIR = $(ROOTDIR)/build
INCDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
LIBDIR = $(ROOTDIR)/lib

INDICES ?= $(foreach m,$(INDICES_M),$(foreach n,$(INDICES_N),$(foreach k,$(INDICES_K),$m_$n_$k)))

TARGET_COMPILE_C_KNC := icc -std=c99 -mkl=sequential -O2 -fPIC -fno-alias -ansi-alias -mmic
TARGET_COMPILE_C_HST := icc -std=c99 -mkl=sequential -O2 -fPIC -fno-alias -ansi-alias -mavx -axCORE-AVX2 -offload-attribute-target=mic
AR := xiar

SRCFILES = $(patsubst %,mm_%.c,$(INDICES))
OBJFILES_KNC = $(patsubst %,$(OBJDIR)/mic/mm_%.o,$(INDICES))
OBJFILES_HST = $(patsubst %,$(OBJDIR)/intel64/mm_%.o,$(INDICES))

LIB_KNC  ?= $(LIBDIR)/mic/libxsmm.a
LIB_HST  ?= $(LIBDIR)/intel64/libxsmm.a
INC_KNC   = $(INCDIR)/libxsmm.h
MAIN  = $(SRCDIR)/libxsmm.c


lib_all: lib_knc lib_hst

header_knc: $(INC_KNC)
$(INC_KNC): $(INCDIR)/libxsmm.0 $(INCDIR)/libxsmm.1 $(INCDIR)/libxsmm.2
	@cat $(INCDIR)/libxsmm.0 > $@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) $(THRESHOLD) $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) >> $@
	@echo >> $@
	@cat $(INCDIR)/libxsmm.1 >> $@
	@echo >> $@
	@python $(SCRDIR)/libxsmm_interface.py $(ROW_MAJOR) $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) >> $@
	@cat $(INCDIR)/libxsmm.2 >> $@

source_knc: $(addprefix $(SRCDIR)/,$(SRCFILES))
$(SRCDIR)/%.c: $(INC_KNC)
	@mkdir -p $(SRCDIR)
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) -1 `echo $* | awk -F_ '{ print $$2" "$$3" "$$4 }'` > $@

main_knc: $(MAIN)
$(MAIN): $(INC_KNC)
	@mkdir -p $(SRCDIR)
	@python $(SCRDIR)/libxsmm_dispatch.py $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) > $@

compile_knc: $(OBJFILES_KNC)
$(OBJDIR)/mic/%.o: $(SRCDIR)/%.c $(INCDIR)/libxsmm_isa.h
	@mkdir -p $(OBJDIR)/mic
	$(TARGET_COMPILE_C_KNC) -I$(INCDIR) -c $< -o $@

compile_hst: $(OBJFILES_HST)
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)/intel64
	$(TARGET_COMPILE_C_HST) -I$(INCDIR) -c $< -o $@

lib_knc: $(LIB_KNC)
ifeq ($(origin NO_MAIN), undefined)
$(LIB_KNC): $(OBJFILES_KNC) $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/mic/%.o,$(MAIN))
else
$(LIB_KNC): $(OBJFILES_KNC)
endif
	@mkdir -p $(LIBDIR)/mic
	$(AR) -rs $@ $^

lib_hst: $(LIB_HST)
ifeq ($(origin NO_MAIN), undefined)
$(LIB_HST): $(OBJFILES_HST) $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/intel64/%.o,$(MAIN))
else
$(LIB_HST): $(OBJFILES_HST)
endif
	@mkdir -p $(LIBDIR)/intel64
	$(AR) -rs $@ $^

clean:
	rm -rf $(SRCDIR) $(OBJDIR) $(ROOTDIR)/*~ $(ROOTDIR)/*/*~

realclean: clean
	rm -rf $(LIBDIR) $(INC_KNC)

install: lib_all clean
