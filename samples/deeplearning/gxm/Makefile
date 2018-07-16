#############################################################################
# Copyright (c) 2017-2018, Intel Corporation                                #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Sasikanth Avancha, Dhiraj Kalamkar, Alexander Heinecke (Intel Corp.)
#############################################################################

PROJECT := gxm

CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)
ifeq ($(RELEASE_BUILD_DIR),)
	RELEASE_BUILD_DIR := .$(BUILD_DIR)_release
endif
ifeq ($(DEBUG_BUILD_DIR),)
	DEBUG_BUILD_DIR := .$(BUILD_DIR)_debug
endif

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
	OTHER_BUILD_DIR := $(RELEASE_BUILD_DIR)
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
	OTHER_BUILD_DIR := $(DEBUG_BUILD_DIR)
endif

# All of the directories containing code.
SRC_DIRS := $(shell find * -type d -exec bash -c "find {} -maxdepth 1 \
	\( -name '*.cpp' -o -name '*.proto' \) | grep -q ." \; -print)

BINARY_NAME := $(PROJECT)
BIN_BUILD_DIR := $(BUILD_DIR)/bin
FULL_BIN_NAME := $(BIN_BUILD_DIR)/$(BINARY_NAME)

# Source files
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src/ ! -name "test_*.cpp" -name "*.cpp" | sort)
#$(info CXX_SRCS = $(CXX_SRCS))
# BUILD_INCLUDE_DIR contains any generated header files we want to include.
BUILD_INCLUDE_DIR := $(BUILD_DIR)
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRC_DIR := proto
PROTO_SRCS := $(wildcard $(PROTO_SRC_DIR)/*.proto)
# PROTO_BUILD_DIR will contain the .cc and obj files generated from
# PROTO_SRCS; PROTO_BUILD_INCLUDE_DIR will contain the .h header files
PROTO_BUILD_DIR := $(BUILD_DIR)/$(PROTO_SRC_DIR)
PROTO_BUILD_INCLUDE_DIR := $(BUILD_DIR)/proto
NONGEN_CXX_SRCS := $(shell find \
	src/ \
	include/ \
	-name "*.cpp" -or -name "*.hpp")

# Generated files
# The generated files for protocol buffers
PROTO_GEN_HEADER_SRCS := $(addprefix $(PROTO_BUILD_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_HEADER := $(addprefix $(PROTO_BUILD_INCLUDE_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_CC := $(addprefix $(BUILD_DIR)/, ${PROTO_SRCS:.proto=.pb.cc})
# Source file objects
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
PROTO_OBJS := ${PROTO_GEN_CC:.cc=.o}
OBJS := $(PROTO_OBJS) $(CXX_OBJS)
# Output files for automatic dependency generation
DEPS := ${CXX_OBJS:.o=.d}

# Compiler warning locations
WARNS_EXT := warnings.txt
CXX_WARNS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o.$(WARNS_EXT)})
ALL_WARNS := $(ALL_CXX_WARNS)
EMPTY_WARN_REPORT := $(BUILD_DIR)/.$(WARNS_EXT)
NONEMPTY_WARN_REPORT := $(BUILD_DIR)/$(WARNS_EXT)

# include and lib directories

INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ./src ./include
LIBRARIES += glog gflags protobuf m

USE_OPENCV ?= 1

ifeq ($(USE_OPENCV), 1)
	LIBRARIES += opencv_core opencv_highgui opencv_imgproc

	ifeq ($(OPENCV_VERSION), 3)
		LIBRARIES += opencv_imgcodecs
	endif
endif

WARNINGS := -Wall -Wno-sign-compare

# build directories
LIB_BUILD_DIR := ./lib
ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(SRC_DIRS)) \
	$(LIB_BUILD_DIR) $(BIN_BUILD_DIR) $(PROTO_BUILD_INCLUDE_DIR))

# build
# Determine platform
UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LINUX := 1
endif

# Linux
ifeq ($(LINUX), 1)
	LIBRARIES += stdc++
endif

# Custom compiler
ifdef CUSTOM_CXX
	CXX := $(CUSTOM_CXX)
endif

# Architecture
ifeq ($(ARCH), avx2)
	COMMON_FLAGS += -xCORE-AVX2
endif
ifeq ($(ARCH), avx512_common)
	COMMON_FLAGS += -xCOMMON-AVX512
	CXXFLAGS += -DUSE_NTS_SPLIT -DUSE_NTS_BN -DUSE_BLOCKING_BN
endif
ifeq ($(ARCH), avx512_mic)
	COMMON_FLAGS += -xMIC-AVX512
	CXXFLAGS += -DUSE_NTS_SPLIT -DUSE_NTS_BN
endif

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -g3 -O0 -gdwarf-3
else
	COMMON_FLAGS += -DNDEBUG -O2 -ip -ipo
endif

# configure IO libraries
ifeq ($(USE_OPENCV), 1)
	COMMON_FLAGS += -DUSE_OPENCV
endif

# CPU-only configuration
ifeq ($(CPU_ONLY), 1)
	OBJS := $(PROTO_OBJS) $(CXX_OBJS)
	ALL_WARNS := $(ALL_CXX_WARNS)
	COMMON_FLAGS += -DCPU_ONLY
endif

ifeq ($(OPENMP), 1)
	COMMON_FLAGS += -qopenmp
endif

# BLAS configuration (default = mkl)
ifeq ($(BLAS), openblas)
	# OpenBLAS
	LIBRARIES += openblas

	BLAS_INCLUDE = $(OPENBLAS)/include
	BLAS_LIB = $(OPENBLAS)/lib
	INCLUDE_DIRS += $(BLAS_INCLUDE)
	LIBRARY_DIRS += $(BLAS_LIB)
endif
ifeq ($(BLAS), mkl)
	COMMON_FLAGS += -mkl
endif

INCLUDE_DIRS += $(GXM_LIBRARY_PATH)/include
LIBRARY_DIRS += $(GXM_LIBRARY_PATH)/lib

LIBRARY_DIRS += $(LIB_BUILD_DIR)

# libxsmm paths
LIBRARIES += xsmm xsmmext
INCLUDE_DIRS += $(LIBXSMM_PATH)/include
LIBRARY_DIRS += $(LIBXSMM_PATH)/lib

#MLSL paths
ifeq ($(MLSL), 1)
  LIBRARIES += mlsl
  INCLUDE_DIRS += $(MLSL_ROOT)/intel64/include
  LIBRARY_DIRS += $(MLSL_ROOT)/intel64/lib
  CXXFLAGS += -DUSE_MLSL
  CXX = mpiicpc
endif

ifeq ($(LMDB), 1)
  LIBRARIES += lmdb
  LIBRARY_DIRS += $(GXM_LIBRARY_PATH)/lib
  INCLUDE_DIRS += $(GXM_LIBRARY_PATH)/include
  CXXFLAGS += -DUSE_LMDB
endif

# Optimized Buffer allocation for BP
ifeq ($(BPOPT_ALLOC), 1)
  CXXFLAGS += -DUSE_OPTBP_ALLOC
endif

ifeq ($(NUMA_ON), 1)
  CXXFLAGS += -DUSE_NUMA
endif

ifeq ($(XSMM_TIMING), 1)
  CXXFLAGS += -DUSE_XSMM_TIMING
endif

# Return all
ifeq ($(RETURN_NC), 1)
	CXXFLAGS += -DRETURNALL
endif

# Timing per layer
ifeq ($(TIME), 1)
	CXXFLAGS += -DTIMING
endif

# Stats for layer activations/weights
ifeq ($(STATS), 1)
  	CXXFLAGS += -DGETSTATS
endif

ifeq ($(DUMP_ACT), 1)
  	CXXFLAGS += -DDUMP_ACT_DATA
endif

ifeq ($(DUMP_WT), 1)
  	CXXFLAGS += -DDUMP_WT_DATA
endif

ifeq ($(CANCHECK), 1)
  	CXXFLAGS += -DCANARY_CHECK
endif

ifeq ($(FP32_BU), 1)
  	CXXFLAGS += -DCHECK_BLOWUP_FP32
endif

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += $(COMMON_FLAGS) $(WARNINGS) -std=c++11
LINKFLAGS += $(COMMON_FLAGS) $(WARNINGS)

LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))

# build targets
all: bin

bin: $(FULL_BIN_NAME)

$(BUILD_DIR_LINK): $(BUILD_DIR)/.linked

$(BUILD_DIR)/.linked:
	@ mkdir -p $(BUILD_DIR)
	@ $(RM) $(OTHER_BUILD_DIR)/.linked
	@ $(RM) -r $(BUILD_DIR_LINK)
	@ ln -s $(BUILD_DIR) $(BUILD_DIR_LINK)
	@ touch $@

$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p $@

$(FULL_BIN_NAME): $(OBJS)
		$(CXX) -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp | $(ALL_BUILD_DIRS)
	@ echo CXX $<
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)

$(PROTO_BUILD_DIR)/%.pb.o: $(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_GEN_HEADER) \
		| $(PROTO_BUILD_DIR)
	@ echo CXX $<
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)

proto: $(PROTO_GEN_CC) $(PROTO_GEN_HEADER)

$(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_BUILD_DIR)/%.pb.h : \
		$(PROTO_SRC_DIR)/%.proto | $(PROTO_BUILD_DIR)
	@ echo PROTOC $<
	protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<

clean:
	@- $(RM) -rf $(ALL_BUILD_DIRS)
	@- $(RM) -rf $(OTHER_BUILD_DIR)
	@- $(RM) -rf $(BUILD_DIR_LINK)

-include $(DEPS)
