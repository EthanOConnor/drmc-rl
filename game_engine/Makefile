CXX ?= g++
CC ?= clang
THREAD_FLAGS ?= -pthread
CXXFLAGS ?= -std=c++17 -Wall -O2
CFLAGS ?= -std=c11 -O3 -DNDEBUG
LDFLAGS ?=

CXXFLAGS += $(THREAD_FLAGS)
LDFLAGS += $(THREAD_FLAGS)

TARGET = drmario_engine
MONITOR = monitor
SRCS = main.cpp GameLogic.cpp
OBJS = $(SRCS:.cpp=.o)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHLIB_EXT := dylib
SHLIB_LDFLAGS := -dynamiclib
else ifeq ($(UNAME_S),Linux)
SHLIB_EXT := so
SHLIB_LDFLAGS := -shared
else
SHLIB_EXT := so
SHLIB_LDFLAGS := -shared
endif

BUILD_DIR := build
POOL_LIB := $(BUILD_DIR)/libdrmario_pool.$(SHLIB_EXT)
POOL_SRCS_CPP := DrMarioPool.cpp drmario_pool_capi.cpp GameLogic.cpp
POOL_SRCS_C := ../reach_native/drm_reach_full.c
POOL_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(POOL_SRCS_CPP)) $(BUILD_DIR)/drm_reach_full.o

all: $(TARGET) $(MONITOR)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(MONITOR): monitor.cpp
	$(CXX) $(CXXFLAGS) monitor.cpp -o $(MONITOR) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O3 -fPIC -c $< -o $@

$(BUILD_DIR)/drm_reach_full.o: $(POOL_SRCS_C) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $(POOL_SRCS_C) -o $@

libdrmario_pool: $(POOL_LIB)

$(POOL_LIB): $(POOL_OBJS) | $(BUILD_DIR)
	$(CXX) $(SHLIB_LDFLAGS) -o $@ $(POOL_OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) $(MONITOR)
	rm -rf $(BUILD_DIR)
