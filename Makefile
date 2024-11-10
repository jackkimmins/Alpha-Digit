# Compiler for C++ executables
CXX = g++

# Compiler Flags for C++ executables
CXXFLAGS = -std=c++20 -g -Iinclude

# Compiler for WASM
EMCC = em++

# Compilation Flags for WASM (used during compilation of .cpp to .o)
EMCC_COMPILE_FLAGS = -std=c++20 -O3 -Iinclude -fexceptions

# Linking Flags for WASM (used during linking of .o to .js/.wasm)
EMCC_LINK_FLAGS = -s WASM=1 \
                  -s MODULARIZE=1 \
                  -s 'EXPORT_NAME="NeuralNetModule"' \
                  -s EXPORTED_FUNCTIONS='["_initialise_nn", "_classify_digit", "_cleanup_nn"]' \
                  -s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
                  -s DISABLE_EXCEPTION_CATCHING=0 \
                  --preload-file models/best_model.dat@/models/best_model.dat

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BIN_DIR = bin
OBJ_DIR_CPP = obj/cpp
OBJ_DIR_WASM = obj/wasm
WEB_DIR = web
WASM_SUBDIR = wasm

# Source Files
TRAIN_MAIN = main_train.cpp
INFER_MAIN = main_infer.cpp
WASM_MAIN = main_wasm.cpp
SRC_SOURCES = $(wildcard src/*.cpp)

# Object Files for Standard Executables (Compiled with g++)
TRAIN_OBJECTS = $(OBJ_DIR_CPP)/main_train.o $(patsubst src/%.cpp,$(OBJ_DIR_CPP)/src_%.o,$(SRC_SOURCES))
INFER_OBJECTS = $(OBJ_DIR_CPP)/main_infer.o $(patsubst src/%.cpp,$(OBJ_DIR_CPP)/src_%.o,$(SRC_SOURCES))

# Object Files for WASM (Compiled with em++)
WASM_OBJECTS = $(OBJ_DIR_WASM)/main_wasm.o $(patsubst src/%.cpp,$(OBJ_DIR_WASM)/src_%.o,$(SRC_SOURCES))

# Targets
TRAIN_TARGET = $(BIN_DIR)/train
INFER_TARGET = $(BIN_DIR)/infer
WASM_OUTPUT_JS = $(WEB_DIR)/$(WASM_SUBDIR)/NeuralNetModule.js
WASM_OUTPUT_WASM = $(WEB_DIR)/$(WASM_SUBDIR)/NeuralNetModule.wasm
WASM_OUTPUT_DATA = $(WEB_DIR)/$(WASM_SUBDIR)/NeuralNetModule.data

# All Target: Build Standard Executables and WASM Module
all: $(TRAIN_TARGET) $(INFER_TARGET) $(WASM_OUTPUT_JS) $(WASM_OUTPUT_WASM) $(WASM_OUTPUT_DATA)

# Train Executable
$(TRAIN_TARGET): $(TRAIN_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Infer Executable
$(INFER_TARGET): $(INFER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile WASM Module JavaScript and WASM
$(WASM_OUTPUT_JS) $(WASM_OUTPUT_WASM) $(WASM_OUTPUT_DATA): $(WASM_OBJECTS)
	@mkdir -p $(WEB_DIR)/$(WASM_SUBDIR)
	$(EMCC) $(EMCC_LINK_FLAGS) -o $(WASM_OUTPUT_JS) $^

# Explicit Rule: Compile main_train.cpp to obj/cpp/main_train.o
$(OBJ_DIR_CPP)/main_train.o: main_train.cpp
	@mkdir -p $(OBJ_DIR_CPP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Explicit Rule: Compile main_infer.cpp to obj/cpp/main_infer.o
$(OBJ_DIR_CPP)/main_infer.o: main_infer.cpp
	@mkdir -p $(OBJ_DIR_CPP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Explicit Rule: Compile main_wasm.cpp to obj/wasm/main_wasm.o
$(OBJ_DIR_WASM)/main_wasm.o: main_wasm.cpp
	@mkdir -p $(OBJ_DIR_WASM)
	$(EMCC) $(EMCC_COMPILE_FLAGS) -c $< -o $@

# Pattern Rule: Compile src/*.cpp to obj/cpp/src_*.o (Compiled with g++)
$(OBJ_DIR_CPP)/src_%.o: src/%.cpp
	@mkdir -p $(OBJ_DIR_CPP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pattern Rule: Compile src/*.cpp to obj/wasm/src_%.o (Compiled with em++)
$(OBJ_DIR_WASM)/src_%.o: src/%.cpp
	@mkdir -p $(OBJ_DIR_WASM)
	$(EMCC) $(EMCC_COMPILE_FLAGS) -c $< -o $@

# Run Training Executable
run_train: $(TRAIN_TARGET)
	@echo "Running Training Executable: $(TRAIN_TARGET)"
	./$(TRAIN_TARGET)

# Run Inference Executable
run_infer: $(INFER_TARGET)
	@echo "Running Inference Executable: $(INFER_TARGET)"
	./$(INFER_TARGET)

# Clean Build Files
clean:
	rm -f $(OBJ_DIR_CPP)/*.o
	rm -f $(OBJ_DIR_WASM)/*.o
	rm -f $(BIN_DIR)/*
	rm -rf $(WEB_DIR)/$(WASM_SUBDIR)

# Phony Targets
.PHONY: all clean run_train run_infer