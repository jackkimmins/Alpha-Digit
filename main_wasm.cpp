// main_wasm.cpp
#include "NeuralNetwork.h"
#include <emscripten.h>
#include <iostream>
#include <string>

// Global pointer to the neural network instance
NeuralNetwork* nn = nullptr;

extern "C" {

// Function to initialize the neural network and load the model
EMSCRIPTEN_KEEPALIVE
void initialize_nn() {
    if (nn != nullptr) {
        delete nn;
        nn = nullptr;
    }

    // Neural network architecture parameters
    int input_size = 784; // 28x28 pixels
    std::vector<int> hidden_layers = { 128, 64 };
    int output_size = 10; // Digits 0-9
    unsigned int seed = 42;

    // Initialize the neural network
    nn = new NeuralNetwork(input_size, hidden_layers, output_size, seed);

    // Load the trained model from the embedded filesystem
    std::string model_load_path = "/models/best_model.dat";
    nn->load_model(model_load_path);
}

// Function to classify a digit given a CSV string
EMSCRIPTEN_KEEPALIVE
int classify_digit(const char* input_csv) {
    if (nn == nullptr) {
        std::cerr << "Neural network not initialized.\n";
        return -1;
    }

    std::string input(input_csv);
    int classification = nn->classify_user_input(input);
    return classification;
}

// Function to clean up and delete the neural network instance
EMSCRIPTEN_KEEPALIVE
void cleanup_nn() {
    if (nn != nullptr) {
        delete nn;
        nn = nullptr;
    }
}

} // extern "C"
