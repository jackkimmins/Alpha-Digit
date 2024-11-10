#include "NeuralNetwork.h"
#include <emscripten.h>
#include <iostream>
#include <string>

NeuralNetwork* nn = nullptr;

extern "C" {
    // Init the neural network and load the model
    EMSCRIPTEN_KEEPALIVE
    void initialise_nn() {
        if (nn != nullptr) {
            delete nn;
            nn = nullptr;
        }

        // Architecture Parameters - I would like to move these to a config file, future Jack problem! 😇
        int input_size = 784; // 28x28 pixels
        std::vector<int> hidden_layers = { 128, 64 };
        int output_size = 10; // Digits 0-9
        unsigned int seed = 42;

        // Initialise the neural network
        nn = new NeuralNetwork(input_size, hidden_layers, output_size, seed);

        // Load the trained model from the embedded WASM filesystem
        std::string model_load_path = "/models/best_model.dat";
        nn->load_model(model_load_path);
    }

    // Classify a digit given a CSV string
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

    // Clean up and delete the network instance
    EMSCRIPTEN_KEEPALIVE
    void cleanup_nn() {
        if (nn != nullptr) {
            delete nn;
            nn = nullptr;
        }
    }
}