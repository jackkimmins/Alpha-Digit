#include "Database.h"
#include "NeuralNetwork.h"
#include "Evaluator.h"
#include <iostream>
#include <filesystem>
#include <string>

int main() {
    unsigned int seed = 42;
    std::string model_load_path = "models/best_model.dat";

    // Check if the model file exists
    if (!std::filesystem::exists(model_load_path)) {
        std::cerr << "Model file does not exist: " << model_load_path << "\n";
        return 1;
    }

    // Initialize Neural Network with same architecture
    int input_size = 784; // 28x28 pixels
    std::vector<int> hidden_layers = { 128, 64 };
    int output_size = 10; // Digits 0-9
    NeuralNetwork nn(input_size, hidden_layers, output_size, seed);

    // Load the trained model
    nn.load_model(model_load_path);

    // Interactive Inference Loop
    std::cout << "------------------------------\n";
    std::cout << "Neural Network Inference\n";
    std::cout << "Enter 784 comma-separated pixel values (0-255) or type 'exit' to quit.\n";
    std::cout << "Example: 0,0,0,...,255\n";
    std::cout << "------------------------------\n";

    while (true) {
        std::cout << "Enter pixel values: ";
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input.empty()) {
            std::cout << "No input detected. Please enter 784 comma-separated values.\n";
            continue;
        }

        if (user_input == "exit" || user_input == "quit") {
            std::cout << "Exiting inference.\n";
            break;
        }

        int classification = nn.classify_user_input(user_input);
        if (classification != -1) {
            std::cout << "Predicted Digit: " << classification << "\n";
        } else {
            std::cout << "Failed to classify the input. Please check your input format and values.\n";
        }

        std::cout << "------------------------------\n";
    }

    return 0;
}