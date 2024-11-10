#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include <utility>
#include "Database.h"

class NeuralNetwork {
public:
    // Constructor sets up the architecture and initializes weights
    NeuralNetwork(int input_size, const std::vector<int>& hidden_layers_sizes, int output_size, unsigned int seed);

    // Trains the network with specified parameters
    void train(const std::vector<Database::DataPoint>& train_data, const std::vector<Database::DataPoint>& validation_data, int epochs, int batch_size, double initial_learning_rate, double decay_rate, int decay_steps, bool early_stopping = false, int patience = 5);

    // Predicts the label for a given input vector
    int predict(const std::vector<double>& input) const;

    // Saves the model parameters to a file
    void save_model(const std::string& filename) const;

    // Loads the model parameters from a file
    void load_model(const std::string& filename);

    // Inference Method - Classify user input string
    int classify_user_input(const std::string& input) const;

private:
    // Forward declarations of private helper functions
    std::vector<double> feedforward(const std::vector<double>& input) const;
    void update_mini_batch(const std::vector<Database::DataPoint>& batch);
    double compute_loss(const std::vector<Database::DataPoint>& data) const;
    double compute_accuracy(const std::vector<Database::DataPoint>& data) const;
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> backprop(const Database::DataPoint& data_point) const;

    // Activation functions
    std::vector<double> sigmoid(const std::vector<double>& z) const;
    std::vector<double> sigmoid_prime(const std::vector<double>& z) const;
    std::vector<double> softmax(const std::vector<double>& z) const;
    std::vector<double> cost_derivative(const std::vector<double>& output_activations, int label) const;

    // Utility functions
    std::vector<double> vec_add(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<double> hadamard_product(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<double> mat_vec_mul(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) const;
    std::vector<double> mat_vec_mul(const std::vector<double>& mat_flat, const std::vector<double>& vec, size_t rows, size_t cols) const;
    std::vector<std::vector<double>> outer_product(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& mat) const;
    std::vector<double> flatten(const std::vector<std::vector<double>>& mat) const;
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> compute_partial_gradients(const std::vector<Database::DataPoint>& subset) const;

    // Member variables
    int input_size;
    int output_size;
    std::vector<int> layer_sizes;                               // Sizes of all layers
    std::vector<std::vector<double>> biases;                    // Biases for each layer
    std::vector<std::vector<std::vector<double>>> weights;      // Weights between layers

    // Adam optimizer parameters
    std::vector<std::vector<std::vector<double>>> m_w;          // First moment estimates for weights
    std::vector<std::vector<double>> m_b;                       // First moment estimates for biases
    std::vector<std::vector<std::vector<double>>> v_w;          // Second moment estimates for weights
    std::vector<std::vector<double>> v_b;                       // Second moment estimates for biases

    double beta1 = 0.9;                                         // Exponential decay rate for first moment
    double beta2 = 0.999;                                       // Exponential decay rate for second moment
    double epsilon = 1e-8;                                      // Constant to prevent division by zero
    double learning_rate = 0.001;                               // Learning rate
    int t_step = 0;                                             // Time step for Adam optimizer

    unsigned int base_seed;                                     // Base seed for deterministic behavior
    std::mt19937 rng_weights;                                   // RNG for weight initialization
};

#endif