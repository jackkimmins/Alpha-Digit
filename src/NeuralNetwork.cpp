#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <future>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <cstdint>

// Constructor
NeuralNetwork::NeuralNetwork(int input_size, const std::vector<int>& hidden_layers_sizes, int output_size, unsigned int seed) : input_size(input_size), output_size(output_size), base_seed(seed), rng_weights(seed + 1) {
    // Layer Sizes
    layer_sizes.push_back(input_size);
    layer_sizes.insert(layer_sizes.end(), hidden_layers_sizes.begin(), hidden_layers_sizes.end());
    layer_sizes.push_back(output_size);

    std::cout << "Network Architecture: ";
    for (size_t i = 0; i < layer_sizes.size(); ++i) std::cout << layer_sizes[i] << (i < layer_sizes.size() - 1 ? " -> " : "\n");

    // Initialise weights and biases with random values
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        int rows = layer_sizes[i];
        int cols = layer_sizes[i - 1];
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(cols));

        // Initialise weights
        std::vector<std::vector<double>> layer_weights(rows, std::vector<double>(cols));
        for (auto& row : layer_weights)
            for (auto& weight : row)
                weight = dist(rng_weights);
        weights.push_back(layer_weights);

        // Initialise biases
        std::vector<double> layer_biases(rows);
        for (auto& bias : layer_biases) bias = dist(rng_weights);
        biases.push_back(layer_biases);

        // Initialise Adam optimizer parameters to zero
        m_w.emplace_back(rows, std::vector<double>(cols, 0.0));
        v_w.emplace_back(rows, std::vector<double>(cols, 0.0));
        m_b.emplace_back(rows, 0.0);
        v_b.emplace_back(rows, 0.0);
    }
}

// Train the neural network with the given hyperparameters
void NeuralNetwork::train(const std::vector<Database::DataPoint>& train_data, const std::vector<Database::DataPoint>& validation_data, int epochs, int batch_size, double initial_learning_rate, double decay_rate, int decay_steps, bool early_stopping, int patience) {
    if (train_data.empty()) {
        std::cerr << "Training data is empty. Cannot train the network." << std::endl;
        return;
    }

    std::cout << "Starting training...\n";
    std::cout << "------------------------------\n";

    double best_validation_loss = std::numeric_limits<double>::max();
    int epochs_without_improvement = 0;
    learning_rate = initial_learning_rate;

    // Set precision for floating-point output
    std::cout << std::fixed << std::setprecision(5);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Update learning rate with exponential decay
        learning_rate = initial_learning_rate * std::pow(decay_rate, static_cast<double>(epoch) / decay_steps);

        // Shuffle training data
        std::vector<Database::DataPoint> shuffled_data = train_data;
        std::mt19937 rng_train(base_seed + 2 + epoch);
        std::shuffle(shuffled_data.begin(), shuffled_data.end(), rng_train);

        // Process mini-batches
        for (size_t i = 0; i < shuffled_data.size(); i += batch_size) {
            size_t end = std::min(shuffled_data.size(), i + batch_size);
            std::vector<Database::DataPoint> batch(shuffled_data.begin() + i, shuffled_data.begin() + end);
            update_mini_batch(batch);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // Compute loss and accuracy
        double train_loss = compute_loss(train_data);
        double train_accuracy = compute_accuracy(train_data);
        double validation_loss = compute_loss(validation_data);
        double validation_accuracy = compute_accuracy(validation_data);

        std::cout << "Epoch " << epoch + 1 << " / " << epochs << "\n";
        std::cout << "Time Elapsed: " << elapsed.count() << " seconds\n";
        std::cout << "Learning Rate: " << learning_rate << "\n";
        std::cout << "Training Loss: " << train_loss << "\t\t Training Accuracy: " << train_accuracy * 100 << "%\n";
        std::cout << "Validation Loss: " << validation_loss << "\t Validation Accuracy: " << validation_accuracy * 100 << "%\n";

        // Indicate if the model was saved
        if (early_stopping) {
            if (validation_loss < best_validation_loss) {
                best_validation_loss = validation_loss;
                epochs_without_improvement = 0;
                save_model("models/best_model.dat");
                std::cout << "Model improved. Saved to models/best_model.dat\n";
            } else {
                epochs_without_improvement++;
                std::cout << "No improvement for " << epochs_without_improvement << " epoch(s).\n";
                if (epochs_without_improvement >= patience) {
                    std::cout << "Early stopping triggered after " << epoch + 1 << " epochs.\n";
                    load_model("models/best_model.dat");
                    break;
                }
            }
        }

        std::cout << "------------------------------\n";
    }
}

// Inference using the trained model
int NeuralNetwork::predict(const std::vector<double>& input) const {
    if (input.size() != static_cast<size_t>(input_size)) {
        std::cerr << "Input size mismatch. Expected " << input_size << " but got " << input.size() << "." << std::endl;
        return -1;
    }

    std::vector<double> output = feedforward(input);
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

// Save the model to file
void NeuralNetwork::save_model(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for saving model: " << filename << std::endl;
        return;
    }

    uint32_t num_layers = static_cast<uint32_t>(layer_sizes.size());
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(layer_sizes.data()), num_layers * sizeof(uint32_t));

    for (size_t l = 0; l < weights.size(); ++l) {
        uint32_t rows = static_cast<uint32_t>(weights[l].size());
        uint32_t cols = static_cast<uint32_t>(weights[l][0].size());
        out.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
        for (const auto& row : weights[l]) out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));

        uint32_t biases_size = static_cast<uint32_t>(biases[l].size());
        out.write(reinterpret_cast<const char*>(&biases_size), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(biases[l].data()), biases_size * sizeof(double));
    }

    out.close();
    std::cout << "Model saved to " << filename << "\n";
}

// Load the model from file
void NeuralNetwork::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open file for loading model: " << filename << std::endl;
        return;
    }

    uint32_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    std::cout << "Number of layers: " << num_layers << std::endl;

    if (num_layers < 2) { // Minimum 2 layers (input and output)
        std::cerr << "Error: Number of layers is too small: " << num_layers << std::endl;
        return;
    }

    layer_sizes.resize(num_layers);
    in.read(reinterpret_cast<char*>(layer_sizes.data()), num_layers * sizeof(uint32_t));
    for (size_t i = 0; i < layer_sizes.size(); ++i) std::cout << "Layer " << i << " size: " << layer_sizes[i] << std::endl;

    weights.clear();
    biases.clear();
    m_w.clear();
    v_w.clear();
    m_b.clear();
    v_b.clear();

    for (size_t l = 0; l < num_layers - 1; ++l) {
        uint32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        std::cout << "Layer " << l << " dimensions (rows x cols): " << rows << " x " << cols << std::endl;

        // Sanity check for dimensions
        if (rows != static_cast<uint32_t>(layer_sizes[l + 1]) || cols != static_cast<uint32_t>(layer_sizes[l])) {
            std::cerr << "Error: Unexpected dimensions for layer " << l 
                      << " (expected " << layer_sizes[l + 1] << " x " << layer_sizes[l] 
                      << ", got " << rows << " x " << cols << ")\n";
            return;
        }

        // Read weights for the layer
        std::vector<std::vector<double>> layer_weights(rows, std::vector<double>(cols));
        for (size_t r = 0; r < rows; ++r) {
            in.read(reinterpret_cast<char*>(layer_weights[r].data()), cols * sizeof(double));
            if (in.gcount() != static_cast<std::streamsize>(cols * sizeof(double))) {
                std::cerr << "Error: Failed to read weights for layer " << l << ", row " << r << std::endl;
                return;
            }
        }
        weights.push_back(layer_weights);

        // Read biases for the layer
        uint32_t biases_size;
        in.read(reinterpret_cast<char*>(&biases_size), sizeof(uint32_t));
        std::cout << "Biases size for layer " << l << ": " << biases_size << std::endl;

        // Another sanity check; biases should match next layer size
        if (biases_size != static_cast<uint32_t>(layer_sizes[l + 1])) {
            std::cerr << "Error: Biases size (" << biases_size 
                      << ") does not match layer output size (" << layer_sizes[l + 1] 
                      << ") for layer " << l << std::endl;
            return;
        }

        std::vector<double> layer_biases(biases_size);
        in.read(reinterpret_cast<char*>(layer_biases.data()), biases_size * sizeof(double));
        if (in.gcount() != static_cast<std::streamsize>(biases_size * sizeof(double))) {
            std::cerr << "Error: Failed to read biases for layer " << l << std::endl;
            return;
        }
        biases.push_back(layer_biases);

        // ReInit Adam optimiser parameters to zero
        m_w.emplace_back(rows, std::vector<double>(cols, 0.0));
        v_w.emplace_back(rows, std::vector<double>(cols, 0.0));
        m_b.emplace_back(layer_sizes[l + 1], 0.0);
        v_b.emplace_back(layer_sizes[l + 1], 0.0);
    }

    in.close();
    std::cout << "Model loaded from " << filename << "\n";
}

// Take user input, validate and classify
int NeuralNetwork::classify_user_input(const std::string& input) const {
    std::vector<double> features;
    std::stringstream ss(input);
    std::string token;
    size_t count = 0;

    while (std::getline(ss, token, ',')) {
        if (count >= 784) {
            std::cerr << "Error: More than 784 values provided.\n";
            return -1;
        }
        try {
            double val = std::stod(token);
            if (val < 0.0 || val > 255.0) {
                std::cerr << "Error: Pixel value out of range (0-255): " << val << "\n";
                return -1;
            }
            features.push_back(val / 255.0);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing value '" << token << "': " << e.what() << "\n";
            return -1;
        }
        ++count;
    }

    if (count != 784) {
        std::cerr << "Error: Expected 784 values, but received " << count << ".\n";
        return -1;
    }

    return predict(features);
}

// Feedforward pass through the network
std::vector<double> NeuralNetwork::feedforward(const std::vector<double>& input) const {
    std::vector<double> activation = input;
    for (size_t i = 0; i < weights.size() - 1; ++i) activation = sigmoid(vec_add(mat_vec_mul(weights[i], activation), biases[i]));
    activation = softmax(vec_add(mat_vec_mul(weights.back(), activation), biases.back()));
    return activation;
}

// Update weights and biases using mini-batch gradient descent
void NeuralNetwork::update_mini_batch(const std::vector<Database::DataPoint>& batch) {
    if (batch.empty()) return;

    // Initialise gradients to zero
    auto nabla_w = weights;
    auto nabla_b = biases;
    for (auto& layer : nabla_w)
        for (auto& neuron : layer)
            std::fill(neuron.begin(), neuron.end(), 0.0);
    for (auto& layer : nabla_b) std::fill(layer.begin(), layer.end(), 0.0);

    // Number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;
    size_t batch_size_per_thread = batch.size() / num_threads;
    size_t remaining = batch.size() % num_threads;

    // Launch asynchronous tasks to compute partial gradients
    std::vector<std::future<std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>>> futures;
    size_t start_idx = 0;
    for (unsigned int thread_num = 0; thread_num < num_threads; ++thread_num) {
        size_t end_idx = start_idx + batch_size_per_thread + (thread_num < remaining ? 1 : 0);
        std::vector<Database::DataPoint> subset(batch.begin() + start_idx, batch.begin() + end_idx);
        futures.emplace_back(std::async(std::launch::async, &NeuralNetwork::compute_partial_gradients, this, subset));
        start_idx = end_idx;
    }

    // Collect and accumulate partial gradients
    for (auto& f : futures) {
        auto partial = f.get();
        auto& partial_nabla_w = partial.first;
        auto& partial_nabla_b = partial.second;

        for (size_t l = 0; l < nabla_w.size(); ++l) {
            for (size_t r = 0; r < nabla_w[l].size(); ++r)
                for (size_t c = 0; c < nabla_w[l][r].size(); ++c)
                    nabla_w[l][r][c] += partial_nabla_w[l][r][c];
            for (size_t r = 0; r < nabla_b[l].size(); ++r) nabla_b[l][r] += partial_nabla_b[l][r];
        }
    }

    // Update weights and biases using Adam optimiser - found the following in this paper: https://arxiv.org/abs/1412.6980
    ++t_step;
    for (size_t l = 0; l < weights.size(); ++l) {                                                   // For each layer
        for (size_t r = 0; r < weights[l].size(); ++r) {                                            // For each neuron in the layer
            for (size_t c = 0; c < weights[l][r].size(); ++c) {                                     // For each weight in the neuron
                double grad = nabla_w[l][r][c] / batch.size();                                      // Compute average gradient
                m_w[l][r][c] = beta1 * m_w[l][r][c] + (1 - beta1) * grad;                           // Update first moment estimate
                v_w[l][r][c] = beta2 * v_w[l][r][c] + (1 - beta2) * grad * grad;                    // Update second moment estimate
                double m_hat = m_w[l][r][c] / (1 - std::pow(beta1, t_step));                        // Bias-corrected first moment estimate
                double v_hat = v_w[l][r][c] / (1 - std::pow(beta2, t_step));                        // Bias-corrected second moment estimate
                weights[l][r][c] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);           // Update weights
            }
        }
        for (size_t r = 0; r < biases[l].size(); ++r) {                                             // For each bias in the layer
            double grad = nabla_b[l][r] / batch.size();                                             // Compute average gradient
            m_b[l][r] = beta1 * m_b[l][r] + (1 - beta1) * grad;                                     // Update first moment estimate
            v_b[l][r] = beta2 * v_b[l][r] + (1 - beta2) * grad * grad;                              // Update second moment estimate
            double m_hat = m_b[l][r] / (1 - std::pow(beta1, t_step));                               // Bias-corrected first moment estimate
            double v_hat = v_b[l][r] / (1 - std::pow(beta2, t_step));                               // Bias-corrected second moment estimate
            biases[l][r] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);                   // Update biases
        }
    }
}

// Compute loss on the given data
double NeuralNetwork::compute_loss(const std::vector<Database::DataPoint>& data) const {
    double total_loss = 0.0;
    for (const auto& data_point : data) {
        std::vector<double> output = feedforward(data_point.features);
        double loss = -std::log(output[data_point.label] + 1e-15); // Just trying to prevent log(0), not sure if this is the best way 😇
        total_loss += loss;
    }
    return total_loss / data.size();
}

// Compute accuracy on the given data
double NeuralNetwork::compute_accuracy(const std::vector<Database::DataPoint>& data) const {
    int correct = 0;
    for (const auto& data_point : data) {
        int prediction = predict(data_point.features);
        if (prediction == data_point.label) ++correct;
    }
    return static_cast<double>(correct) / data.size();
}

// Compute gradients using backpropagation
std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> NeuralNetwork::backprop(const Database::DataPoint& data_point) const {
    std::vector<std::vector<double>> activations;       // Activations of each layer
    std::vector<std::vector<double>> zs;                // Weighted inputs for each layer

    // Forward pass
    std::vector<double> activation = data_point.features;
    activations.push_back(activation);
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z = vec_add(mat_vec_mul(weights[i], activation), biases[i]);
        zs.push_back(z);
        activation = (i == weights.size() - 1) ? softmax(z) : sigmoid(z);
        activations.push_back(activation);
    }

    // Initialise gradients to zero
    auto nabla_w = weights;
    auto nabla_b = biases;
    for (auto& layer : nabla_w)
        for (auto& neuron : layer)
            std::fill(neuron.begin(), neuron.end(), 0.0);
    for (auto& layer : nabla_b) std::fill(layer.begin(), layer.end(), 0.0);

    // Compute output error
    std::vector<double> delta = cost_derivative(activations.back(), data_point.label);
    nabla_b.back() = delta;
    nabla_w.back() = outer_product(delta, activations[activations.size() - 2]);

    // Backward pass
    for (int l = static_cast<int>(weights.size()) - 2; l >= 0; --l) {
        std::vector<double> sp = sigmoid_prime(zs[l]);
        std::vector<double> transposed_weights = flatten(transpose(weights[l + 1]));

        size_t rows = layer_sizes[l + 1];
        size_t cols = weights[l + 1].size();
        std::vector<double> weighted_delta = mat_vec_mul(transposed_weights, delta, rows, cols);

        delta = hadamard_product(weighted_delta, sp);
        nabla_b[l] = delta;
        nabla_w[l] = outer_product(delta, activations[l]);
    }

    return { nabla_w, nabla_b };
}


// The following functions are just utility functions for this class:

std::vector<double> NeuralNetwork::sigmoid(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    for (size_t i = 0; i < z.size(); ++i)
        result[i] = 1.0 / (1.0 + std::exp(-z[i]));
    return result;
}

std::vector<double> NeuralNetwork::sigmoid_prime(const std::vector<double>& z) const {
    std::vector<double> sig = sigmoid(z);
    std::vector<double> result(z.size());
    for (size_t i = 0; i < z.size(); ++i)
        result[i] = sig[i] * (1 - sig[i]);
    return result;
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    double max_z = *std::max_element(z.begin(), z.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        result[i] = std::exp(z[i] - max_z);
        sum_exp += result[i];
    }
    for (double& val : result)
        val /= sum_exp;
    return result;
}

std::vector<double> NeuralNetwork::cost_derivative(const std::vector<double>& output_activations, int label) const {
    std::vector<double> result(output_activations.size(), 0.0);
    if (label < 0 || label >= static_cast<int>(output_activations.size())) {
        std::cerr << "Invalid label: " << label << ". Expected between 0 and " << output_activations.size() - 1 << "." << std::endl;
        return result;
    }
    result[label] = 1.0;
    for (size_t i = 0; i < output_activations.size(); ++i)
        result[i] = output_activations[i] - result[i];
    return result;
}

std::vector<double> NeuralNetwork::vec_add(const std::vector<double>& a, const std::vector<double>& b) const {
    assert(a.size() == b.size() && "Vector sizes must match for addition.");
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

std::vector<double> NeuralNetwork::hadamard_product(const std::vector<double>& a, const std::vector<double>& b) const {
    assert(a.size() == b.size() && "Vector sizes must match for Hadamard product.");
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * b[i];
    return result;
}

std::vector<double> NeuralNetwork::mat_vec_mul(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) const {
    assert(mat.empty() || mat[0].size() == vec.size() && "Matrix columns must match vector size for multiplication.");
    std::vector<double> result(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i)
        for (size_t j = 0; j < mat[i].size(); ++j)
            result[i] += mat[i][j] * vec[j];
    return result;
}

std::vector<double> NeuralNetwork::mat_vec_mul(const std::vector<double>& mat_flat, const std::vector<double>& vec, size_t rows, size_t cols) const {
    assert(mat_flat.size() == rows * cols && "Flattened matrix size does not match specified dimensions.");
    assert(cols == vec.size() && "Matrix columns must match vector size for multiplication.");
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[i] += mat_flat[i * cols + j] * vec[j];
    return result;
}

std::vector<std::vector<double>> NeuralNetwork::outer_product(const std::vector<double>& a, const std::vector<double>& b) const {
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b.size()));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            result[i][j] = a[i] * b[j];
    return result;
}

std::vector<std::vector<double>> NeuralNetwork::transpose(const std::vector<std::vector<double>>& mat) const {
    if (mat.empty()) return {};
    std::vector<std::vector<double>> result(mat[0].size(), std::vector<double>(mat.size()));
    for (size_t i = 0; i < mat.size(); ++i)
        for (size_t j = 0; j < mat[i].size(); ++j)
            result[j][i] = mat[i][j];
    return result;
}

std::vector<double> NeuralNetwork::flatten(const std::vector<std::vector<double>>& mat) const {
    std::vector<double> flat;
    for (const auto& row : mat) flat.insert(flat.end(), row.begin(), row.end());
    return flat;
}

std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> NeuralNetwork::compute_partial_gradients(const std::vector<Database::DataPoint>& subset) const {
    // Initialise local gradients to zero
    auto partial_nabla_w = weights;
    auto partial_nabla_b = biases;
    for (auto& layer : partial_nabla_w)
        for (auto& neuron : layer)
            std::fill(neuron.begin(), neuron.end(), 0.0);
    for (auto& layer : partial_nabla_b)
        std::fill(layer.begin(), layer.end(), 0.0);

    // Accumulate gradients from each data point in the subset
    for (const auto& data_point : subset) {
        auto delta = backprop(data_point);
        auto& delta_nabla_w = delta.first;
        auto& delta_nabla_b = delta.second;

        for (size_t l = 0; l < partial_nabla_w.size(); ++l) {
            for (size_t r = 0; r < partial_nabla_w[l].size(); ++r)
                for (size_t c = 0; c < partial_nabla_w[l][r].size(); ++c)
                    partial_nabla_w[l][r][c] += delta_nabla_w[l][r][c];
            for (size_t r = 0; r < partial_nabla_b[l].size(); ++r) partial_nabla_b[l][r] += delta_nabla_b[l][r];
        }
    }

    return { partial_nabla_w, partial_nabla_b };
}