#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cassert>
#include <future>
#include <limits>
#include <iomanip>

class Database {
public:
    struct DataPoint {
        int label;                      // Digit label (0-9)
        std::vector<double> features;   // normalised pixel values (0 to 1)
    };

    // Constructor that loads data from a CSV file with a given seed
    Database(const std::string& filepath, unsigned int seed) : rng(seed) { load_data(filepath); }

    // Splits data into training, validation, and test sets
    void split_data(double train_ratio, double validation_ratio = 0.1) {
        size_t train_size = static_cast<size_t>(data.size() * train_ratio);
        size_t validation_size = static_cast<size_t>(data.size() * validation_ratio);
        train_data.assign(data.begin(), data.begin() + train_size);
        validation_data.assign(data.begin() + train_size, data.begin() + train_size + validation_size);
        test_data.assign(data.begin() + train_size + validation_size, data.end());

        std::cout << "Split data into " << train_data.size() << " training, "
                  << validation_data.size() << " validation, and "
                  << test_data.size() << " testing data points.\n";
    }

    const std::vector<DataPoint>& get_train_data() const { return train_data; }
    const std::vector<DataPoint>& get_validation_data() const { return validation_data; }
    const std::vector<DataPoint>& get_test_data() const { return test_data; }

private:
    std::vector<DataPoint> data;                // All data points
    std::vector<DataPoint> train_data;          // Training data
    std::vector<DataPoint> validation_data;     // Validation data
    std::vector<DataPoint> test_data;           // Test data
    std::mt19937 rng;                           // Random number generator

    // Loads data from a CSV file
    void load_data(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open the dataset file: " << filepath << std::endl;
            exit(1);
        }

        std::string line;
        if (!std::getline(file, line)) { // Skip header
            std::cerr << "Dataset file is empty." << std::endl;
            exit(1);
        }

        const size_t expected_columns = 785; // 1 label + 784 features
        size_t line_number = 1;

        while (std::getline(file, line)) {
            ++line_number;
            if (line.empty()) continue;

            DataPoint data_point;
            std::stringstream ss(line);
            std::string cell;
            size_t cell_idx = 0;

            while (std::getline(ss, cell, ',')) {
                if (cell_idx == 0) {
                    try {
                        data_point.label = std::stoi(cell);
                        if (data_point.label < 0 || data_point.label > 9) throw std::out_of_range("Label out of range");
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing label at line " << line_number << ": " << e.what() << std::endl;
                        data_point.label = -1;
                        break;
                    }
                } else {
                    try {
                        double pixel = std::stod(cell);
                        if (pixel < 0.0 || pixel > 255.0)
                            throw std::out_of_range("Pixel value out of range");
                        data_point.features.push_back(pixel / 255.0);
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing pixel at line " << line_number << ", column " << cell_idx + 1 << ": " << e.what() << std::endl;
                        data_point.features.clear();
                        break;
                    }
                }
                ++cell_idx;
            }

            if (cell_idx != expected_columns) {
                std::cerr << "Warning: Line " << line_number << " has " << cell_idx << " columns, expected " << expected_columns << ". Skipping this line.\n";
                continue;
            }

            if (data_point.label != -1 && data_point.features.size() == 784) data.push_back(data_point);
            else std::cerr << "Warning: Invalid data point at line " << line_number << ". Skipping.\n";
        }

        file.close();

        if (data.empty()) {
            std::cerr << "No valid data loaded from the dataset." << std::endl;
            exit(1);
        }

        std::shuffle(data.begin(), data.end(), rng);
        std::cout << "Loaded " << data.size() << " data points from " << filepath << "\n";
    }
};

class NeuralNetwork {
public:
    // Constructor sets up the architecture and initialises weights
    NeuralNetwork(int input_size, const std::vector<int>& hidden_layers_sizes, int output_size, unsigned int seed) : input_size(input_size), output_size(output_size), base_seed(seed), rng_weights(seed + 1) {
        // Define layer sizes
        layer_sizes.push_back(input_size);
        layer_sizes.insert(layer_sizes.end(), hidden_layers_sizes.begin(), hidden_layers_sizes.end());
        layer_sizes.push_back(output_size);

        std::cout << "Network Architecture: ";
        for (size_t i = 0; i < layer_sizes.size(); ++i) std::cout << layer_sizes[i] << (i < layer_sizes.size() - 1 ? " -> " : "\n");

        initialize_weights();
    }

    // Trains the network with specified parameters
    void train(const std::vector<Database::DataPoint>& train_data,
               const std::vector<Database::DataPoint>& validation_data,
               int epochs, int batch_size,
               double initial_learning_rate, double decay_rate, int decay_steps,
               bool early_stopping = false, int patience = 5) {
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
                    save_model("best_model.dat");
                    std::cout << "Model improved. Saved to best_model.dat\n";
                } else {
                    epochs_without_improvement++;
                    std::cout << "No improvement for " << epochs_without_improvement << " epoch(s).\n";
                    if (epochs_without_improvement >= patience) {
                        std::cout << "Early stopping triggered after " << epoch + 1 << " epochs.\n";
                        load_model("best_model.dat");
                        break;
                    }
                }
            }

            std::cout << "------------------------------\n";
        }
    }

    // Predicts the label for a given input
    int predict(const std::vector<double>& input) const {
        if (input.size() != input_size) {
            std::cerr << "Input size mismatch. Expected " << input_size << " but got " << input.size() << "." << std::endl;
            return -1;
        }

        std::vector<double> output = feedforward(input);
        return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    }

    // Saves the model parameters to a file
    void save_model(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Failed to open file for saving model: " << filename << std::endl;
            return;
        }

        size_t num_layers = layer_sizes.size();
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(layer_sizes.data()), num_layers * sizeof(int));

        for (size_t l = 0; l < weights.size(); ++l) {
            size_t rows = weights[l].size();
            size_t cols = weights[l][0].size();
            out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
            for (const auto& row : weights[l]) out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
            size_t biases_size = biases[l].size();
            out.write(reinterpret_cast<const char*>(&biases_size), sizeof(size_t));
            out.write(reinterpret_cast<const char*>(biases[l].data()), biases_size * sizeof(double));
        }

        out.close();
    }

    // Loads the model parameters from a file
    void load_model(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Failed to open file for loading model: " << filename << std::endl;
            return;
        }

        size_t num_layers;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
        layer_sizes.resize(num_layers);
        in.read(reinterpret_cast<char*>(layer_sizes.data()), num_layers * sizeof(int));

        weights.clear();
        biases.clear();

        for (size_t l = 0; l < num_layers - 1; ++l) {
            size_t rows, cols;
            in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
            in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

            std::vector<std::vector<double>> layer_weights(rows, std::vector<double>(cols));
            for (auto& row : layer_weights) in.read(reinterpret_cast<char*>(row.data()), cols * sizeof(double));
            weights.push_back(layer_weights);

            size_t biases_size;
            in.read(reinterpret_cast<char*>(&biases_size), sizeof(size_t));
            std::vector<double> layer_biases(biases_size);
            in.read(reinterpret_cast<char*>(layer_biases.data()), biases_size * sizeof(double));
            biases.push_back(layer_biases);
        }

        in.close();
        std::cout << "Model loaded from " << filename << "\n";
    }

private:
    int input_size;
    int output_size;
    std::vector<int> layer_sizes;                               // Sizes of all layers
    std::vector<std::vector<double>> biases;                    // Biases for each layer
    std::vector<std::vector<std::vector<double>>> weights;      // Weights between layers

    // Adam optimiser parameters
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
    std::mt19937 rng_weights;                                   // RNG for weight initialisation

    // Initialises weights and biases with small random values
    void initialize_weights() {
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

            // Initialise Adam optimiser parameters to zero
            m_w.emplace_back(rows, std::vector<double>(cols, 0.0));
            v_w.emplace_back(rows, std::vector<double>(cols, 0.0));
            m_b.emplace_back(rows, 0.0);
            v_b.emplace_back(rows, 0.0);
        }
    }

    // Performs feedforward operation
    std::vector<double> feedforward(const std::vector<double>& input) const {
        std::vector<double> activation = input;
        for (size_t i = 0; i < weights.size() - 1; ++i) activation = sigmoid(vec_add(mat_vec_mul(weights[i], activation), biases[i]));
        activation = softmax(vec_add(mat_vec_mul(weights.back(), activation), biases.back()));
        return activation;
    }

    // Updates weights and biases using gradients from the mini-batch
    void update_mini_batch(const std::vector<Database::DataPoint>& batch) {
        if (batch.empty()) return;

        // Initialise gradients to zero
        auto nabla_w = weights;
        auto nabla_b = biases;
        for (auto& layer : nabla_w)
            for (auto& neuron : layer)
                std::fill(neuron.begin(), neuron.end(), 0.0);
        for (auto& layer : nabla_b)
            std::fill(layer.begin(), layer.end(), 0.0);

        // Determine the number of threads
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
                for (size_t r = 0; r < nabla_b[l].size(); ++r)
                    nabla_b[l][r] += partial_nabla_b[l][r];
            }
        }

        // Update weights and biases using Adam optimiser
        ++t_step;
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t r = 0; r < weights[l].size(); ++r) {
                for (size_t c = 0; c < weights[l][r].size(); ++c) {
                    double grad = nabla_w[l][r][c] / batch.size();
                    m_w[l][r][c] = beta1 * m_w[l][r][c] + (1 - beta1) * grad;
                    v_w[l][r][c] = beta2 * v_w[l][r][c] + (1 - beta2) * grad * grad;
                    double m_hat = m_w[l][r][c] / (1 - std::pow(beta1, t_step));
                    double v_hat = v_w[l][r][c] / (1 - std::pow(beta2, t_step));
                    weights[l][r][c] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
            for (size_t r = 0; r < biases[l].size(); ++r) {
                double grad = nabla_b[l][r] / batch.size();
                m_b[l][r] = beta1 * m_b[l][r] + (1 - beta1) * grad;
                v_b[l][r] = beta2 * v_b[l][r] + (1 - beta2) * grad * grad;
                double m_hat = m_b[l][r] / (1 - std::pow(beta1, t_step));
                double v_hat = v_b[l][r] / (1 - std::pow(beta2, t_step));
                biases[l][r] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }

    // Computes the loss over a dataset
    double compute_loss(const std::vector<Database::DataPoint>& data) const {
        double total_loss = 0.0;
        for (const auto& data_point : data) {
            std::vector<double> output = feedforward(data_point.features);
            double loss = -std::log(output[data_point.label] + 1e-15); // Prevent log(0)
            total_loss += loss;
        }
        return total_loss / data.size();
    }

    // Computes the accuracy over a dataset
    double compute_accuracy(const std::vector<Database::DataPoint>& data) const {
        int correct = 0;
        for (const auto& data_point : data) {
            int prediction = predict(data_point.features);
            if (prediction == data_point.label) ++correct;
        }
        return static_cast<double>(correct) / data.size();
    }

    // Performs backpropagation to compute gradients for a data point
    // Reference: https://en.wikipedia.org/wiki/Backpropagation
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    backprop(const Database::DataPoint& data_point) const {
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
        for (auto& layer : nabla_b)
            std::fill(layer.begin(), layer.end(), 0.0);

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

    // Sigmoid activation function
    // Reference: https://en.wikipedia.org/wiki/Sigmoid_function
    std::vector<double> sigmoid(const std::vector<double>& z) const {
        std::vector<double> result(z.size());
        for (size_t i = 0; i < z.size(); ++i)
            result[i] = 1.0 / (1.0 + std::exp(-z[i]));
        return result;
    }

    // Derivative of the sigmoid function
    std::vector<double> sigmoid_prime(const std::vector<double>& z) const {
        std::vector<double> sig = sigmoid(z);
        std::vector<double> result(z.size());
        for (size_t i = 0; i < z.size(); ++i)
            result[i] = sig[i] * (1 - sig[i]);
        return result;
    }

    // Softmax activation function
    // Reference: https://en.wikipedia.org/wiki/Softmax_function
    std::vector<double> softmax(const std::vector<double>& z) const {
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

    // Computes the derivative of the cost function
    std::vector<double> cost_derivative(const std::vector<double>& output_activations, int label) const {
        std::vector<double> result(output_activations.size(), 0.0);
        if (label < 0 || label >= static_cast<int>(output_activations.size())) {
            std::cerr << "Invalid label: " << label << ". Expected between 0 and " << output_activations.size() - 1 << "." << std::endl;
            return result;
        }
        result[label] = 1.0;
        for (size_t i = 0; i < output_activations.size(); ++i) result[i] = output_activations[i] - result[i];
        return result;
    }

    // Adds two vectors
    std::vector<double> vec_add(const std::vector<double>& a, const std::vector<double>& b) const {
        assert(a.size() == b.size() && "Vector sizes must match for addition.");
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
        return result;
    }

    // Element-wise multiplication of two vectors
    std::vector<double> hadamard_product(const std::vector<double>& a, const std::vector<double>& b) const {
        assert(a.size() == b.size() && "Vector sizes must match for Hadamard product.");
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * b[i];
        return result;
    }

    // Multiplies a matrix with a vector
    std::vector<double> mat_vec_mul(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) const {
        assert(mat.empty() || mat[0].size() == vec.size() && "Matrix columns must match vector size for multiplication.");
        std::vector<double> result(mat.size(), 0.0);
        for (size_t i = 0; i < mat.size(); ++i)
            for (size_t j = 0; j < mat[i].size(); ++j)
                result[i] += mat[i][j] * vec[j];
        return result;
    }

    // Overloaded matrix-vector multiplication for transposed weights
    std::vector<double> mat_vec_mul(const std::vector<double>& mat_flat, const std::vector<double>& vec, size_t rows, size_t cols) const {
        assert(mat_flat.size() == rows * cols && "Flattened matrix size does not match specified dimensions.");
        assert(cols == vec.size() && "Matrix columns must match vector size for multiplication.");
        std::vector<double> result(rows, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i] += mat_flat[i * cols + j] * vec[j];
        return result;
    }

    // Computes the outer product of two vectors
    std::vector<std::vector<double>> outer_product(const std::vector<double>& a, const std::vector<double>& b) const {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(b.size()));
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < b.size(); ++j)
                result[i][j] = a[i] * b[j];
        return result;
    }

    // Transposes a matrix
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& mat) const {
        if (mat.empty()) return {};
        std::vector<std::vector<double>> result(mat[0].size(), std::vector<double>(mat.size()));
        for (size_t i = 0; i < mat.size(); ++i)
            for (size_t j = 0; j < mat[i].size(); ++j)
                result[j][i] = mat[i][j];
        return result;
    }

    // Flattens a 2D matrix into a 1D vector (row-major order)
    std::vector<double> flatten(const std::vector<std::vector<double>>& mat) const {
        std::vector<double> flat;
        for (const auto& row : mat)
            flat.insert(flat.end(), row.begin(), row.end());
        return flat;
    }

    // Computes partial gradients for a subset of data points
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> compute_partial_gradients(const std::vector<Database::DataPoint>& subset) const {
        // Initialise local gradients to zero
        auto partial_nabla_w = weights;
        auto partial_nabla_b = biases;
        for (auto& layer : partial_nabla_w)
            for (auto& neuron : layer)
                std::fill(neuron.begin(), neuron.end(), 0.0);
        for (auto& layer : partial_nabla_b) std::fill(layer.begin(), layer.end(), 0.0);

        // Accumulate gradients from each data point in the subset
        for (const auto& data_point : subset) {
            auto delta = backprop(data_point);
            auto& delta_nabla_w = delta.first;
            auto& delta_nabla_b = delta.second;

            for (size_t l = 0; l < partial_nabla_w.size(); ++l) {
                for (size_t r = 0; r < partial_nabla_w[l].size(); ++r)
                    for (size_t c = 0; c < partial_nabla_w[l][r].size(); ++c)
                        partial_nabla_w[l][r][c] += delta_nabla_w[l][r][c];
                for (size_t r = 0; r < partial_nabla_b[l].size(); ++r)
                    partial_nabla_b[l][r] += delta_nabla_b[l][r];
            }
        }

        return { partial_nabla_w, partial_nabla_b };
    }
};

// Class to evaluate the neural network's performance
class Evaluator {
public:
    // Constructor that takes a reference to the neural network and test data
    Evaluator(const NeuralNetwork& nn, const std::vector<Database::DataPoint>& test_data)
        : nn(nn), test_data(test_data) {}

    // Evaluates the network and prints accuracy
    void evaluate() const {
        if (test_data.empty()) {
            std::cerr << "Test data is empty. Cannot evaluate the network." << std::endl;
            return;
        }

        int correct = 0;
        for (const auto& data_point : test_data) {
            int prediction = nn.predict(data_point.features);
            if (prediction == data_point.label) ++correct;
        }
        double accuracy = static_cast<double>(correct) / test_data.size() * 100.0;
        std::cout << "------------------------------\n";
        std::cout << "Final Evaluation on Test Data:\n";
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
        std::cout << "------------------------------\n";
    }

private:
    const NeuralNetwork& nn;
    const std::vector<Database::DataPoint>& test_data;
};

int main() {
    std::string dataset_path = "datasets/mnist_test.csv";
    unsigned int seed = 42;

    // Load and split dataset
    Database db(dataset_path, seed);
    db.split_data(0.8);

    // Network Architecture
    int input_size = 784; // 28x28 pixels
    std::vector<int> hidden_layers = { 16, 16 };
    int output_size = 10; // Digits 0-9

    // Initialise NN
    NeuralNetwork nn(input_size, hidden_layers, output_size, seed);

    // Training Parameters
    int epochs = 50;
    int batch_size = 32;
    double initial_learning_rate = 0.001;
    double decay_rate = 0.9;
    int decay_steps = 10;
    bool early_stopping = true;
    int patience = 5;

    // Train the neural network
    nn.train(db.get_train_data(), db.get_validation_data(), epochs, batch_size, initial_learning_rate, decay_rate, decay_steps, early_stopping, patience);

    // Evaluate the neural network
    Evaluator evaluator(nn, db.get_test_data());
    evaluator.evaluate();

    return 0;
}
