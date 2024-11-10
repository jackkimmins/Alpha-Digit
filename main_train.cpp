#include "Database.h"
#include "NeuralNetwork.h"
#include "Evaluator.h"
#include <iostream>
#include <filesystem>

const unsigned int seed = 42;

int main()
{
    std::string dataset_path = "datasets/mnist_test.csv";
    std::string model_save_path = "models/best_model.dat";
    std::filesystem::create_directories("models");

    // Load and split dataset into train, validation, and test sets
    Database db(dataset_path, seed);
    db.split_data(0.8);

    // -=- Network Architecture -=- //
    // Input: 784 (28x28 pixels)
    // Hidden Layers: 128, 64
    // Output: 10 (Digits 0-9)

    int input_size = 784;
    std::vector<int> hidden_layers = { 128, 64 };
    int output_size = 10;

    // -=- Network Architecture -=- //

    // Init Neural Network
    NeuralNetwork nn(input_size, hidden_layers, output_size, seed);

    // Training Hyperparameters
    int epochs = 50;
    int batch_size = 32;
    double initial_learning_rate = 0.001;
    double decay_rate = 0.9;
    int decay_steps = 10;
    bool early_stopping = true;
    int patience = 5;

    // Train the neural network
    nn.train(db.get_train_data(),
             db.get_validation_data(),
             epochs,
             batch_size,
             initial_learning_rate,
             decay_rate,
             decay_steps,
             early_stopping,
             patience);

    // Save the trained model
    nn.save_model(model_save_path);
    std::cout << "Model saved to " << model_save_path << std::endl;

    // Evaluate on test data after training
    Evaluator evaluator(nn, db.get_test_data());
    evaluator.evaluate();

    return 0;
}