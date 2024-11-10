
![AlphaDigit Project Logo](https://github.com/jackkimmins/Alpha-Digit/blob/main/web/favicons/apple-icon-180x180.png)

# Alpha-Digit

This project implements a fully connected [Feedforward Neural Network (FNN)](https://en.wikipedia.org/wiki/Feedforward_neural_network) from scratch in C++ for digit classification on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) - this is a continuation from my [previous attempt](https://github.com/jackkimmins/SimpleNN) at this. The network is optimised for web deployment through [WebAssembly (WASM)](https://webassembly.org/), allowing it to perform inference in the browser.

main_train.cpp:
```cpp
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
```

## Project Details
- Feedforward Neural Network (FNN)
- Mini-Batch Gradient Descent w/ Adam Optimiser
- Early Stopping and Learning Rate Scheduling
- Multithreaded Training
- WebAssembly Module for Inferencing
- Vue3 UI
- High Classification Accuracy of **98.19%** *(on 7,000 image validation slice of the MNIST dataset)*


## Requirements
- C++20 or Later
- g++
- Emscripten (for WASM compilation)

## Demo
The demo site can be found at the following link:
[https://alpha-digit.appserver.uk/](https://alpha-digit.appserver.uk/)

## Mentions and Thanks
Special thanks to the following resources and individuals whose work greatly inspired and supported this project:

- [3Blue1Brown](https://www.youtube.com/@3blue1brown) - For the amazing video demonstrations and intuitive explanations of neural networks, can't recommend his videos enough! 😇
- Research Paper by Diederik P. Kingma and Jimmy Ba: Adam: A Method for Stochastic Optimization ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)).

