#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "NeuralNetwork.h"

class Evaluator {
public:
    // Constructor that takes a reference to the neural network and test data
    Evaluator(const NeuralNetwork& nn, const std::vector<Database::DataPoint>& test_data);

    // Evaluates the network and prints accuracy
    void evaluate() const;

private:
    const NeuralNetwork& nn;
    const std::vector<Database::DataPoint>& test_data;
};

#endif