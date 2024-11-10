#include "Evaluator.h"
#include <iostream>
#include <iomanip>

Evaluator::Evaluator(const NeuralNetwork& nn, const std::vector<Database::DataPoint>& test_data)
    : nn(nn), test_data(test_data) {}

void Evaluator::evaluate() const {
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