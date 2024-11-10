#include "Database.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// Load data from the given file path
Database::Database(const std::string& filepath, unsigned int seed) : rng(seed) { load_data(filepath); }

// Split the data into training, validation, and testing sets
void Database::split_data(double train_ratio, double validation_ratio) {
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    size_t validation_size = static_cast<size_t>(data.size() * validation_ratio);
    train_data.assign(data.begin(), data.begin() + train_size);
    validation_data.assign(data.begin() + train_size, data.begin() + train_size + validation_size);
    test_data.assign(data.begin() + train_size + validation_size, data.end());

    std::cout << "Split data into " << train_data.size() << " training, "
              << validation_data.size() << " validation, and "
              << test_data.size() << " testing data points.\n";
}

const std::vector<Database::DataPoint>& Database::get_train_data() const { return train_data; }
const std::vector<Database::DataPoint>& Database::get_validation_data() const { return validation_data; }
const std::vector<Database::DataPoint>& Database::get_test_data() const { return test_data; }

void Database::load_data(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open the dataset file: " << filepath << std::endl;
        std::exit(1);
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Dataset file is empty." << std::endl;
        std::exit(1);
    }

    // Need to change this to be more flexible, eventually I want to be able to load any dataset
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
                    if (pixel < 0.0 || pixel > 255.0) throw std::out_of_range("Pixel value out of range");
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

        if (data_point.label != -1 && data_point.features.size() == 784) {
            data.push_back(data_point);
        } else {
            std::cerr << "Warning: Invalid data point at line " << line_number << ". Skipping.\n";
        }
    }

    file.close();

    if (data.empty()) {
        std::cerr << "No valid data loaded from the dataset." << std::endl;
        std::exit(1);
    }

    std::shuffle(data.begin(), data.end(), rng);
    std::cout << "Loaded " << data.size() << " data points from " << filepath << "\n";
}