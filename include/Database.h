#ifndef DATABASE_H
#define DATABASE_H

#include <vector>
#include <string>
#include <random>

class Database {
public:
    struct DataPoint {
        int label;                      // Digit label (0-9)
        std::vector<double> features;   // Normalised pixel values (0 to 1)
    };

    // Constructor that loads data from a CSV file with a given seed
    Database(const std::string& filepath, unsigned int seed);

    // Splits data into training, validation, and test sets
    void split_data(double train_ratio, double validation_ratio = 0.1);

    // Accessors
    const std::vector<DataPoint>& get_train_data() const;
    const std::vector<DataPoint>& get_validation_data() const;
    const std::vector<DataPoint>& get_test_data() const;

private:
    std::vector<DataPoint> data;                // All data points
    std::vector<DataPoint> train_data;          // Training data
    std::vector<DataPoint> validation_data;     // Validation data
    std::vector<DataPoint> test_data;           // Test data
    std::mt19937 rng;                           // Random number generator

    // Loads data from a CSV file
    void load_data(const std::string& filepath);
};

#endif