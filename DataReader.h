#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>

class DataReader {
public:
    static Eigen::MatrixXd readImageFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return Eigen::MatrixXd(); // Return an empty matrix
        }

        // Step 2: Parse the file headers
        int magic_number, num_datapoints, rows, cols;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&num_datapoints), sizeof(num_datapoints));
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        magic_number = swapEndian(magic_number);
        num_datapoints = swapEndian(num_datapoints);
        rows = swapEndian(rows);
        cols = swapEndian(cols);

        std::cout << "Magic number: " << magic_number << std::endl;

        if (magic_number != 2051) {
            std::cerr << "Invalid magic number, not an IDX file." << std::endl;
            return Eigen::MatrixXd(); // Return an empty matrix
        }

        // Step 3: Read the data into a vector of vectors
        Eigen::MatrixXd data(num_datapoints, rows * cols);
        for(int i = 0; i < num_datapoints; i++) {
            for(int j = 0; j < rows * cols; j++) {
                uint8_t pixelValue;
                file.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue));
                data(i, j) = static_cast<double>(pixelValue) / 255.0; // Normalize pixel values to [0, 1]
            }
        }

        return data.transpose();
    }

    static Eigen::VectorXi readLableFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if(!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return Eigen::VectorXi(); // Return an empty vector
        }

        // Step 2: Parse the file headers
        uint32_t magic_number, num_datapoints;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&num_datapoints), sizeof(num_datapoints));

        magic_number = swapEndian(magic_number);
        num_datapoints = swapEndian(num_datapoints);

        std::cout << "Magic number: " << magic_number << std::endl;

        if(magic_number != 2049) {
            std::cerr << "Invalid magic number, not an IDX file." << std::endl;
            return Eigen::VectorXi(); // Return an empty vector
        }

        Eigen::VectorXi data(num_datapoints);
        for(int i = 0; i < num_datapoints; ++i) {
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));
            data(i) = static_cast<int>(label);
        }

        return data;
    }

private:
    static uint32_t swapEndian(uint32_t value) {
        return ((value & 0xFF) << 24) |
            (((value >> 8) & 0xFF) << 16) |
            (((value >> 16) & 0xFF) << 8) |
            ((value >> 24) & 0xFF);
    }   
};