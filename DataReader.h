#pragma once

#include <iostream>
#include <fstream>
#include <vector>

class DataReader {
public:
    static std::vector<std::vector<uint8_t>> readImageFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return std::vector<std::vector<uint8_t>>(); // Return an empty vector of vectors
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
            return std::vector<std::vector<uint8_t>>(); // Return an empty vector of vectors
        }

        // Step 3: Read the data into a vector of vectors
        std::vector<std::vector<uint8_t>> data(num_datapoints, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < num_datapoints; ++i) {
            file.read(reinterpret_cast<char*>(data[i].data()), rows * cols);
        }

        return data;
    }

    static std::vector<uint8_t> readLableFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return std::vector<uint8_t>(); // Return an empty vector
        }

        // Step 2: Parse the file headers
        int magic_number, num_datapoints;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&num_datapoints), sizeof(num_datapoints));

        magic_number = swapEndian(magic_number);
        num_datapoints = swapEndian(num_datapoints);

        std::cout << "Magic number: " << magic_number << std::endl;

        if (magic_number != 2049) {
            std::cerr << "Invalid magic number, not an IDX file." << std::endl;
            return std::vector<uint8_t>(); // Return an empty vector of vectors
        }

        // Step 3: Read the data into a vector of vectors
        std::vector<uint8_t> data(num_datapoints);
        file.read(reinterpret_cast<char*>(data.data()), num_datapoints);

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