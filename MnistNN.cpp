// MnistNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "DataReader.h"

int main() {
    std::string trainingImagesFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\train-images.idx3-ubyte";
    std::string traininglabelsFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\train-labels.idx1-ubyte";
    std::string testImagesFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\t10k-images.idx3-ubyte";
    std::string testlabelsFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\t10k-labels.idx1-ubyte";
    std::vector<std::vector<uint8_t>> trainingImages = DataReader::readImageFile(trainingImagesFile);
    std::vector<uint8_t> trainingLabels = DataReader::readLableFile(traininglabelsFile);
    std::vector<std::vector<uint8_t>> testImages = DataReader::readImageFile(testImagesFile);
    std::vector<uint8_t> testLabels = DataReader::readLableFile(testlabelsFile);

    std::cout << trainingImages.size() << ", " << trainingImages[0].size() << std::endl;
    std::cout << trainingLabels.size() << std::endl;
    std::cout << testImages.size() << ", " << testImages[0].size() << std::endl;
    std::cout << testLabels.size() << std::endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
