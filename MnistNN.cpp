// MnistNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <ctime>
#include <Eigen/Dense>
#include <iostream>
#include "DataReader.h"
#include "Helper.h"
#include "Network.h"

int main() {
    srand(time(0));
    std::string trainingImagesFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\train-images.idx3-ubyte";
    std::string traininglabelsFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\train-labels.idx1-ubyte";
    std::string testImagesFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\t10k-images.idx3-ubyte";
    std::string testlabelsFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\t10k-labels.idx1-ubyte";
    Eigen::MatrixXd trainingImages = DataReader::readImageFile(trainingImagesFile);
    Eigen::MatrixXd trainingLabels = Helper::oneHotEncode(DataReader::readLableFile(traininglabelsFile), 10);
    Eigen::MatrixXd testImages = DataReader::readImageFile(testImagesFile);
    Eigen::MatrixXd testLabels = Helper::oneHotEncode(DataReader::readLableFile(testlabelsFile), 10);

    std::cout << trainingImages.rows() << ", " << trainingImages.cols() << std::endl;
    std::cout << trainingLabels.size() << std::endl;
    std::cout << testImages.rows() << ", " << testImages.cols() << std::endl;
    std::cout << testLabels.size() << std::endl;

    double validationRatio = 0.9f;
    int splitIndex = static_cast<int>(trainingImages.rows() * validationRatio);

    Network network = Network({784, 30, 10});
    network.stochasticGradientDescent(trainingImages.topRows(splitIndex), trainingLabels.topRows(splitIndex), trainingImages.bottomRows(trainingImages.rows() - splitIndex), trainingLabels.bottomRows(trainingLabels.rows() - splitIndex), 50, 1000, 3.0f);
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
