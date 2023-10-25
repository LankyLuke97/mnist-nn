// MnistNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <chrono>
#include <ctime>
#include <Eigen/Dense>
#include <iostream>
#include "DataReader.h"
#include "Helper.h"
#include "Network.h"

#include "ConvLayer.h"

int main() {
    srand(time(0));
    std::string trainingImagesFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\train-images.idx3-ubyte";
    std::string traininglabelsFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\train-labels.idx1-ubyte";
    std::string testImagesFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\t10k-images.idx3-ubyte";
    std::string testlabelsFile = "C:\\Users\\lhowd\\Documents\\Studio\\NeuralNetworks\\MnistNN\\t10k-labels.idx1-ubyte";
    Eigen::MatrixXd trainingImages = DataReader::readImageFile(trainingImagesFile).transpose();
    Eigen::MatrixXd trainingLabels = Helper::oneHotEncode(DataReader::readLableFile(traininglabelsFile), 10).transpose();
    Eigen::MatrixXd testImages = DataReader::readImageFile(testImagesFile).transpose();
    Eigen::MatrixXd testLabels = Helper::oneHotEncode(DataReader::readLableFile(testlabelsFile), 10).transpose();

    std::cout << trainingImages.rows() << ", " << trainingImages.cols() << std::endl;
    std::cout << trainingLabels.rows() << ", " << trainingLabels.cols() << std::endl;
    std::cout << testImages.rows() << ", " << testImages.cols() << std::endl;
    std::cout << testLabels.rows() << ", " << testLabels.cols() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd reshapedTrainingImages = Helper::convolveInput(trainingImages, 1, 5, 5, 28, 28);

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Convolved Matrix:\n" << reshapedTrainingImages.rows() << "x" << reshapedTrainingImages.cols() << "\nTook " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) << std::endl;

    start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd reshapedTestImages = Helper::convolveInput(testImages, 1, 5, 5, 28, 28);

    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Convolved Matrix:\n" << reshapedTestImages.rows() << "x" << reshapedTestImages.cols() << "\nTook " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) << std::endl;

    ConvLayer layer = ConvLayer(3, 1, 2, 2);

    /*Eigen::MatrixXd test(3, 16);
    test << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216;
    Eigen::MatrixXd processedTest = Helper::convolveInput(test, 1, 3, 3, 4, 4);

    std::cout << "Processed test: \n" << processedTest << std::endl;

    processedTest = Helper::convolveInput(test, 2, 2, 2, 4, 4);

    std::cout << "Processed test: \n" << processedTest << std::endl;

    /*double validationRatio = 0.90f;
    int splitIndex = static_cast<int>(trainingImages.cols() * validationRatio);
    int epochs = 1000;
    int miniBatchSize = 10;
    float learningRate = 3.0f;
    int learningRateSchedule = 6;

    std::vector<double> convolutionalNetwork_trainingCost;
    std::vector<double> convolutionalNetwork_trainingAccuracy;
    std::vector<double> convolutionalNetwork_validationCost;
    std::vector<double> convolutionalNetwork_validationAccuracy;

    Network convolutionalNetwork = Network({ 784, 100, 10 }, 10, 2.0, 1, 0);
    convolutionalNetwork.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), epochs, miniBatchSize, learningRate, learningRateSchedule, convolutionalNetwork_trainingCost, convolutionalNetwork_trainingAccuracy, convolutionalNetwork_validationCost, convolutionalNetwork_validationAccuracy);

    int testCorrect30_2 = convolutionalNetwork.evaluate(testImages, testLabels);
    std::cout << "On test data, 30 hidden neurons, lambda = 2: " << testCorrect30_2 << " / " << testImages.cols() << std::endl;*/

    /*std::vector<double> network30_2_trainingCost;
    std::vector<double> network30_2_trainingAccuracy;
    std::vector<double> network30_2_validationCost;
    std::vector<double> network30_2_validationAccuracy;

    Network network30_2 = Network({784, 30, 10}, 10, 2.0, 1, 0);
    network30_2.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), epochs, miniBatchSize, learningRate, learningRateSchedule, network30_2_trainingCost, network30_2_trainingAccuracy, network30_2_validationCost, network30_2_validationAccuracy);

    int testCorrect30_2 = network30_2.evaluate(testImages, testLabels);
    std::cout << "On test data, 30 hidden neurons, lambda = 2: " << testCorrect30_2 << " / " << testImages.cols() << std::endl;*/
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
