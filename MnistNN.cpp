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
    Eigen::MatrixXd trainingLabels = Helper::oneHotEncode(DataReader::readLableFile(traininglabelsFile), 10).transpose();
    Eigen::MatrixXd testImages = DataReader::readImageFile(testImagesFile);
    Eigen::MatrixXd testLabels = Helper::oneHotEncode(DataReader::readLableFile(testlabelsFile), 10).transpose();

    std::cout << trainingImages.rows() << ", " << trainingImages.cols() << std::endl;
    std::cout << trainingLabels.rows() << ", " << trainingLabels.cols() << std::endl;
    std::cout << testImages.rows() << ", " << testImages.cols() << std::endl;
    std::cout << testLabels.rows() << ", " << testLabels.cols() << std::endl;

    double validationRatio = 0.90f;
    int splitIndex = static_cast<int>(trainingImages.cols() * validationRatio);

    Network network30 = Network({784, 30, 10});
    network30.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), 30, 10, 3.0f);
    Network network100 = Network({ 784, 100, 10 });
    network100.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), 30, 10, 3.0f);
    Network network30_large = Network({ 784, 30, 10 }, 1);
    network30_large.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), 30, 10, 3.0f);
    Network network100_large = Network({ 784, 100, 10 }, 1);
    network100_large.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), 30, 10, 3.0f);

    int testCorrect30 = network30.evaluate(testImages, testLabels);
    int testCorrect100 = network100.evaluate(testImages, testLabels);
    int testCorrect30_large = network30_large.evaluate(testImages, testLabels);
    int testCorrect100_large = network100_large.evaluate(testImages, testLabels);
    std::cout << "On test data, 30 hidden neurons: " << testCorrect30 << " / " << testImages.cols() << std::endl;
    std::cout << "On test data, 100 hidden neurons: " << testCorrect100 << " / " << testImages.cols() << std::endl;
    std::cout << "On test data, 30 hidden neurons, large weight initialisation: " << testCorrect30_large << " / " << testImages.cols() << std::endl;
    std::cout << "On test data, 100 hidden neurons, large weight initialisation: " << testCorrect100_large << " / " << testImages.cols() << std::endl;
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
