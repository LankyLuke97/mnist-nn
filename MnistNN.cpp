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
    int epochs = 200;

    std::vector<double> network30_2_trainingCost;
    std::vector<double> network30_2_trainingAccuracy;
    std::vector<double> network30_2_validationCost;
    std::vector<double> network30_2_validationAccuracy;

    Network network30_2 = Network({784, 30, 10}, 10, 2.0, 1, 0);
    network30_2.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), epochs, 10, 3.0f, network30_2_trainingCost, network30_2_trainingAccuracy, network30_2_validationCost, network30_2_validationAccuracy);

    std::vector<double> network100_2_trainingCost;
    std::vector<double> network100_2_trainingAccuracy;
    std::vector<double> network100_2_validationCost;
    std::vector<double> network100_2_validationAccuracy;

    Network network100_2 = Network({ 784, 100, 10 }, 10, 2.0, 1, 0);
    network100_2.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), epochs, 10, 3.0f, network100_2_trainingCost, network100_2_trainingAccuracy, network100_2_validationCost, network100_2_validationAccuracy);

    std::vector<double> network30_5_trainingCost;
    std::vector<double> network30_5_trainingAccuracy;
    std::vector<double> network30_5_validationCost;
    std::vector<double> network30_5_validationAccuracy;

    Network network30_5 = Network({ 784, 30, 10 }, 10, 5.0, 1, 0);
    network30_5.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), epochs, 10, 3.0f, network30_5_trainingCost, network30_5_trainingAccuracy, network30_5_validationCost, network30_5_validationAccuracy);

    std::vector<double> network100_5_trainingCost;
    std::vector<double> network100_5_trainingAccuracy;
    std::vector<double> network100_5_validationCost;
    std::vector<double> network100_5_validationAccuracy;

    Network network100_5 = Network({ 784, 100, 10 }, 10, 5.0, 1, 0);
    network100_5.stochasticGradientDescent(trainingImages.leftCols(splitIndex), trainingLabels.leftCols(splitIndex), trainingImages.rightCols(trainingImages.cols() - splitIndex), trainingLabels.rightCols(trainingLabels.cols() - splitIndex), epochs, 10, 3.0f, network100_5_trainingCost, network100_5_trainingAccuracy, network100_5_validationCost, network100_5_validationAccuracy);
    
    int testCorrect30_2 = network30_2.evaluate(testImages, testLabels);
    int testCorrect100_2 = network100_2.evaluate(testImages, testLabels);
    int testCorrect30_5= network30_5.evaluate(testImages, testLabels);
    int testCorrect100_5= network100_5.evaluate(testImages, testLabels);
    std::cout << "On test data, 30 hidden neurons, lambda = 2: " << testCorrect30_2 << " / " << testImages.cols() << std::endl;
    std::cout << "On test data, 100 hidden neurons, lambda = 2: " << testCorrect100_2 << " / " << testImages.cols() << std::endl;
    std::cout << "On test data, 30 hidden neurons, lambda = 5: " << testCorrect30_5 << " / " << testImages.cols() << std::endl;
    std::cout << "On test data, 100 hidden neurons, lambda = 5: " << testCorrect100_5 << " / " << testImages.cols() << std::endl;


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
