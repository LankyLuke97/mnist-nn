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
#include "FullyConnectedLayer.h"
#include "PoolingLayer.h"

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

    /*auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd reshapedTrainingImages = Helper::convolveInput(trainingImages, 1, 5, 5, 28, 28);

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Convolved Matrix:\n" << reshapedTrainingImages.rows() << "x" << reshapedTrainingImages.cols() << "\nTook " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) << std::endl;

    start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd reshapedTestImages = Helper::convolveInput(testImages, 1, 5, 5, 28, 28);

    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Convolved Matrix:\n" << reshapedTestImages.rows() << "x" << reshapedTestImages.cols() << "\nTook " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) << std::endl;
    */
    PoolingLayer pool = PoolingLayer(2, 2, 2);

    Eigen::MatrixXd test(3, 16);
    test << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216;
    Eigen::MatrixXd processedTest = Helper::convolveInput(test, 1, 3, 3, 4, 4);

    std::cout << "Processed test: \n" << processedTest << std::endl;

    processedTest = Helper::convolveInput(test, 2, 2, 2, 4, 4);

    std::cout << "Processed test: \n" << processedTest << std::endl;

    std::vector<Eigen::MatrixXd> processedPooled = pool.feedForward(std::vector<Eigen::MatrixXd> {test});


    std::cout << "Pooled test: \n" << processedPooled[0] << std::endl;

    Eigen::MatrixXd additionTest(2, 3);
    additionTest << 1, 2, 3,
        11, 12, 13;
    Eigen::VectorXd add(2);
    add << 5, 10;

    std::cout << "Added: \n" << (additionTest.colwise() + add) << std::endl;

    ConvLayer testConv = ConvLayer(20, 1, 5, 5);
    FullyConnectedLayer testConnected = FullyConnectedLayer(2880, 100);
    PoolingLayer testPool = PoolingLayer(2, 2, 2);

    std::vector<int> batchSizes = { 1, 10, 100 };
    std::vector<Eigen::MatrixXd> input;
    std::vector<Eigen::MatrixXd> mapped;
    std::vector<Eigen::MatrixXd> pooled;
    Eigen::MatrixXd pooledFlattened;
    Eigen::MatrixXd output;


    for(int batchSize : batchSizes) {
        auto start = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < trainingImages.rows(); i += batchSize) {
            input = { trainingImages(Eigen::seqN(i, batchSize), Eigen::all) };

            mapped = testConv.feedForward(input); // Vector: 1 element for each feature map, each contaning m rows by n columns, where m is the number of images and n is the number of features
            pooled = testPool.feedForward(mapped);// Vector: 1 element for each feature map, each contaning m rows by n columns, where m is the number of images and n is the number of features

            pooledFlattened.resize(batchSize, pooled.size() * pooled[0].cols());
            int colOffset = pooled[0].cols();

            for(int i = 0; i < pooled.size(); i++) {
                pooledFlattened.middleCols(colOffset * i, colOffset) = pooled[i];
            }

            if(i == 0)
                std::cout  << pooledFlattened.rows() << ", " << pooledFlattened.cols() << std::endl;

            output = testConnected.feedForward(pooledFlattened.transpose()); // ---- this transpose needs to go - it's because I had to reverse the matrices elsewhere for the maths to work

            if(i == 0)
                std::cout << output.rows() << ", " << output.cols() << std::endl;
        }

        auto stop = std::chrono::high_resolution_clock::now();

        std::cout << "Process with batch size " << batchSize << " took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()) << " milliseconds" << std::endl;
        std::cout << "Shape of pooled: " << pooled.size() << "x" << pooled[0].rows() << "x" << pooled[0].cols() << std::endl;
        std::cout << "Shape of pooledFlattened: " << pooledFlattened.rows() << "x" << pooledFlattened.cols() << std::endl;
        std::cout << "Shape of output: " << output.rows() << "x" << output.cols() << std::endl;
    }

    std::cout << "Is it quicker, since I know the size of the matrices at each level,\n\
                to precompute the indeces needed to slice the incoming matrix into its\n\
                convolved version, rather than looping each time? I suspect yes. Explore\n\
                this next." << std::endl;

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
