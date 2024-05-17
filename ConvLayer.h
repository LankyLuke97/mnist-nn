#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Layer.h"
#include "Neuron.h"
#include "Helper.h"

class ConvLayer : Layer {
public:
	int numberFeatures, stride, windowHeight, windowWidth;
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;

	ConvLayer(int numberFeatures, int stride, int windowHeight, int windowWidth) : numberFeatures(numberFeatures), stride(stride), windowHeight(windowHeight), windowWidth(windowWidth) {
		biases = Eigen::VectorXd::Random(numberFeatures);
		weights = Eigen::MatrixXd::Random(numberFeatures, windowHeight * windowWidth) / sqrt(windowHeight * windowWidth);
	}

	std::vector<Eigen::MatrixXd> feedForward(const std::vector<Eigen::MatrixXd> &_input) {
		/*
		* Input is a w * (j * k * m) matrix, where:
		* w is the flattened window size (windowHeight * windowWidth)
		* j is the number of windows across each input
		* k is the number of windows down each input
		* m is the number of inputs
		* Each column is a slice of the original input corresponding to the window as it passes over
		* the input. These are stacked in columns next to each other, and then each input
		* is stacked columnwise next to each other. The output is an n * (j * k * m) matrix
		* where n is the number of features. 
		*/

		std::vector<Eigen::MatrixXd> mapped;
		mapped.reserve(numberFeatures * _input.size());

		for(Eigen::MatrixXd input : _input) {
			//std::cout << "New input for feature mapping" << std::endl;
			int dim = std::sqrt(input.cols());
			int numInputs = input.rows();
			//std::cout << "Convolving" << std::endl;
			Eigen::MatrixXd convolved = Helper::convolveInput(input, stride, windowHeight, windowWidth, dim, dim); // dim, dim only works for squares
			//std::cout << "Convolving finished" << std::endl;

			Eigen::MatrixXd featureMapped = (weights * convolved).colwise() + biases;

			for(int i = 0; i < featureMapped.rows(); i++) {
				//std::cout << "Feature " << i << std::endl;
				mapped.push_back(featureMapped.row(i).reshaped<Eigen::RowMajor>(numInputs, featureMapped.cols() / numInputs));
			}
		}

		return mapped;
	}
};