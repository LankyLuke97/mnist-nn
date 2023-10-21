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

	ConvLayer(int numberFeatures, int stride, int windowHeight, int windowWidth) : numberFeatures(numberFeatures), stride(stride), windowHeight(windowHeight), windowWidth(windowWidth) {
		biases = Eigen::VectorXd::Random(numberFeatures);
		weights = Eigen::MatrixXd::Random(numberFeatures, windowHeight * windowWidth) / sqrt(windowHeight * windowWidth);
	}

	Eigen::MatrixXd feedForward(Eigen::MatrixXd input) {
		/*
		* Input is a w * (j * k * m) matrix, where:
		* w is the flattened window size (windowHeight * windowWidth)
		* j is the number of windows across each input
		* k is the number of windows down each input
		* m is the number of inputs
		* The inputs are already convoluted by the time they're passed here. Each column
		* is a slice of the original input corresponding to the window as it passes over
		* the input. These are stacked in columns next to each other, and then each input
		* is stacked columnwise next to each other. The output is an n * (j * k * m) matrix
		* where n is the number of features. 
		*/

		assert(input.rows() == weights.cols());
		return (weights * input).colwise() + biases;
	}
};