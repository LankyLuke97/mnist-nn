#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Neuron.h"
#include "Helper.h"

class ConvLayer {
public:
	int numberFeatures, stride, windowHeight, windowWidth;
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;

	ConvLayer(int numberFeatures, int stride, int windowHeight, int windowWidth) : numberFeatures(numberFeatures), stride(stride), windowHeight(windowHeight), windowWidth(windowWidth) {
		biases = Eigen::VectorXd::Random(numberFeatures);
		weights = Eigen::MatrixXd::Random(numberFeatures, windowHeight * windowWidth) / sqrt(windowHeight * windowWidth);
	}

	Eigen::MatrixXd preprocessInput(Eigen::MatrixXd input) {
		int outputRows = ((input.rows() - windowHeight) / stride) + 1;
		int outputCols = ((input.cols() - windowWidth) / stride) + 1;

		Eigen::MatrixXd result(windowHeight * windowWidth, outputRows * outputCols);

		for(int i = 0; i < outputRows; ++i) {
			for(int j = 0; j < outputCols; ++j) {
				int startRow = i * stride;
				int startCol = j * stride;

				int colOffset = i * outputCols + j;

				Eigen::Map<const Eigen::MatrixXd> window(
					input.block(startRow, startCol, windowHeight, windowWidth).data(),
					windowHeight * windowWidth, 1
				);

				result.col(colOffset) = window;
			}
		}

		return result;
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