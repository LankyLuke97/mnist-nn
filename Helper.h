#pragma once
#include <cmath>
#include <Eigen/Dense>

class Helper {
public:
	static double sigmoid(double input) {
		return 1.0f / (1.0f + exp(-input));
	}

	static double sigmoidPrime(double input) {
		double sig = sigmoid(input);
		return sig * (1 - sig);
	}

	static Eigen::RowVectorXd softmax(Eigen::VectorXd input) {
		Eigen::RowVectorXd exponential = input.array().exp();
		return exponential / exponential.sum();
	}

	static Eigen::MatrixXd applyRowWiseSoftmax(const Eigen::MatrixXd& input) {
		Eigen::MatrixXd expMatrix = input.array().exp();
		Eigen::VectorXd rowSums = expMatrix.rowwise().sum();
		return (expMatrix.array().colwise() / rowSums.array()).matrix();
	}

	static Eigen::MatrixXd oneHotEncode(Eigen::VectorXi input, int classes) {
		int numSamples = input.size();
		Eigen::MatrixXd oneHotEncoded = Eigen::MatrixXd::Zero(numSamples, classes);

		for(int i = 0; i < numSamples; ++i) {
			int label = input[i];
			oneHotEncoded(i, label) = 1.0;
		}

		return oneHotEncoded;
	}

	static Eigen::MatrixXd convolveInput(Eigen::MatrixXd input, int stride, int windowHeight, int windowWidth, int origInputHeight, int origInputWidth) {
		int outputRows = ((origInputHeight - windowHeight) / stride) + 1;
		int outputCols = ((origInputWidth - windowWidth) / stride) + 1;

		Eigen::MatrixXd result(windowHeight * windowWidth, outputRows * outputCols * input.rows());
		int offset = 0;

		for(int rowCount = 0; rowCount < input.rows(); rowCount++) {
			Eigen::RowVectorXd row = input.row(rowCount);
			std::cout << "Row size: " << row.cols() << std::endl;
			for(int _y = 0; _y < outputRows; _y += stride) {
				for(int _x = 0; _x < outputCols; _x += stride) {
					Eigen::VectorXd window(windowWidth * windowHeight);
					int windowIdx = 0;

					for(int j = 0; j < windowHeight; j += stride) {
						for(int i = 0; i < windowWidth; i += stride) {
							std::cout << "_y: " << _y << ", _x: " << _x << ", j: " << j << ", i: " << i << ", index: " << row((_y * origInputWidth) + _x + (j * origInputWidth) + i) << std::endl;
							window(windowIdx++) = row((_y * origInputWidth) + _x + (j * origInputWidth) + i);
						}
					}

					result.col(offset++) = window;
					std::cout << "Shifting window right" << std::endl;
				}
				std::cout << "Shifting window down" << std::endl;
			}
		}

		return result;
	}
};