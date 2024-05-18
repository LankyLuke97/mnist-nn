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

	static double relu(double input) {
		if(input <= 0.0) return 0.0;
		else return input;
	}

	static double reluPrime(double input) {
		if(input <= 0.0) return 0.0;
		else return 1.0;
	}

	static double leakyRelu(double input) {
		if(input <= 0.0) return 0.01 * input;
		else return input;
	}

	static double leakyReluPrime(double input) {
		if(input <= 0.0) return 0.01;
		else return 1.0;
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

	static Eigen::MatrixXd convolveInput(const Eigen::MatrixXd & input, int stride, int windowHeight, int windowWidth, int origInputHeight, int origInputWidth) {
		int extra = stride > 1 ? 1 : 0;
		int outputRows = ((origInputHeight - windowHeight) / stride) + 1;
		int outputCols = ((origInputWidth - windowWidth) / stride) + 1;

		//std::cout << "outputRows: " << outputRows << ", outputCols: " << outputCols << std::endl;

		Eigen::MatrixXd result(windowHeight * windowWidth, outputRows * outputCols * input.rows());
		int offset = 0;

		for(int rowCount = 0; rowCount < input.rows(); rowCount++) {
			Eigen::RowVectorXd row = input.row(rowCount);
			//std::cout << "Row size: " << row.cols() << std::endl;
			for(int _y = 0; _y + windowHeight <= origInputHeight; _y += stride) {
				for(int _x = 0; _x + windowWidth <= origInputWidth; _x += stride) {
					Eigen::VectorXd window(windowWidth * windowHeight);
					int windowIdx = 0;

					for(int j = 0; j < windowHeight; j++) {
						for(int i = 0; i < windowWidth; i++) {
							//std::cout << "_y: " << _y << ", _x: " << _x << ", j: " << j << ", i: " << i << ", index: " << row((_y * origInputWidth) + _x + (j * origInputWidth) + i) << std::endl;
							window(windowIdx++) = row((_y * origInputWidth) + _x + (j * origInputWidth) + i);
						}
					}

					result.col(offset++) = window;
					//std::cout << "Shifting window right" << std::endl;
				}
				//std::cout << "Shifting window down" << std::endl;
			}
		}

		return result;
	}

	static void displayCharacter(const Eigen::VectorXd& character, int charSize) {
		const char intensityChars[] = { ' ', '.', ':', '-', '=', '+', '*', '#', '%', '@' };
		int i = 0;

		for(const auto& val : character) {
			std::cout << intensityChars[int(val / 0.1)];
			if(++i % charSize == 0) std::cout << std::endl;
		}
	}
};