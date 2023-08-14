#pragma once
#include <cmath>
#include <Eigen/Dense>

class Helper {
public:
	static double sigmoid(double input) {
		return 1.0f / (1.0f + exp(input));
	}

	static double sigmoidPrime(double input) {
		double sig = sigmoid(input);
		return sig * (1 - sig);
	}

	static Eigen::RowVectorXd softmax(Eigen::VectorXd input) {
		Eigen::RowVectorXd exponential = input.array().exp();
		return exponential / exponential.sum();
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
};