#pragma once
#include <cmath>
#include <Eigen/Dense>

static class Helper {
public:
	static float sigmoid(float input) {
		return 1.0f / (1.0f + exp(input));
	}

	static Eigen::RowVectorXf softmax(Eigen::VectorXf input) {
		Eigen::RowVectorXf exponential = input.array().exp();
		return exponential / exponential.sum();
	}
};