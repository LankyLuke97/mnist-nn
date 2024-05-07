#pragma once
#include <cmath>
#include <Eigen/Dense>

class Neuron {
public:
	double bias;
	Eigen::RowVectorXd weights;

	Neuron(int inputs) {
		bias = (double)rand() / RAND_MAX;
		weights = Eigen::RowVectorXd::Random(inputs);
	}

	double sigmoidActivation(const Eigen::RowVectorXd &x) {
		return 1.0f / (1.0f + exp((weights.array() * x.array()).sum() + bias));
	}
};