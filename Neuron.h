#pragma once
#include <cmath>
#include <Eigen/Dense>

class Neuron {
public:
	float bias;
	Eigen::RowVectorXf weights;

	Neuron(int inputs) {
		bias = (float)rand() / RAND_MAX;
		weights = Eigen::RowVectorXf::Random(inputs);
	}

	float sigmoidActivation(Eigen::RowVectorXf x) {
		return 1.0f / (1.0f + exp((weights.array() * x.array()).sum() + bias));
	}
};