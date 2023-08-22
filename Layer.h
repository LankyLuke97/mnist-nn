#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Neuron.h"
#include "Helper.h"

class Layer {
public:
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;

	Layer(int inputs, int numberOfNeurons) {
		biases = Eigen::VectorXd::Random(numberOfNeurons);
		weights = Eigen::MatrixXd::Random(numberOfNeurons, inputs);
	}

	Eigen::MatrixXd feedForward(Eigen::MatrixXd input) {
		assert(input.rows() == weights.cols());
		assert(biases.rows() == weights.rows());

		return (weights * input).colwise() + biases;
	}
};