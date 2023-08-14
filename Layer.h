#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Neuron.h"
#include "Helper.h"

class Layer {
public:
	Eigen::RowVectorXd biases;
	Eigen::MatrixXd weights;

	Layer(int inputs, int numberOfNeurons) {
		biases = Eigen::RowVectorXd::Random(numberOfNeurons);
		weights = Eigen::MatrixXd::Random(inputs, numberOfNeurons);
	}

	Eigen::MatrixXd feedForward(Eigen::MatrixXd input) {
		assert(input.cols() == weights.rows());
		assert(biases.cols() == weights.cols());

		return (input * weights).rowwise() + biases; // Parameterise activation function
	}
};