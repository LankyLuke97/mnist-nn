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
		weights = scaledWeightInitialisation(inputs, numberOfNeurons);
	}

	Layer(int inputs, int numberOfNeurons, int initialisationType) {
		biases = Eigen::VectorXd::Random(numberOfNeurons);

		if(initialisationType == 0) {
			weights = scaledWeightInitialisation(inputs, numberOfNeurons);
		} else if(initialisationType == 1) {
			weights = largeWeightInitialisation(inputs, numberOfNeurons);
		} else {
			assert(false);
		}
	}

	Eigen::MatrixXd feedForward(const Eigen::MatrixXd &input) {
		assert(input.rows() == weights.cols());
		assert(biases.rows() == weights.rows());

		return (weights * input).colwise() + biases;
	}

	Eigen::MatrixXd scaledWeightInitialisation(int inputs, int numberOfNeurons) {
		return Eigen::MatrixXd::Random(numberOfNeurons, inputs) / sqrt(numberOfNeurons);
	}

	Eigen::MatrixXd largeWeightInitialisation(int inputs, int numberOfNeurons) {
		return Eigen::MatrixXd::Random(numberOfNeurons, inputs);
	}
};