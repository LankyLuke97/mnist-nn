#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Neuron.h"
#include "Helper.h"

class Layer {
public:
	Eigen::RowVectorXf biases;
	Eigen::MatrixXf weights;

	Layer(int inputs, int numberOfNeurons) {
		biases = Eigen::RowVectorXf::Random(numberOfNeurons);
		weights = Eigen::MatrixXf::Random(inputs, numberOfNeurons);
	}

	Eigen::MatrixXf feedForward(Eigen::MatrixXf input) {
		assert(input.cols() == weights.rows());
		assert(biases.cols() == weights.cols());

		return ((input * weights).rowwise() + biases).unaryExpr<float(*)(float)>(&Helper::sigmoid); // Parameterise activation function
	}
};