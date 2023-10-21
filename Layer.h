#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Helper.h"
#include "Neuron.h"

class Layer {
public:
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;

	Eigen::MatrixXd feedForward(Eigen::MatrixXd input) {
		assert(input.rows() == weights.cols());
		assert(biases.rows() == weights.rows());
	}
};