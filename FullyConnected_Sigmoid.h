#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Helper.h"
#include "Layer.h"

class FullyConnected_Sigmoid : public Layer {
public:
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;
	Eigen::MatrixXd localInput;
	Eigen::MatrixXd output;
	Eigen::MatrixXd interimOutput;
	Eigen::MatrixXd nabla_b;
	Eigen::MatrixXd nabla_w	;

	FullyConnected_Sigmoid() {
	}

	FullyConnected_Sigmoid(int inputs, int numberOfNeurons) {
		biases = Eigen::VectorXd::Random(numberOfNeurons);
		weights = scaledWeightInitialisation(inputs, numberOfNeurons);
	}

	FullyConnected_Sigmoid(int inputs, int numberOfNeurons, int initialisationType) {
		biases = Eigen::VectorXd::Random(numberOfNeurons);

		if(initialisationType == 0) {
			weights = scaledWeightInitialisation(inputs, numberOfNeurons);
		} else if(initialisationType == 1) {
			weights = largeWeightInitialisation(inputs, numberOfNeurons);
		} else {
			assert(false);
		}
	}

	Eigen::MatrixXd backwardPass(const Eigen::MatrixXd& upstream) override {
		Eigen::MatrixXd deltaThroughSigmoid = upstream.cwiseProduct(interimOutput.unaryExpr<double(*)(double)>(&Helper::sigmoidPrime));
		nabla_b = deltaThroughSigmoid;
		nabla_w = deltaThroughSigmoid * localInput.transpose();

		return weights.transpose() * deltaThroughSigmoid;
	}

	Eigen::MatrixXd forwardPass(const Eigen::MatrixXd &input) override {
		assert(input.rows() == weights.cols());
		assert(biases.rows() == weights.rows());

		localInput = input;
		interimOutput = ((weights * input).colwise() + biases);
		output = interimOutput.unaryExpr<double(*)(double)>(&Helper::sigmoid);
		return output;
	}

	void update(double step) override {
		biases = biases - (step * nabla_b.rowwise().sum());
		weights = weights - (nabla_w * step);// If this were regularised, it would be (regFactor * weights) - ...
	}

	Eigen::VectorXd getBiases() override {
		return biases;
	}

	Eigen::MatrixXd getWeights() override {
		return weights;
	}

	std::unique_ptr<Layer> clone() const override {
		return std::make_unique<FullyConnected_Sigmoid>(*this);
	}

	Eigen::MatrixXd scaledWeightInitialisation(int inputs, int numberOfNeurons) {
		return Eigen::MatrixXd::Random(numberOfNeurons, inputs) / sqrt(inputs);
	}

	Eigen::MatrixXd largeWeightInitialisation(int inputs, int numberOfNeurons) {
		return Eigen::MatrixXd::Random(numberOfNeurons, inputs);
	}
};