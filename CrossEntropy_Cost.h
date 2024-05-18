#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "Cost.h"
#include "Helper.h"
#include "Layer.h"

class CrossEntropy_Cost : public Cost {
public:
	CrossEntropy_Cost() {

	}

	Eigen::MatrixXd calculateDelta(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& labels) override {
		return (inputs - labels);
	}

	double calculateCost(const Eigen::MatrixXd& input, const Eigen::MatrixXd& labels) override {
		Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(labels.rows(), labels.cols());
		Eigen::MatrixXd neg_ones = -ones;

		Eigen::MatrixXd term1 = -labels.array() * (input.array().log());
		Eigen::MatrixXd term2 = (neg_ones - labels).array() * ((ones - input).array().log());

		Eigen::MatrixXd result = term1 + term2;
		result = result.array().isNaN().select(0, result); // Replace NaN with 0

		return result.sum() / input.cols();
	}
};