#pragma once
#include <cmath>
#include <Eigen/Dense>

class Cost {
public:
	Cost(int type) {
		this->type = type;
	}

	double cost(const Eigen::MatrixXd &activations, const Eigen::MatrixXd &labels) {
		if(type == 0) {
			return crossEntropy(activations, labels);
		} else if(type == 1) {
			return quadratic(activations, labels);
		} else if(type == 2) {
			return logLikelihood(activations, labels);
		}
	}

	Eigen::MatrixXd delta(const Eigen::MatrixXd &zs, const Eigen::MatrixXd &activations, const Eigen::MatrixXd &labels) {
		if(type == 0) {
			return crossEntropyDelta(zs, activations, labels);
		} else if(type == 1) {
			return quadraticDelta(zs, activations, labels);
		}
	}
private:
	int type;

	double crossEntropy(const Eigen::MatrixXd &activations, const Eigen::MatrixXd &labels) {
		Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(labels.rows(), labels.cols());
		Eigen::MatrixXd neg_ones = -ones;

		Eigen::MatrixXd term1 = -labels.array() * (activations.array().log());
		Eigen::MatrixXd term2 = (neg_ones - labels).array() * ((ones - activations).array().log());

		Eigen::MatrixXd result = term1 + term2;
		result = result.array().isNaN().select(0, result); // Replace NaN with 0

		return result.sum() / activations.cols();
	}

	Eigen::MatrixXd crossEntropyDelta(const Eigen::MatrixXd &zs, const Eigen::MatrixXd &activations, const Eigen::MatrixXd &labels) {
		return (activations - labels);
	}

	double quadratic(const Eigen::MatrixXd &activations, const Eigen::MatrixXd &labels) {
		return 0.5 * (activations - labels).squaredNorm() / activations.cols();
	}

	Eigen::MatrixXd quadraticDelta(const Eigen::MatrixXd &zs, const Eigen::MatrixXd &activations, const Eigen::MatrixXd &labels) {
		return (activations - labels).cwiseProduct(zs.unaryExpr<double(*)(double)>(&Helper::sigmoidPrime));
	}
};