#pragma once
#include <cmath>
#include "Layer.h"
#include "Helper.h"

class PoolingLayer : Layer {
public:
	int numberFeatures, stride, windowHeight, windowWidth;	

	std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> _input) {
		std::vector<Eigen::MatrixXd> pooled;
		pooled.reserve(_input.size());
		
		for(Eigen::MatrixXd input : _input) {
			int dim = std::sqrt(input.cols());
			int numInputs = input.rows();
			Eigen::MatrixXd convolved = Helper::convolveInput(input, 2, 2, 2, dim, dim);

			//Currently just max pooling
			Eigen::MatrixXd maxConvolved = convolved.colwise().maxCoeff().reshaped<Eigen::RowMajor>(numInputs, convolved.cols() / numInputs);

			pooled.push_back(maxConvolved);
		}

		return pooled;
	}
};