#pragma once
#include <cmath>
#include "Layer.h"
#include "Helper.h"

class PoolingLayer : Layer {
public:
	int stride, windowHeight, windowWidth;	

	PoolingLayer(int stride, int windowHeight, int windowWidth) : stride(stride), windowHeight(windowHeight), windowWidth(windowWidth) {
		
	}

	std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> _input) {
		std::vector<Eigen::MatrixXd> pooled;
		pooled.reserve(_input.size());
		
		int i = 0;

		for(Eigen::MatrixXd input : _input) {
			//std::cout << "New input for pooling: " << i++ << std::endl;
			int dim = std::sqrt(input.cols());
			int numInputs = input.rows();
			//std::cout << "Convolving" << std::endl;
			Eigen::MatrixXd convolved = Helper::convolveInput(input, stride, windowHeight, windowWidth, dim, dim);
			//std::cout << "Convolving finished" << std::endl;

			//Currently just max pooling
			Eigen::MatrixXd maxConvolved = convolved.colwise().maxCoeff().reshaped<Eigen::RowMajor>(numInputs, convolved.cols() / numInputs);

			//std::cout << "Push back pooled" << std::endl;

			pooled.push_back(maxConvolved);
		}

		return pooled;
	}
};