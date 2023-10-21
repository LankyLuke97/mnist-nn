#pragma once
#include "Layer.h"

class PoolingLayer : Layer {
	int numberFeatures, stride, windowHeight, windowWidth;	

	std::vector<Eigen::MatrixXd> feedForward(std::vector<Eigen::MatrixXd> _input) {
		std::vector<Eigen::MatrixXd> pooled = std::vector<Eigen::MatrixXd>(_input.size());
		
		for(Eigen::MatrixXd input : _input) {
			int outputRows = ((input.rows() - windowHeight) / stride) + 1;
			int outputCols = ((input.cols() - windowWidth) / stride) + 1;

			Eigen::MatrixXd result(windowHeight * windowWidth, outputRows * outputCols);

			for(int i = 0; i < outputRows; ++i) {
				for(int j = 0; j < outputCols; ++j) {
					int startRow = i * stride;
					int startCol = j * stride;

					int colOffset = i * outputCols + j;

					Eigen::Map<const Eigen::MatrixXd> window(
						input.block(startRow, startCol, windowHeight, windowWidth).data(),
						windowHeight * windowWidth, 1
					);

					result.col(colOffset) = window;
				}
			}

			pooled.push_back(result);
		}

		return pooled;
	}
};