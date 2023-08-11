#pragma once
#include <Eigen/Dense>
#include <vector>
#include "Helper.h"
#include "Layer.h"

class Network {
public:
	std::vector<Layer> layers;

	Network(std::vector<int> layerStructure) {
		for(int i = 1; i < layerStructure.size(); i++) layers.push_back(Layer(layerStructure[i - 1], layerStructure[i]));
	}

	Eigen::MatrixXf forwardPass(Eigen::MatrixXf input) {
		Eigen::MatrixXf activations;

		for(int i = 0; i < layers.size(); i++) {
			activations = layers[i].feedForward(input);
		}

		return activations;
	}

	Eigen::MatrixXf calculateCost(Eigen::MatrixXf oneHotEncodedLabels, Eigen::MatrixXf predicitons) {

	}
};