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

	Eigen::MatrixXf backPropagation() {

	}

	float calculateCost(Eigen::MatrixXf oneHotEncodedLabels, Eigen::MatrixXf predictions) {
		return (oneHotEncodedLabels - predictions).array().square().sum() / (2 * predictions.rows());
	}

	Eigen::MatrixXf forwardPass(Eigen::MatrixXf activations) {

		for(int i = 0; i < layers.size(); i++) {
			activations = layers[i].feedForward(activations);
		}

		return activations;
	}

	void stochasticGradientDescent(Eigen::MatrixXf trainingData, Eigen::MatrixXf validationData, int epochs, int miniBatchSize, float eta) {
		int numberOfTrainingInputs = trainingData.rows();
		for(int j = 0; j < epochs; j++) {
			std::vector<int> shuffledIndeces(numberOfTrainingInputs);
			for(int i = 0; i < numberOfTrainingInputs; i++) {
				shuffledIndeces[i] = i;
			}

#if __cplusplus >= 201703L // Check if C++17 or higher
			// Use std::shuffle for C++17 and higher
			std::random_device rd;
			std::default_random_engine rng(rd());
			std::shuffle(shuffledIndeces.begin(), shuffledIndeces.end(), rng);
#else
			// Use std::random_shuffle for pre-C++17
			std::random_shuffle(shuffledIndeces.begin(), shuffledIndeces.end());
#endif	
			for(int k = 0; k < numberOfTrainingInputs; k += miniBatchSize) {
				Eigen::MatrixXf miniBatch = trainingData(Eigen::placeholders::all, std::vector<int>(shuffledIndeces.begin() + k, shuffledIndeces.begin() + k + miniBatchSize));
				updateMiniBatch(miniBatch, eta);
			}

			/*if(validationData) {

			}*/
		}
	}

	void updateMiniBatch(Eigen::MatrixXf miniBatch, float eta) {
		
	}
};