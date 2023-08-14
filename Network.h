#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Helper.h"
#include "Layer.h"

class Network {
public:
	std::vector<Layer> layers;

	Network(std::vector<int> layerStructure) {
		for(int i = 1; i < layerStructure.size(); i++) layers.push_back(Layer(layerStructure[i - 1], layerStructure[i]));
	}

	std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> backPropagation(Eigen::MatrixXd trainingData, Eigen::MatrixXd oneHotEncodedLabels) {
		/*
		* trainingData is m*n, where m is the number of samples in the minibatch and n is the number of features
		* oneHotEncodedLabels is m*10
		*/
		std::vector<Eigen::MatrixXd> nabla_b(layers.size(), Eigen::MatrixXd::Zero(1, 1));
		std::vector<Eigen::MatrixXd> nabla_w(layers.size(), Eigen::MatrixXd::Zero(1, 1));

		/* for(int i = 0; i < layers.size(); i++) {
			nabla_b.push_back(Eigen::RowVectorXd::Zero(layers[i].biases.cols()));
			nabla_w.push_back(Eigen::MatrixXd::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
		} */

		/*
		* layers.biases = [1 * 30, 1 * 10]
		* layers.weights = [784 * 30, 30 * 10]
		*/

		Eigen::MatrixXd activation = trainingData;
		std::vector<Eigen::MatrixXd> activations = { trainingData };
		std::vector<Eigen::MatrixXd> zs;

		for(int i = 0; i < layers.size(); i++) {
			Eigen::MatrixXd z = layers[i].feedForward(activation);
			zs.push_back(z);
			activation = z.unaryExpr<double(*)(double)>(&Helper::sigmoid);
			activations.push_back(activation);
		}
		/*
		* Take an example network here - 784 inputs, 30 hidden neurons, 10 output neurons
		* activations will have dimensions [m * 784, m * 30, m * 10]
		* zs will have dimensions [m * 30, m * 10]
		* Delta, then is also going to be a matrix - these need to be elementwise multiplications
		* Delta dimensions here will be m * 10
		* Important here - weights and biases both contain 2 items, as does zs, but num_layers is 3, as is the length 
		* of activations, as we start with the input data
		*/

		Eigen::MatrixXd delta = (activations.back() - oneHotEncodedLabels).cwiseProduct(zs.back().unaryExpr<double(*)(double)>(&Helper::sigmoidPrime));
		nabla_b.reserve(layers.size());
		nabla_w.reserve(layers.size());
		nabla_b[nabla_b.size() - 1] = delta;
		nabla_w[nabla_w.size() - 1] = activations[activations.size() - 2].transpose() * delta;

		/* 
		* At this point, nabla_b is [ _, m * 10], nabla_w is [ _, 30 * 10]
		*/

		for(int i = 2; i < activations.size(); i++) {
			/*
			* i = 2, sp and zs[0] = m * 30, delta is m * 10 initially, layers[1].weights.transpose is 10 * 30, delta ends as m * 30
			* activations[0].transpose = 784 * m
			*/
			Eigen::MatrixXd sp = zs[zs.size() - i].unaryExpr<double(*)(double)>(&Helper::sigmoidPrime);
			delta = (delta * layers[layers.size() - i + 1].weights.transpose()).cwiseProduct(sp);
			nabla_b[nabla_b.size() - i] = delta;
			nabla_w[nabla_w.size() - i] = activations[activations.size() - i - 1].transpose() * delta;
		}

		/*
		* At this point, nabla_b is [m * 30, m * 10], nabla_w is [784 * 30, 30 * 10] - check whether these should be reversed (i.e n * m)
		*/

		return std::make_pair(nabla_b, nabla_w);
	}

	double calculateCost(Eigen::MatrixXd oneHotEncodedLabels, Eigen::MatrixXd predictions) {
		return (oneHotEncodedLabels - predictions).array().square().sum() / (2 * predictions.rows());
	}

	int evaluate(Eigen::MatrixXd validationData, Eigen::MatrixXd validationLabels) {
		int correct = 0;
		Eigen::MatrixXd predictions = forwardPass(validationData);

		Eigen::VectorXi maxIndices(predictions.rows());
		for(Eigen::Index i = 0; i < predictions.rows(); ++i) {
			Eigen::Index labelIndex;
			Eigen::Index predictionIndex;
			validationLabels.row(i).maxCoeff(&labelIndex);
			predictions.row(i).maxCoeff(&predictionIndex);
			correct += 1 * (predictionIndex == labelIndex);
		}

		return correct;
	}

	Eigen::MatrixXd forwardPass(Eigen::MatrixXd activations) {
		/*
		* Activations passed in first as an m*n matrix, where m is the number of samples and n is the number of features
		*/

		for(int i = 0; i < layers.size(); i++) {
			//Need to apply softmax by row as well
			activations = layers[i].feedForward(activations).unaryExpr<double(*)(double)>(&Helper::sigmoid);
		}

		/*
		* At this point, activations should be an m*10 matrix, where m is the number of samples
		* and each column corresponds to the likelihood of that sample being a certain digit
		*/
		return activations;
	}

	void stochasticGradientDescent(Eigen::MatrixXd trainingData, Eigen::MatrixXd trainingLabels, Eigen::MatrixXd validationData, Eigen::MatrixXd validationLabels, int epochs, int miniBatchSize, double eta) {
		/*
		* trainingData is m*n, where m is number of samples and n is number of featurs
		* trainingLabels is m*10
		* validationData is k*n, where k is the number of samples in the validation set
		* validationLabels is k*10
		*/
		int numberOfTrainingInputs = trainingData.rows();
		for(int j = 0; j < epochs; j++) {
			std::vector<int> shuffledIndeces(numberOfTrainingInputs);
			for(int i = 0; i < numberOfTrainingInputs; i++) {
				shuffledIndeces[i] = i;
				if(i >= trainingData.rows()) std::cout << "uh oh" << std::endl;
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
				std::cout << "K: " << k << std::endl;
				/*
				* miniBatchData is miniBatchSize*n
				* miniBatchLabels is miniBatchSize*10
				*/

				int idx = k + miniBatchSize >= numberOfTrainingInputs ? numberOfTrainingInputs - k : miniBatchSize;

				Eigen::VectorXi miniBatchIndices(idx);
				for(int i = 0; i < idx; i++) {
					miniBatchIndices[i] = shuffledIndeces[k + i];
				}

				Eigen::MatrixXd miniBatchData = trainingData(miniBatchIndices, Eigen::all);
				Eigen::MatrixXd miniBatchLabels = trainingLabels(miniBatchIndices, Eigen::all);

				updateMiniBatch(miniBatchData, miniBatchLabels, eta);
			}

			int correct = evaluate(validationData, validationLabels);
			std::cout << "Epoch " << j << ": " << correct << "/" << validationData.rows() << std::endl;
		}
	}

	void updateMiniBatch(Eigen::MatrixXd trainingMiniBatch, Eigen::MatrixXd oneHotEncodedMiniBatchLabels, double eta) {
		/*
		* trainingMiniBatch is m*n, where m is the number of samples in the minibatch and n is the number of features
		* oneHotEncodedMiniBatchLabels0 is m*10
		*/
		std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> pair = backPropagation(trainingMiniBatch, oneHotEncodedMiniBatchLabels);
		std::vector<Eigen::MatrixXd> nabla_b = pair.first;
		std::vector<Eigen::MatrixXd> nabla_w = pair.second;

		/*
		* At this point, nabla_b is [m * 30, m * 10], nabla_w is [784 * 30, 30 * 10] - check whether these should be reversed (i.e n * m)
		*/

		for(int i = 0; i < layers.size(); i++) {
			layers[i].biases = layers[i].biases - ((eta / trainingMiniBatch.rows()) * nabla_b[i].colwise().sum());
			layers[i].weights = layers[i].weights - ((eta / trainingMiniBatch.rows()) * nabla_w[i]);
		}
	}
};