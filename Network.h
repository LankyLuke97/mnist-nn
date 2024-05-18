#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <numeric>
#include <vector>
#include "Cost.h"
#include "CrossEntropy_Cost.h"
#include "Helper.h"
#include "Layer.h"
#include "FullyConnected_Sigmoid.h"

class Network {
public:
	std::unique_ptr<Cost> cost;
	int stepsOnCurrentLearningRate = 0;
	int earlyStopThreshold = -1;
	double eta;
	int etaReductionCount = 0;
	double lambda = 0.0;
	int learningSchedule;
	std::vector<std::unique_ptr<Layer>> layers;

	double bestAccuracy = 0.0;
	std::vector<std::unique_ptr<Layer>> bestModel;

	Network(std::vector<int> layerStructure) {
		for(int i = 1; i < layerStructure.size(); i++) layers.push_back(std::make_unique<FullyConnected_Sigmoid>(layerStructure[i - 1], layerStructure[i]));
		cost = std::make_unique<CrossEntropy_Cost>();
	}

	Network(std::vector<int> layerStructure, int earlyStopThreshold, double lambda, int costType, int weightInitialisationType) {
		this->lambda = lambda;
		this->earlyStopThreshold = earlyStopThreshold;
		for(int i = 1; i < layerStructure.size(); i++) layers.push_back(std::make_unique<FullyConnected_Sigmoid>(layerStructure[i - 1], layerStructure[i], weightInitialisationType));
		cost = std::make_unique<CrossEntropy_Cost>();
	}

/*	std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> backPropagation(const Eigen::MatrixXd& trainingData, const Eigen::MatrixXd& oneHotEncodedLabels) {
		/*
		* trainingData is n*m, where m is the number of samples in the minibatch and n is the number of features
		* oneHotEncodedLabels is m*10
		* /
		std::vector<Eigen::MatrixXd> nabla_b(layers.size(), Eigen::MatrixXd::Zero(1, 1));
		std::vector<Eigen::MatrixXd> nabla_w(layers.size(), Eigen::MatrixXd::Zero(1, 1));

		/* for(int i = 0; i < layers.size(); i++) {
			nabla_b.push_back(Eigen::RowVectorXd::Zero(layers[i].biases.cols()));
			nabla_w.push_back(Eigen::MatrixXd::Zero(layers[i].weights.rows(), layers[i].weights.cols()));
		} * /

		/*
		* layers.biases = [30 * 1, 10 * 1]
		* layers.weights = [30 * 784, 10 * 30]
		* /

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
		* activations will have dimensions [784 * m, 30 * m, 10 * m]
		* zs will have dimensions [30 * m, 10 * m]
		* Delta, then is also going to be a matrix - these need to be elementwise multiplications
		* Delta dimensions here will be 10 * m
		* Important here - weights and biases both contain 2 items, as does zs, but num_layers is 3, as is the length 
		* of activations, as we start with the input data
		* /

		Eigen::MatrixXd delta = cost.delta(zs.back(), activations.back(), oneHotEncodedLabels);
		nabla_b.reserve(layers.size());
		nabla_w.reserve(layers.size());
		nabla_b[nabla_b.size() - 1] = delta;
		nabla_w[nabla_w.size() - 1] = delta * activations[activations.size() - 2].transpose();

		/* 
		* At this point, nabla_b is [ _, 10 * m, nabla_w is [ _, 10 * 30]
		* /

		for(int i = 2; i < activations.size(); i++) {
			/*
			* i = 2, sp and zs[0] = 30 * m, delta is 10 * m initially, layers[1].weights.transpose is 30 * 10, delta ends as 30 * m
			* activations[0] = 784 * m
			* /
			Eigen::MatrixXd sp = zs[zs.size() - i].unaryExpr<double(*)(double)>(&Helper::sigmoidPrime);
			delta = (layers[layers.size() - i + 1].weights.transpose() * delta).cwiseProduct(sp);
			nabla_b[nabla_b.size() - i] = delta;
			nabla_w[nabla_w.size() - i] = delta * activations[activations.size() - i - 1].transpose();
		}

		/*
		* At this point, nabla_b is [m * 30, m * 10], nabla_w is [784 * 30, 30 * 10] - check whether these should be reversed (i.e n * m)
		* /

		return std::make_pair(nabla_b, nabla_w);
	}
*/

	bool earlyStop(const std::vector<double> &accuracies) {
		if(accuracies.size() < earlyStopThreshold) return false;

#if __cplusplus >= 201703L // Check if C++17 or higher
		// Use std::shuffle for C++17 and higher
		double averageAccuracy = std::reduce(accuracies.end() - earlyStopThreshold, accuracies.end(), 0.0) / earlyStopThreshold;
		double sum_of_squared_diff = std::reduce(accuracies.end() - earlyStopThreshold, accuracies.end(), 0.0,
			[mean](double sum, double x) {
				double diff = x - mean;
				return sum + diff * diff;
			}
		);

		double standardDeviation = std::sqrt(sum_of_squared_diff / earlyStopThreshold);
#else
		// Use std::random_shuffle for pre-C++17
		double averageAccuracy = std::accumulate(accuracies.end() - earlyStopThreshold, accuracies.end(), 0.0) / earlyStopThreshold;
		double sum_of_squared_diff = std::accumulate(accuracies.end() - earlyStopThreshold, accuracies.end(), 0.0,
			[averageAccuracy](double sum, double x) {
				double diff = x - averageAccuracy;
				return sum + diff * diff;
			}
		);

		// Calculate the standard deviation
		double standardDeviation = std::sqrt(sum_of_squared_diff / earlyStopThreshold);
#endif
		if(accuracies.back() < averageAccuracy - standardDeviation || accuracies.back() < bestAccuracy - 3 * standardDeviation) {
			std::cout << "Current accuracy: " << accuracies.back() << "\nBest accuracy: " << bestAccuracy << "\nAverage and std dev over last " << earlyStopThreshold << " evaluations: " << averageAccuracy << ", " << standardDeviation << std::endl;
			eta /= 2;
			stepsOnCurrentLearningRate = 0;
			etaReductionCount++;

			if(etaReductionCount > learningSchedule) return true;
		}
		return false;
	}

	int evaluate(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels) {
		int correct = 0;
		Eigen::MatrixXd predictions = forwardPass(data);

		for(Eigen::Index i = 0; i < predictions.cols(); ++i) {
			Eigen::Index labelIndex;
			Eigen::Index predictionIndex;
			labels.col(i).maxCoeff(&labelIndex);
			predictions.col(i).maxCoeff(&predictionIndex);

			correct += 1 * (predictionIndex == labelIndex);
		}

		return correct;
	}

	Eigen::MatrixXd forwardPass(Eigen::MatrixXd activations) {
		/*
		* Activations passed in first as an m*n matrix, where m is the number of samples and n is the number of features
		*/

		for(int i = 0; i < layers.size(); i++) {
			activations = layers[i]->forwardPass(activations);
		}

		return activations;
	}

	void stochasticGradientDescent(const Eigen::MatrixXd &trainingData, const Eigen::MatrixXd &trainingLabels, const Eigen::MatrixXd &validationData, const Eigen::MatrixXd &validationLabels, int epochs, int miniBatchSize, double learningRate, int learningRateSchedule, std::vector<double>& trainingCost, std::vector<double>& trainingAccuracy, std::vector<double>& evaluationCost, std::vector<double>& evaluationAccuracy ) {
		/*
		* trainingData is n*m, where m is number of samples and n is number of featurs - in this case, 784 * ~50000
		* trainingLabels is 10*m
		* validationData is n*k, where k is the number of samples in the validation set - in this case 784 * ~10000
		* validationLabels is 10*k
		*/
		eta = learningRate;
		learningSchedule = learningRateSchedule;

		int numberOfTrainingInputs = trainingData.cols();
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
				/*
				* miniBatchData is miniBatchSize*n
				* miniBatchLabels is miniBatchSize*10
				*/

				int idx = k + miniBatchSize >= numberOfTrainingInputs ? numberOfTrainingInputs - k : miniBatchSize;

				Eigen::VectorXi miniBatchIndices(idx);
				for(int i = 0; i < idx; i++) {
					miniBatchIndices[i] = shuffledIndeces[k + i];
				}

				Eigen::MatrixXd miniBatchData = trainingData(Eigen::all, miniBatchIndices);
				Eigen::MatrixXd miniBatchLabels = trainingLabels(Eigen::all, miniBatchIndices);
				double step = eta / miniBatchData.cols();

				Eigen::MatrixXd input = miniBatchData;

				for(auto it = layers.begin(); it != layers.end(); it++) {
					input = (*it)->forwardPass(input);
				}

				Eigen::MatrixXd upstream = cost->calculateDelta(input, miniBatchLabels); // This needs to be integrated into the final layers - as cost will just be another layer eventually
																					  // The first 'input' here was zs.back() - which is the interimOutput of the current layers.
				
				for(auto it = layers.rbegin(); it != layers.rend(); it++) {
					upstream = (*it)->backwardPass(upstream);
				}

				for(auto it = layers.begin(); it != layers.end(); it++) {
					(*it)->update(step);
				}
			}

			int correct = evaluate(validationData, validationLabels);
			std::cout<< "Epoch " << j << ": " << correct << "/" << validationData.cols() << std::endl;

			trainingCost.push_back(totalCost(trainingData, trainingLabels, lambda));
			trainingAccuracy.push_back(evaluate(trainingData, trainingLabels));
			evaluationCost.push_back(totalCost(validationData, validationLabels, lambda));
			evaluationAccuracy.push_back(evaluate(validationData, validationLabels));

			if(evaluationAccuracy.back() > bestAccuracy) {
				bestAccuracy = evaluationAccuracy.back();
				bestModel.clear();
				bestModel.reserve(layers.size());
				for(const auto& layer : layers) {
					bestModel.push_back(layer->clone());
				}
			}
			//std::cout << "Cost on training data: " << trainingCost.back() << std::endl;
			//std::cout << "Accuracy on training data: " << trainingAccuracy.back() << "/" << trainingData.cols() << std::endl;
			//std::cout << "Cost on validation data: " << evaluationCost.back() << std::endl;
			//std::cout << "Accuracy on validation data: " << evaluationAccuracy.back() << "/" << validationData.cols() << std::endl;

			if(earlyStopThreshold > -1 && stepsOnCurrentLearningRate > earlyStopThreshold && earlyStop(evaluationAccuracy)) {
				std::cout << "Stopping early - accuracy on validation set stagnating. Resetting model weights and biases to best version." << std::endl;
				return;
			}

			stepsOnCurrentLearningRate++;
		}
	}

	double totalCost(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels, double lambda) {
		Eigen::MatrixXd activations = forwardPass(data);
		double totalCost = cost->calculateCost(activations, labels);
		for(int i = 0; i < layers.size(); i++) {
			totalCost += 0.5 * (lambda / data.cols()) * layers[i]->getWeights().squaredNorm();
		}

		return totalCost;
	}
/*
	void updateMiniBatch(const Eigen::MatrixXd &trainingMiniBatch, const Eigen::MatrixXd &oneHotEncodedMiniBatchLabels, double eta, double lambda, int n) {
		/*
		* trainingMiniBatch is n*m, where m is the number of samples in the minibatch and n is the number of features
		* oneHotEncodedMiniBatchLabels0 is m*10
		* /
		std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> pair = backPropagation(trainingMiniBatch, oneHotEncodedMiniBatchLabels);
		std::vector<Eigen::MatrixXd> nabla_b = pair.first;
		std::vector<Eigen::MatrixXd> nabla_w = pair.second;

		/*
		* At this point, nabla_b is [m * 30, m * 10], nabla_w is [784 * 30, 30 * 10] - check whether these should be reversed (i.e n * m)
		* /

		for(int i = 0; i < layers.size(); i++) {
			layers[i].biases = layers[i].biases - () * nabla_b[i].rowwise().sum());
			layers[i].weights = ((1 - eta * (lambda / n)) * layers[i].weights) - ((eta / trainingMiniBatch.cols()) * nabla_w[i]);
		}
	}
*/
};