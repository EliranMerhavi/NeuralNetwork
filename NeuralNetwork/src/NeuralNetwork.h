#pragma once
#include <vector>

#include "NeuralNetworkUtils.h"

struct Layer 
{
	ActivationFunctionType functionType;
	std::vector<std::vector<float>> weights;
	std::vector<float> biases;
	int input_length, output_length;

	Layer(int input_length, int output_length, ActivationFunctionType type);
	
	/*activates the layer with the input and return the output */
	std::vector<float> activate(std::vector<float> input);
	
	/*update the weights and biases based on the parameters and returns*/
	float backprop(std::vector<float> output_neurons, std::vector<float> input_neurons, std::vector<float> error, float learning_rate);
	float backprop(std::vector<float> output_neurons, std::vector<float> input_neurons, float error, float learning_rate);

	~Layer();
};

class NeuralNetwork
{
private:
	int input_layer_length;
	std::vector<Layer> layers;
	const float learning_rate = 0.1f;
public:
	NeuralNetwork(int input_layer_length);
	~NeuralNetwork();

	void addLayer(int length, ActivationFunctionType type);
	
	/*returns the cost*/
	float train(std::vector<float> input_layer, std::vector<float> answer);
	
	/*returns the predictions*/
	std::vector<float> predict(std::vector<float> input_layer);

private:

};