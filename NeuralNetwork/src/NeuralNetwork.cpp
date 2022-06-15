#include "NeuralNetwork.h"
#include <iostream>


Layer::Layer(int input_length, int output_length, ActivationFunctionType type)
	: 
	functionType(type), 
	weights(output_length, std::vector<float>(input_length)), 
	biases(output_length),
	input_length(input_length), 
	output_length(output_length)
{
	for (int i = 0; i < output_length; i++) 
	{
		biases[i] = (float)rand() / RAND_MAX;
		for (int j = 0; j < input_length; j++)
		{
			weights[i][j] = (float)rand() / RAND_MAX;
		}
	}
}

std::vector<float> Layer::activate(std::vector<float> input)
{
	std::vector<float> output(output_length);
	Function f = get_activation_function(functionType);

	for (int i = 0; i < output_length; i++)
	{
		output[i] = f(dot(input, weights[i]) + biases[i]);
	}

	return output;
}

float Layer::backprop(std::vector<float> output_neurons, std::vector<float> input_neurons, std::vector<float> error, float learning_rate)
{
	float sum = 0.0f;
	Function df = get_derivative_function(functionType);

	for (int i = 0; i < output_length; i++)
	{

		biases[i] += learning_rate * df(output_neurons[i]) * error[i];

		for (int j = 0; j < input_length; j++)
		{
			sum += weights[i][j] * df(output_neurons[i]) * error[i];
			weights[i][j] += learning_rate * df(output_neurons[i]) * input_neurons[j] * error[i];
		}

	}
	
	
	return sum;
}

float Layer::backprop(std::vector<float> output_neurons, std::vector<float> input_neurons, float error, float learning_rate)
{
	float sum = 0.0f;
	Function df = get_derivative_function(functionType);

	for (int i = 0; i < output_length; i++)
	{

		biases[i] += learning_rate * df(output_neurons[i]) * error;
		
		for (int j = 0; j < input_length; j++)
		{
			sum += weights[i][j] * df(output_neurons[i]) * error;
			weights[i][j] += learning_rate * df(output_neurons[i]) * input_neurons[j] * error;
		}

	}

	return sum;
}

Layer::~Layer()
{
}

NeuralNetwork::NeuralNetwork(int input_layer_length)
	: input_layer_length(input_layer_length), layers()
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::addLayer(int length, ActivationFunctionType type)
{
	if (layers.size() == 0)
	{
		layers.emplace_back(
			input_layer_length, 
			length, 
			type
		);
	}
	else 
	{
		layers.emplace_back(
			layers.back().output_length, 
			length,
			type
		);
	}
}

float NeuralNetwork::train(std::vector<float> input_layer, std::vector<float> answer)
{
	if (layers.size() == 0)
	{
		return 0.0f;
	}

	std::vector<std::vector<float>> neurons(layers.size() + 1);
	neurons[0] = input_layer;
	
	
	for (int i = 1; i <= layers.size(); i++)
	{
		Layer& layer = layers[i - 1];
		
		neurons[i] = layer.activate(neurons[i-1]);
	}

	Function df;
	std::vector<float> to_calc_cost(layers.back().output_length);
	std::vector<float> error(layers.back().output_length);
	float sum;

	for (int i = 0; i < layers.back().output_length; i++)
	{
		error[i] = -(neurons.back()[i] - answer[i]);
		to_calc_cost[i] = 0.5f * (neurons.back()[i] - answer[i]) * (neurons.back()[i] - answer[i]);
	}

	sum = layers.back().backprop(
		neurons.back(),
		neurons[layers.size() - 1],
		error,
		learning_rate
	);

	for (int k = layers.size() - 2; k >= 0; k--)
	{
		sum = layers[k].backprop(neurons[k + 1], neurons[k], sum, learning_rate);
	}
	
	float cost = 0.0f;
	
	for (float& f : to_calc_cost) 
	{
		cost += f;
	}

	return cost;
}

std::vector<float> NeuralNetwork::predict(std::vector<float> input_layer)
{
	std::vector<float> input = input_layer;
	std::vector<float> output = input_layer;
	
	for (Layer& layer : layers)
	{		
		output = layer.activate(input);
		input = output;
	}

	return output;
}


