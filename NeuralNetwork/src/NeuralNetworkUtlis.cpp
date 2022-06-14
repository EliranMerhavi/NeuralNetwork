#include "NeuralNetworkUtils.h"
#include <algorithm>

float dot(std::vector<float> v1, std::vector<float> v2)
{
	float product = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		product += v1[i] * v2[i];
	}

	return product;
}

float relu(float x)
{
	return std::max(0.0f, x);
}

float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

float _tanh(float x)
{
	return tanh(x);
}

float d_relu(float x)
{
	return x < 0 ? 0.0f : 1.0f;
}

float d_sigmoid(float x)
{
	float y = sigmoid(x);
	return y * (1 - y);
}

float d_tanh(float x)
{
	return -1.0f;
}

Function get_activation_function(ActivationFunctionType type)
{
	switch (type)
	{
		case ActivationFunctionType::SIGMOID: return sigmoid;
		case ActivationFunctionType::RELU:    return relu;
		case ActivationFunctionType::TANH:    return _tanh;
	}
}

Function get_derivative_function(ActivationFunctionType type)
{
	switch (type)
	{
		case ActivationFunctionType::SIGMOID: return d_sigmoid;
		case ActivationFunctionType::RELU:    return d_relu;
		case ActivationFunctionType::TANH:    return d_tanh;
	}
}