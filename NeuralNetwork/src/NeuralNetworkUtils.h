#pragma once
#include <vector>


float dot(std::vector<float> v1,std::vector<float> v2);


typedef float (*Function)(float);

enum class ActivationFunctionType
{
	SIGMOID, RELU, TANH
};

Function get_activation_function(ActivationFunctionType type);
Function get_derivative_function(ActivationFunctionType type);