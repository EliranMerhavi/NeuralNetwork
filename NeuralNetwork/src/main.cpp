#include <iostream>
#include "NeuralNetwork.h"

//TODO : add softmax and batch learning


void print_vec(std::vector<float> v)
{
	std::cout << '[';
	for (float& f : v)
	{
		std::cout << f << ", ";
	}

	std::cout << "\b\b]";
}

int main()
{
	NeuralNetwork network(3);

	network.addLayer(2, ActivationFunctionType::TANH);
	

	std::vector<std::vector<float>> inputs = { 
		{ 1.0f, 1.0f, 1.0f },
		{ 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f },
		
	};

	std::vector<std::vector<float>> answers = { 
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f },
		{ 0.0f, 0.0f }
	};

	for (int i = 0; i < 1000; i++)
	{
		for (int i = 0; i < inputs.size(); i++)
		{
			network.train(inputs[i], answers[i]);
		}
	}

	for (int i = 0; i < inputs.size(); i++)
	{
		std::vector<float> output = network.predict(inputs[i]);
		print_vec(output);
		std::cout << " : ";
		print_vec(answers[i]);
		std::cout << '\n';
	}


	

	system("pause");
}

