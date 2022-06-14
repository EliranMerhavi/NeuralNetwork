#include <iostream>
#include "NeuralNetwork.h"

void print_vec(std::vector<float> v)
{
	std::cout << '[';
	for (float& f : v)
	{
		std::cout << f << ", ";
	}

	std::cout << "\b\b]\n";
}

int main()
{
	NeuralNetwork network(3);

	network.addLayer(2, ActivationFunctionType::RELU);


	std::vector<std::vector<float>> inputs = { 
		{ 1.0f, 1.0f, 1.0f },
		{ 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f }
	};
	std::vector<std::vector<float>> answers = { 
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f },
		{ 0.0f, 0.0f }
	};

	float cost;
	for (int i = 0; i < 100; i++)
	{
		for (int i = 0; i < inputs.size(); i++)
		{
			cost = network.train(inputs[i], answers[i]);

			std::cout << "cost: " << cost << '\n';
		}

	}

	for (int i = 0; i < inputs.size(); i++)
	{
		std::vector<float> output = network.predict(inputs[i]);
		print_vec(output);
		std::cout << '\n';
	}


	

	system("pause");
}

