#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>
#include <vector>

extern "C" {
	DLLEXPORT double* linear_model_create();
	DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size);
}


int main()
{
	int const taille_blue_points(2);
	int const taille_red_points(4);
	int const taille_inputs(taille_blue_points + taille_red_points);
	int const taille_X = ((taille_blue_points + taille_red_points) / 2) * 3;

	std::cout << "tab 1 size : " << taille_blue_points << "\n"
		<< "tab 2 size : " << taille_red_points << "\n"
		<< "tab X size : " << taille_X << std::endl;

	// lire le tableau deux par deux pour avoir le x et y d'un point
	double blue_points[taille_blue_points] =
	{
		0.35, 0.5
	};


	double red_points[taille_red_points] = {
		0.6, 0.6,
		0.55, 0.7
	};


	// creation de X
	double* p_blue = blue_points;
	double* p_red = red_points;


	std::vector<double> vect_X;

	int cursor_X;
	for (cursor_X = 0; cursor_X < taille_blue_points; cursor_X++) {
		if (cursor_X % 2 == 0) {
			vect_X.push_back(1.0);
		}
		vect_X.push_back(p_blue[cursor_X]);
	}

	for (cursor_X = 0; cursor_X < taille_red_points; cursor_X++) {
		if (cursor_X % 2 == 0) {
			vect_X.push_back(1.0);
		}
		vect_X.push_back(p_red[cursor_X]);
	}

	for (cursor_X = 0; cursor_X < taille_X; cursor_X++) {
		std::cout << vect_X[cursor_X] << std::endl;
	}

	double X[taille_X];
	std::copy(vect_X.begin(), vect_X.end(), X);



	// creation de inputs
	double inputs[taille_inputs];

	int cursor_input = 0;

	for (cursor_input; cursor_input < taille_blue_points; cursor_input++) {
		inputs[cursor_input] = p_blue[cursor_input];
	}

	for (cursor_input = taille_blue_points; cursor_input < taille_blue_points + taille_red_points; cursor_input++) {
		inputs[cursor_input] = p_red[cursor_input - taille_blue_points];
	}

	for (cursor_input = 0; cursor_input < taille_inputs; cursor_input++) {
		std::cout << inputs[cursor_input] << std::endl;
	}

	double Y[3] = { 1.0, -1.0, -1.0 };



	double* model = linear_model_create();
	std::cout << linear_model_predict_regression(model, inputs, taille_inputs);


	return 0;
}