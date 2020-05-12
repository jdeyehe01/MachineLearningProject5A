#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>
#include <Eigen/Dense>
#include <vector>

extern "C" {
	DLLEXPORT double* linear_model_create(int dim_size);
	DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size);
	DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size);
	DLLEXPORT void linear_model_train_regression(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size);
	DLLEXPORT void linear_model_train_classification(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs, int outputs_size,
		int interations_count, double alpha);
}


/*int main()
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



	double* model = linear_model_create(2);
	linear_model_train_classification(model, inputs, 3, 2, Y, 3, 10000, 0.01);


	//double result = linear_model_predict_regression(model, inputs, taille_inputs);
	//std::cout << "resultat" << result << std::endl;

}*/

int main() {

	double blue_points[1][2] = {
		{0.35, 0.5}
	};

	double red_points[2][2] = {
		{0.6, 0.6},
		{0.55, 0.7}
	};

	double X[3][3] = {
		{1, 0.35, 0.5},
		{1, 0.6, 0.6},
		{1, 0.55, 0.7}
	};

	//double inputs[3][2]{
	//    {0.35, 0.5 },
	//    {0.6, 0.6 },
	//    {0.55, 0.7 }
	//};

	//double** inputs = new double*[3];
	//for (int i = 0; i < 3; i++) {
	//    inputs[i] = new double[2];
	//}
	//inputs[0][0] = 0.35;
	//inputs[0][1] = 0.5;
	//inputs[1][0] = 0.6;
	//inputs[1][1] = 0.6;
	//inputs[2][0] = 0.55;
	//inputs[2][1] = 0.7;


	double* inputs = new double[6];
	inputs[0] = 0.35;
	inputs[1] = 0.5;
	inputs[2] = 0.6;
	inputs[3] = 0.6;
	inputs[4] = 0.55;
	inputs[5] = 0.7;


	double Y[3] = { 1, -1, -1 };

	double* model = linear_model_create(2);
	/*double* model = new double[3];
	model[0] = 0.519944;
	model[1] = 0.00601215;
	model[2] = 0.226081;*/
	for (int i = 0; i < 3; i++) {
		std::cout << "model =" << model[i] << " " << std::endl;
	}
	std::cout << linear_model_predict_classification(model, &(inputs[0]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[2]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[4]), 2);

	//linear_model_train_classification(model, inputs, 3, 2, Y, 3, 1000000, 0.01);
	linear_model_train_regression(model, inputs, 3, 2, Y, 3);

	std::cout << linear_model_predict_classification(model, &(inputs[0]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[2]), 2);
	std::cout << linear_model_predict_classification(model, &(inputs[4]), 2);
	std::cout << std::endl;

	for (int i = 0; i < 3; i++) {
		std::cout << model[i] << " ";
	}

	return 0;
}