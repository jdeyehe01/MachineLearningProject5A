#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <cstdlib>
#include <Eigen/Dense>
#include <stdlib.h>
#include <iostream>

using namespace Eigen;


extern "C"
{
	DLLEXPORT double* linear_model_create(int dim_size)
	{
		auto tab = new double[dim_size + 1];

		for (auto i = 0; i < dim_size + 1; i++)
		{
			tab[i] = ((double)rand() / (double)RAND_MAX);
		}

		return tab;
	}

	DLLEXPORT double linear_model_predict_regression(double *model,double *inputs, int inputs_size)
	{
		// model W => { x0, x1, x2 }
		// inputs X => { x1, x2 }
		// W[0] + W[1] * X[1] + W[2] * X[2]

		double result = 0.0;
		for (size_t i = 1; i < inputs_size + 1; i++)
		{
			result += model[i] * inputs[i - 1];
		}
		return result + model[0];
	}

	DLLEXPORT double linear_model_predict_classification(double *model,double *inputs, int inputs_size)
	{
		// Meme chose que la regression mais avec un fonction signe
		return linear_model_predict_regression(model, inputs, inputs_size) >= 0 ? 1.0 : -1.0;
	}

	DLLEXPORT void linear_model_train_classification(double *model,double* dataset_inputs, int dataset_length, int inputs_size,double* dataset_expected_outputs,
		int interations_count,double alpha)
	{
		for (int i = 0; i < interations_count; i++) {

			int indexRand = rand() % dataset_length;
			int pos = indexRand * inputs_size;
			double g_x_k = linear_model_predict_classification(model, &dataset_inputs[pos] , inputs_size);
			double grad = alpha * (dataset_expected_outputs[indexRand] - g_x_k);
			model[0] += grad * 1;
			for (int k = 0; k < inputs_size; k++) {
				model[k + 1] += grad * dataset_inputs[pos+k];
			}
		}
	}

	DLLEXPORT void linear_model_train_regression(double *model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs)
	{
		// TODO : Train PseudoInverse moore penrose
		MatrixXd x(dataset_length, inputs_size + 1);
		MatrixXd y(dataset_length, 1);

		for (int i = 0; i < dataset_length; i++) {
			y(i, 0) = dataset_expected_outputs[i];
			x(i, 0) = 1;

			for (int k = 1; k < (inputs_size + 1); k++) {
				x(i, k) = dataset_inputs[i * inputs_size + (k-1)];
			}
		}

		MatrixXd result = ((x.transpose() * x).inverse() * x.transpose()) * y;
		for (int j = 0; j < inputs_size + 1; j++) {
			model[j] = result(j, 0);
		}
	}

	DLLEXPORT void linear_model_delete(double *model)
	{
		delete model;
	}

	struct MLP {
		int* npl;
		int npl_size;
		double*** w;
		double** x;
		double** deltas;
	};

	// mlp_model_create([2, 3, 4, 1], 4);
	DLLEXPORT struct MLP* mlp_model_create(int* npl, int npl_size)
	{
		// TODO
		return 0;
	}
}