#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <cstdlib>
#include <Eigen/Dense>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace Eigen;

extern "C"
{
	DLLEXPORT double* linear_model_create(int dim_size)
	{
		int size = dim_size + 1;
		auto tab = new double[size];

		for (auto i = 0; i < size; i++)
		{
			tab[i] = ((double)rand() / (double)RAND_MAX);
		}

		return tab;
	}

	DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size)
	{
		double result = 0.0;
		for (int i = 1; i < inputs_size + 1; i++)
		{
			result += model[i] * inputs[i - 1];
		}
		return result + model[0];
	}

	DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size)
	{
		return linear_model_predict_regression(model, inputs, inputs_size) >= 0 ? 1.0 : -1.0;
	}

	DLLEXPORT void linear_model_train_classification(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs,
		int interations_count, double alpha)
	{
		for (int i = 0; i < interations_count; i++) {
			int indexRand = rand() % dataset_length;
			int pos = indexRand * inputs_size;
			double g_x_k = linear_model_predict_classification(model, &dataset_inputs[pos], inputs_size);
			double grad = alpha * (dataset_expected_outputs[indexRand] - g_x_k);
			model[0] += grad * 1;
			for (int k = 0; k < inputs_size; k++) {
				model[k + 1] += grad * dataset_inputs[pos + k];
			}
		}
	}

	DLLEXPORT void linear_model_train_regression(double* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs)
	{
		// Train PseudoInverse moore penrose
		MatrixXd x(dataset_length, inputs_size + 1);
		MatrixXd y(dataset_length, 1);

		for (int i = 0; i < dataset_length; i++) {
			y(i, 0) = dataset_expected_outputs[i];
			x(i, 0) = 1;

			for (int k = 1; k < (inputs_size + 1); k++) {
				x(i, k) = dataset_inputs[i * inputs_size + (k - 1)];
			}
		}

		MatrixXd result = ((x.transpose() * x).inverse() * x.transpose()) * y;
		for (int j = 0; j < inputs_size + 1; j++) {
			model[j] = result(j, 0);
		}
	}

	DLLEXPORT void linear_model_delete(double* model)
	{
		delete[] model;
	}

	struct MLP {
		int* npl;
		int npl_size;
		double*** w;// valeur des poids [0] =layer (l), [1] = i, [2] = j 
		double** x; // enregistrement des valeurs x resultant de la propagation 
		double** deltas; //  les deltas a stocker dans l'apprentissage 
	};

	void mlp_propagation(MLP* model, double* inputs, bool regression) {

		for (int j = 1; j < model->npl[0] + 1; j++) {
			model->x[0][j] = inputs[j - 1];
		}

		for (int l = 1; l < model->npl_size; l++) {
			for (int j = 1; j < model->npl[l] + 1; j++) {

				double sum = 0.0;

				for (int i = 0; i < model->npl[l - 1] + 1; i++) {
					sum += model->w[l][i][j] * model->x[l - 1][i];
				}

				model->x[l][j] = ((l == model->npl_size - 1) && regression) ? sum : tanh(sum);
			}
		}
	}


	double* mlp_propagation_and_extract_result(MLP* model, double* inputs, bool regression) {
		mlp_propagation(model, inputs, regression);
		double* result = new double[model->npl[model->npl_size - 1]];

		for (int j = 1; j < model->npl[model->npl_size - 1] + 1; j++) {

			result[j - 1] = model->x[model->npl_size - 1][j];
		}

		return result;
	}

	// mlp_model_create([2, 3, 4, 1], 4); // a récupérer avec un IntPtr
	DLLEXPORT struct MLP* mlp_model_create(int* npl, int npl_size)
	{
		auto model = new MLP();
		model->npl = new int[npl_size];

		for (int i = 0; i < npl_size; i++) {
			model->npl[i] = npl[i];
		}

		model->npl_size = npl_size;
		model->w = new double**[npl_size];

		for (int layer = 1; layer < npl_size; layer++) {
			model->w[layer] = new double*[npl[layer - 1] + 1];

			for (int i_neurones = 0; i_neurones < npl[layer - 1] + 1; i_neurones++) {
				model->w[layer][i_neurones] = new double[npl[layer] + 1];

				for (int j_neurones = 1; j_neurones < npl[layer] + 1; j_neurones++) {
					model->w[layer][i_neurones][j_neurones] = ((double)rand()) / RAND_MAX * 2.0 - 1.0;
				}
			}
		}

		model->x = new double*[npl_size];
		model->deltas = new double*[npl_size];

		for (int layer = 0; layer < npl_size; layer++) {
			model->x[layer] = new double[npl[layer] + 1];
			model->x[layer][0] = 1.0;
			model->deltas[layer] = new double[npl[layer] + 1];
		}

		return model;
	}

	DLLEXPORT double* mlp_model_predict_regression(struct MLP* model, double* inputs) {
		return mlp_propagation_and_extract_result(model, inputs, true);
	}


	DLLEXPORT double* mlp_model_predict_classification(struct MLP* model, double* inputs) {
		return mlp_propagation_and_extract_result(model, inputs, false);
	}

	DLLEXPORT void mlp_model_train_classification(struct MLP* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs,
		int outputs_size, int interations_count, double alpha) {

		for (int epoch = 0; epoch < interations_count; epoch++)
		{
			int random = (int)floor(((double)std::min(rand(), RAND_MAX - 1)) / RAND_MAX * dataset_length);
			auto inputs = dataset_inputs + random * inputs_size;
			auto expected_outputs = dataset_expected_outputs + random * outputs_size;

			mlp_propagation(model, inputs, false);

			for (int j = 1; j < model->npl[model->npl_size - 1] + 1; j++) {
				model->deltas[model->npl_size - 1][j] = (1 - pow(model->x[model->npl_size - 1][j], 2)) * (model->x[model->npl_size - 1][j] - expected_outputs[j - 1]);
			}

			for (int l = model->npl_size - 1; l >= 2; l--)
			{
				for (int i = 1; i < model->npl[l - 1] + 1; i++) {
					double sum = 0.0;
					for (int j = 1; j < model->npl[l] + 1; j++)
					{
						sum += model->w[l][i][j] * model->deltas[l][j];
					}
					model->deltas[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * sum;
				}
			}

			for (int l = 1; l < model->npl_size; l++)
			{
				for (int i = 0; i < model->npl[l - 1] + 1; i++)
				{
					for (int j = 1; j < model->npl[l] + 1; j++)
					{
						model->w[l][i][j] -= alpha * model->x[l - 1][i] * model->deltas[l][j];
					}
				}
			}
		}
	}

	DLLEXPORT void mlp_model_train_regression(struct MLP* model, double* dataset_inputs, int dataset_length, int inputs_size, double* dataset_expected_outputs,
		int outputs_size, int interations_count, double alpha)
	{
		for (int epoch = 0; epoch < interations_count; epoch++)
		{
			int random = (int)floor(((double)std::min(rand(), RAND_MAX - 1)) / RAND_MAX * dataset_length);
			auto inputs = dataset_inputs + random * inputs_size;
			auto expected_outputs = dataset_expected_outputs + random * outputs_size;

			mlp_propagation(model, inputs, true);

			for (int j = 1; j < model->npl[model->npl_size - 1] + 1; j++) {
				model->deltas[model->npl_size - 1][j] = (1 - pow(model->x[model->npl_size - 1][j], 2)) * (model->x[model->npl_size - 1][j] - expected_outputs[j - 1]);
			}

			for (int l = model->npl_size - 1; l >= 2; l--)
			{
				for (int i = 1; i < model->npl[l - 1] + 1; i++) {
					double sum = 0.0;
					for (int j = 1; j < model->npl[l] + 1; j++)
					{
						sum += model->w[l][i][j] * model->deltas[l][j];
					}
					model->deltas[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * sum;
				}
			}

			for (int l = 1; l < model->npl_size; l++)
			{
				for (int i = 0; i < model->npl[l - 1] + 1; i++)
				{
					for (int j = 1; j < model->npl[l] + 1; j++)
					{
						model->w[l][i][j] -= alpha * model->x[l - 1][i] * model->deltas[l][j];
					}
				}
			}
		}
	}

	DLLEXPORT void mlp_model_delete(double* model)
	{
		delete[] model;
	}
}