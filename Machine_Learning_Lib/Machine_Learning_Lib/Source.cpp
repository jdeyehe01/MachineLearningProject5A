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

	// faire un new de ce truc 
	struct MLP {
		int* npl; 
		int npl_size; 
		double*** w;// valeur des poids [0] =layer (l), [1] = i, [2] = j 
		double** x; // enregistrement des valeurs x résultant de la propagation 
		double** deltas; //  les deltas a stocker dans l'apprentissage 
	};

	// mlp_model_create([2, 3, 4, 1], 4); // a récupérer avec un IntPtr
	DLLEXPORT struct MLP* mlp_model_create(int* npl, int npl_size)
	{ 


		// npl tableau d'entier ( neurone per layer )
		// [2, 3, 4, 1]
		// 2 =>  Deux entrée
		// 3, 4 => deux couches cachée avec respectivement 3 et 4 neurones sans compter le neurone fictive 1
		// 1 => une sortie 

		// np)l_size = la taille du tableau


		// Création de la structure mlp pour sauvegarder tout ce dont on a besoin 



		// Nombre d'entrée + 1 pour l'entrée fictive
		/* Seule chose qui change entre le mlp de classification pour la régression et pour la classification c
		*/

		/*
			w[l][i][j]
			l => layer neurone arrive /-> taille nb layer
			i => position i neurone départ /-> taille nb neurones
			j => position j neurone arrive /-> taille nb neurones
		*/
		double*** w_ptr = new double** [npl_size];
		for (int layer = 1; layer < npl_size; ++layer) {
			w_ptr[layer] = new double* [npl[layer -1] + 1]; // nombre de neurone du layer l + neurone fictif
			
			for (int neurone_i = 0; neurone_i < npl[layer -1 ] + 1; ++neurone_i) {
				w_ptr[layer][neurone_i] = new double[npl[layer] + 1]; // nombre de neurone du layer l + neurone fictif

				for (int neurone_j = 0; neurone_j < npl[layer] + 1; ++neurone_j) {
					w_ptr[layer][neurone_i][neurone_j] = ((double) rand()) / RAND_MAX * 2.0 - 1.0;
				}
			}
		}


		double** x_ptr = new double* [npl_size]; // x[L][J]
		double** deltas_ptr = new double* [npl_size];
		for (auto l = 0; l < npl_size; l++) {
			//model->x[l] = new double[npl[l] + 1];
		}

		MLP* model = new MLP;

		model->npl = npl;
		model->npl_size = npl_size;
		model->w = w_ptr;
		model->x = x_ptr;
		model->deltas = deltas_ptr;

		return model;
	}
}