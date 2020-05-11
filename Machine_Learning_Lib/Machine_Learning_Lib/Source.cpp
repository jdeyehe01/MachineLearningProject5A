#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include  <cstdlib>

extern "C"
{
	DLLEXPORT int my_add(int x, int y)
	{
		return x + y + 2;
	}

	DLLEXPORT int my_mul(int x, int y)
	{
		return x * y;
	}

	DLLEXPORT double* linear_create_modele(int input_dim) {
		/*
		static double r[10];
		for (int i = 0; i < input_dim; i++) {
			r[i] = rand() % 1;
		}

		return r;
		*/
	}


	DLLEXPORT double linear_model_predict_regression(double *model,double *inputs, int inputs_size)
	{
		// TODO
		return 0.0;
	}

	DLLEXPORT double linear_model_predict_classification(double *model,double *inputs, int inputs_size)
	{
		return linear_model_predict_regression(model, inputs, inputs_size) >= 0 ?
			1.0 : -1.0;
	}

	DLLEXPORT void linear_model_train_classification(double *model,double* dataset_inputs, int dataset_length, int inputs_size,double* dataset_expected_outputs, int outputs_size,
		int interations_count, float alpha)
	{
		for (int i = 0; i <= interations_count; i++) {

			int indexRand = rand() %  sizeof(dataset_inputs);
			double g_x_k = linear_model_predict_classification(model, &dataset_inputs[indexRand] , inputs_size);

			for (int k = 0; k <= dataset_inputs[indexRand]; k++) {
				model[indexRand + 1] += alpha * (dataset_expected_outputs[indexRand]);
			}

		}
		// TODO : Train Rosenblatt
	}

	DLLEXPORT void linear_model_train_regression(double *model,double* dataset_inputs, int dataset_length, int inputs_size,double* dataset_expected_outputs, int outputs_size/*,
		int interations_count, float alpha*/)
	{
		// TODO : Train PseudoInverse moore penrose
	}

	DLLEXPORT void linear_model_delete(double *model)
	{
		// TODO : Delete XXXX ?
	}
}