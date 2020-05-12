using System.Runtime.InteropServices;

public static class VisualStudioLibWrapper
{
    [DllImport("Machine_Learning_Lib")]
    public static extern System.IntPtr linear_model_create(int dim_size);

    [DllImport("Machine_Learning_Lib")]
    public static extern double linear_model_predict_regression(double[] model, double[] inputs, int inputs_size);

    [DllImport("Machine_Learning_Lib")]
    public static extern double linear_model_predict_classification(double[] model, double[] inputs, int inputs_size);

    [DllImport("Machine_Learning_Lib")]
    public static extern void linear_model_train_classification(double[] model, double[] dataset_inputs, int dataset_length, int inputs_size,
        double[] dataset_expected_outputs, int outputs_size, int interations_count, double alpha);

    [DllImport("Machine_Learning_Lib")]
    public static extern void linear_model_train_regression(double[] model, double[] dataset_inputs, int dataset_length, int inputs_size, double[] dataset_expected_outputs);

    [DllImport("Machine_Learning_Lib")]
    public static extern void linear_model_delete(double[] model);

    //double** => Marshal.Copy(IntPtr, Double[], Int32, Int32)
}