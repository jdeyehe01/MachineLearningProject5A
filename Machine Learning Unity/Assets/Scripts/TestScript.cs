using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

public class TestScript : MonoBehaviour
{
    public Transform[] trainSpheresTransforms;
    public Transform[] testSpheresTransforms;

    public double alpha = 0.001;
    public int iterationCount = 1000000;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    System.IntPtr CreateModel(int size)
    {
        Debug.Log("Create Model");
        var modelPtr = VisualStudioLibWrapper.linear_model_create(size);
        /*double[] model = new double[size+1];
        Marshal.Copy(modelPtr, model, 0, size + 1);
        Debug.Log(model[size]);*/

        return modelPtr;
    }

    void RegressionPredict(IntPtr model)
    {
        Debug.Log("Predict Model");

       foreach(var testSpheresTransform in testSpheresTransforms)
        {
            double [] inputs = new double[2];
            inputs[0] = testSpheresTransform.position.x;
            inputs[1] = testSpheresTransform.position.z;

            double result = VisualStudioLibWrapper.linear_model_predict_regression(model, inputs, inputs.Length);

            testSpheresTransform.position = new Vector3(
                testSpheresTransform.position.x,
                (float) result,
                testSpheresTransform.position.z
            );

            Debug.Log("Result Predict Model =  " + result);
        }

        //double[] inputs = input_array.ToArray();

       // Debug.Log("Taille tableau d'inputs  " + inputs.Length);
        //double result = VisualStudioLibWrapper.linear_model_predict_regression(model, inputs, inputs.Length);
        //Debug.Log("Result Predict Model =  " + result);
    }

    void RegressionTrain(IntPtr model)
    {
        Debug.Log("Train Model");
        List<double> inputs = new List<double>();
        List<double> expecteds = new List<double>();

        foreach (var trainSpheresTransform in trainSpheresTransforms)
        {
            inputs.Add(trainSpheresTransform.position.x);
            inputs.Add(trainSpheresTransform.position.z);

            expecteds.Add(trainSpheresTransform.position.y);

        }

        VisualStudioLibWrapper.linear_model_train_regression(model, inputs.ToArray() , inputs.Count() / 2, 2 , expecteds.ToArray());
    }

    void ClassificationPredict(IntPtr model)
    {
        Debug.Log("Predict Model");

        foreach (var testSpheresTransform in testSpheresTransforms)
        {
            double[] inputs = new double[2];
            inputs[0] = testSpheresTransform.position.x;
            inputs[1] = testSpheresTransform.position.z;

            double result = VisualStudioLibWrapper.linear_model_predict_classification(model, inputs, inputs.Length);

            testSpheresTransform.position = new Vector3(
                testSpheresTransform.position.x,
                (float)result,
                testSpheresTransform.position.z
            );

            Debug.Log("Result Predict Model =  " + result);
        }
    }

    void ClassificationTrain(IntPtr model)
    {
        Debug.Log("Train Model");
        List<double> inputs = new List<double>();
        List<double> expecteds = new List<double>();

        foreach (var trainSpheresTransform in trainSpheresTransforms)
        {
            inputs.Add(trainSpheresTransform.position.x);
            inputs.Add(trainSpheresTransform.position.z);

            expecteds.Add(trainSpheresTransform.position.y >= 0 ? 1.0 : -1.0 );
        }

        VisualStudioLibWrapper.linear_model_train_classification(model, inputs.ToArray(), inputs.Count() / 2, 2, expecteds.ToArray(), iterationCount, alpha);
    }


    void Delete(IntPtr model)
    {
        Debug.Log("Delete Model");
        VisualStudioLibWrapper.linear_model_delete(model);
    }


    public void LaunchRegression()
    {
        Debug.Log("Regression : Training and Testing");

        // Create Model
        var model = CreateModel(2);

        // Train Model
        RegressionTrain(model);

        RegressionPredict(model);

        //Delete
        Delete(model);
    }

    public void LaunchClassification()
    {
        Debug.Log("Classification : Training and Testing");

        // Create Model
        var model = CreateModel(2);

        // Train Model
        ClassificationTrain(model);

        ClassificationPredict(model);

        //Delete
        Delete(model);
    }
}
