using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

public class TestVisualStudioScript : MonoBehaviour
{
    public Transform[] trainSpheresTransforms;
    public Transform[] testSpheresTransforms;




    // Start is called before the first frame update
    void Start()
    {/*
        Debug.Log("With Visual Studio DLL : ");
        double[] model = CreateModel(2);

        Train(model);

      Predict(model);*/
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    double[] CreateModel(int size)
    {
        Debug.Log("Create Model");
        var modelPtr = VisualStudioLibWrapper.linear_model_create(size);
        double[] model = new double[size+1];
        Marshal.Copy(modelPtr, model, 0, size + 1);
        Debug.Log(model[size]);

        return model;
    }

    void Predict(double[] model)
    {
        Debug.Log("Predict Model");
        //List<double> input_array = new List<double>();

       foreach(var testSpheresTransform in testSpheresTransforms)
        {
            //input_array.Add(testSpheresTransform.position.x);
            //input_array.Add(testSpheresTransform.position.z);
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

    void Train(double[] model)
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

            VisualStudioLibWrapper.linear_model_train_regression(model, inputs.ToArray() , inputs.Count() , 2 , expecteds.ToArray());


    }


    void Delete(double[] model)
    {
        Debug.Log("Delete Model");
        VisualStudioLibWrapper.linear_model_delete(model);
    }



    public void TrainAndTest()
    {
        Debug.Log("Training and Testing");


        // Create Model

        double[] model = CreateModel(2);

        // Train Model
        Train(model);

        Predict(model);

        //Delete
        //Delete(model);
    }
}
