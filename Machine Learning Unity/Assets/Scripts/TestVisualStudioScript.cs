using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class TestVisualStudioScript : MonoBehaviour
{
    public Transform[] trainSpheresTransforms;
    public Transform[] testSpheresTransforms;

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("With Visual Studio DLL : ");
        double[] model = CreateModel();
        /*foreach ( var testSpheresTransform in testSpheresTransforms)
        {
            testSpheresTransform.position += Vector3.up * 10;
        }*/
        Predict(model);


    }

    // Update is called once per frame
    void Update()
    {
        
    }

    double[] CreateModel()
    {
        Debug.Log("Create Model");
        return VisualStudioLibWrapper.linear_model_create(10);
    }

    void Predict(double[] model)
    {
        Debug.Log("Predict Model");
        /*foreach ( var testSpheresTransform in testSpheresTransforms)
        {
        }*/


        //double result = VisualStudioLibWrapper.linear_model_predict_regression(model, testSpheresTransforms, 121);
        //Debug.Log("Result Predict Model =  " + result);
    }

    void Train()
    {
        Debug.Log("Train Model");
    }
}
