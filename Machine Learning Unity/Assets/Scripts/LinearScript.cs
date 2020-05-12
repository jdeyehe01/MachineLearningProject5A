using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LinearScript : MonoBehaviour
{
    public Transform[] trainSpheresTransforms;

    public Transform[] testSpheresTransforms;

    public void TrainAndTest()
    {
        Debug.Log("Training and Testing");

        // Créer dataset_inputs
        // Créer dataest_expected_outputs

        // Create Model

        // Train Model

        Debug.Log("Number of test : " + testSpheresTransforms.Length);
        Debug.Log("Number of train : " + trainSpheresTransforms.Length);

        // For each testSphere : Predict 
        foreach (var testSpheresTransform in testSpheresTransforms)
        {
            testSpheresTransform.position += Vector3.up * 10;
        }

        // Delete Model
    }
}
