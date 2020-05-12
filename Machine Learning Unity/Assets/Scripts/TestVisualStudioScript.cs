using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestVisualStudioScript : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("With Visual Studio DLL : ");
        Debug.Log(VisualStudioLibWrapper.my_add(42, 51));
        Debug.Log(VisualStudioLibWrapper.my_mul(2, 3));
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
