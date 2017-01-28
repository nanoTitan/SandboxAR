/*============================================================================== 
 * Copyright (c) 2012-2015 Qualcomm Connected Experiences, Inc. All Rights Reserved. 
 * ==============================================================================*/
using UnityEngine;
using System;
using Vuforia;

/// <summary>
/// This script sets up the background shader effect and contains the logic
/// to capture longer touch "drag" events that distort the video background. 
/// </summary>
public class NegativeGrayscaleEffect : MonoBehaviour
{
    #region MONOBEHAVIOUR_METHODS
    void Update()
    {
        float touchX = 2.0f;
        float touchY = 2.0f;
        if (Input.GetMouseButton(0))
        {
            Vector2 touchPos = Input.mousePosition;
            // Adjust the touch point for the current orientation
            touchX = ((touchPos.x / Screen.width) - 0.5f) * 2.0f;
            touchY = ((touchPos.y / Screen.height) - 0.5f) * 2.0f;
        }

        // Pass the touch coordinates to the shader
        Material mat = GetComponent<Renderer>().material;
        mat.SetFloat("_TouchX", touchX);
        mat.SetFloat("_TouchY", touchY);
    }
    #endregion //MONOBEHAVIOUR_METHODS
}
