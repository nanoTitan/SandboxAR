using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

#if UNITY_5_3 || UNITY_5_3_OR_NEWER
using UnityEngine.SceneManagement;
#endif
using OpenCVForUnity;

namespace OpenCVForUnitySample
{
    /// <summary>
    /// WebCamTexture detect face sample.
    /// </summary>
    public class LEDDetector2 : MonoBehaviour
    {
        [SerializeField] Slider hMinSlider;
        [SerializeField] Slider hMaxSlider;
        [SerializeField] Slider sMinSlider;
        [SerializeField] Slider sMaxSlider;
        [SerializeField] Slider vMinSlider;
        [SerializeField] Slider vMaxSlider;

        /// <summary>
        /// The texture.
        /// </summary>
        Texture2D texture;

        /// <summary>
        /// The web cam texture to mat helper.
        /// </summary>
        WebCamTextureToMatHelper webCamTextureToMatHelper;

        Mat blurredMat;
        Mat hsvMat;
        Mat maskMat;
        Mat morphOutputMat;

        // HSV: 180-230, 0-100, 50-100
        Scalar minHSV = new Scalar(0, 0, 0);
        Scalar maxHSV = new Scalar(360, 100, 100);

        // Color buffer to prevent having to reallocate each frame
        Color32[] colors = null;

        // Use this for initialization
        void Start ()
        {
            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper> ();

            #if UNITY_WEBGL && !UNITY_EDITOR
            StartCoroutine (Utils.getFilePathAsync ("lbpcascade_frontalface.xml", (result) => {
                cascade = new CascadeClassifier ();
                cascade.load (result);

                webCamTextureToMatHelper.Init ();
            }));

            #else

            webCamTextureToMatHelper.Init ();
            #endif
            
            OnHMinValueSlider();
            OnHMaxValueSlider();
            OnSMinValueSlider();
            OnSMaxValueSlider();
            OnVMinValueSlider();
            OnVMaxValueSlider();
        }

        /// <summary>
        /// Raises the web cam texture to mat helper inited event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInited ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperInited");
            
            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat ();

            texture = new Texture2D (webCamTextureMat.cols (), webCamTextureMat.rows (), TextureFormat.RGBA32, false);

            gameObject.GetComponent<Renderer> ().material.mainTexture = texture;

            gameObject.transform.localScale = new Vector3 (webCamTextureMat.cols (), webCamTextureMat.rows (), 1);
            Debug.Log ("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);
            
            float width = webCamTextureMat.width ();
            float height = webCamTextureMat.height ();
            
            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }
            
            blurredMat = new Mat();
            hsvMat = new Mat();
            maskMat = new Mat();
            morphOutputMat = new Mat();

            colors = new Color32[texture.width * texture.height];
        }

        /// <summary>
        /// Raises the web cam texture to mat helper disposed event.
        /// </summary>
        public void OnWebCamTextureToMatHelperDisposed ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperDisposed");

            if (blurredMat != null)
                blurredMat.Dispose ();

            if (hsvMat != null)
                hsvMat.Dispose ();

            if (maskMat != null)
                maskMat.Dispose();

            if (morphOutputMat != null)
                morphOutputMat.Dispose();
        }

        /// <summary>
        /// Raises the web cam texture to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode){
            Debug.Log ("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

        // Update is called once per frame
        void Update ()
        {
            if (webCamTextureToMatHelper.IsPlaying () && webCamTextureToMatHelper.DidUpdateThisFrame ())
            {
                Mat rgbaMat = webCamTextureToMatHelper.GetMat ();

                // Empty Matrix
                Mat test = new Mat(new Size(texture.width, texture.height), CvType.CV_8UC4);

                Imgproc.blur(rgbaMat, blurredMat, new Size(7, 7));
                Imgproc.cvtColor (blurredMat, hsvMat, Imgproc.COLOR_BGR2HSV);

                Core.inRange(hsvMat, minHSV, maxHSV, maskMat);

                // Test Screen: hsv color clamped
                //Utils.matToTexture2D(maskMat, texture, webCamTextureToMatHelper.GetBufferColors());

                // morphological operators
                // dilate with large element, erode with small ones
                Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24, 24));
                Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));

                Imgproc.erode(maskMat, morphOutputMat, erodeElement);
                Imgproc.erode(maskMat, morphOutputMat, erodeElement);

                Imgproc.dilate(maskMat, morphOutputMat, dilateElement);
                Imgproc.dilate(maskMat, morphOutputMat, dilateElement);

                List<MatOfPoint> contours = new List<MatOfPoint>();
                Mat hierarchy = new Mat();

                // find contours
                Imgproc.findContours(morphOutputMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

                // if any contour exist...
                if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
                {
                    MatOfPoint currCtr = null;
                    double maxArea = 0, currArea = 0;
                    float[] radius1 = new float[1];
                    float[] radius2 = new float[1];
                    Point center1 = new Point { x = 0, y = 0 };
                    Point center2 = new Point { x = 0, y = 0 };
                    bool[] circleFound = new bool[2] {false, false};

                    // for each contour, display it
                    for (int idx = 0; idx >= 0; idx = (int)hierarchy.get(0, idx)[0])
                    {
                        currCtr = contours[idx];
                        currArea = Imgproc.contourArea(currCtr);
                        //if (currArea > maxArea )
                        {
                            MatOfPoint2f c2f = new MatOfPoint2f(currCtr.toArray());
                            if (!circleFound[0])
                            {   
                                Imgproc.minEnclosingCircle(c2f, center1, radius1);
                                circleFound[0] = true;
                            }
                            else
                            {
                                Imgproc.minEnclosingCircle(c2f, center2, radius2);
                                circleFound[1] = true;
                            }

                            maxArea = currArea;
                        }

                        // Draw the countour
                        Imgproc.drawContours(test, contours, idx, new Scalar(255, 0, 0, 255)); 
                    }

                    if(circleFound[0])
                        Imgproc.circle(test, center1, (int)radius1[0], new Scalar(0, 255, 0, 255), 2);

                    if (circleFound[1])
                        Imgproc.circle(test, center2, (int)radius2[0], new Scalar(0, 0, 255, 255), 2);
                }
                
                
                for (int i = 0; i < colors.Length; ++i)
                {
                    //colors[i].r = 255;
                    //colors[i].g = 255;
                    //colors[i].b = 255;
                    //colors[i].a = 255;
                }
                Utils.matToTexture2D(test, texture, colors);

                //Utils.matToTexture2D(rgbaMat, texture, webCamTextureToMatHelper.GetBufferColors());
            }
        }

        /// <summary>
        /// Raises the disable event.
        /// </summary>
        void OnDisable ()
        {
            webCamTextureToMatHelper.Dispose ();
        }

        /// <summary>
        /// Raises the back button event.
        /// </summary>
        public void OnBackButton ()
        {
            #if UNITY_5_3 || UNITY_5_3_OR_NEWER
            SceneManager.LoadScene ("OpenCVForUnitySample");
            #else
            Application.LoadLevel ("OpenCVForUnitySample");
            #endif
        }

        /// <summary>
        /// Raises the play button event.
        /// </summary>
        public void OnPlayButton ()
        {
            webCamTextureToMatHelper.Play ();
        }

        /// <summary>
        /// Raises the pause button event.
        /// </summary>
        public void OnPauseButton ()
        {
            webCamTextureToMatHelper.Pause ();
        }

        /// <summary>
        /// Raises the stop button event.
        /// </summary>
        public void OnStopButton ()
        {
            webCamTextureToMatHelper.Stop ();
        }

        /// <summary>
        /// Raises the change camera button event.
        /// </summary>
        public void OnChangeCameraButton ()
        {
            webCamTextureToMatHelper.Init (null, webCamTextureToMatHelper.requestWidth, webCamTextureToMatHelper.requestHeight, !webCamTextureToMatHelper.requestIsFrontFacing);
        }

        public void OnHMinValueSlider()
        {
            minHSV.val[0] = hMinSlider.value;
        }

        public void OnHMaxValueSlider()
        {
            maxHSV.val[0] = hMaxSlider.value;
        }

        public void OnSMinValueSlider()
        {
            minHSV.val[1] = sMinSlider.value;
        }

        public void OnSMaxValueSlider()
        {
            maxHSV.val[1] = sMaxSlider.value;
        }

        public void OnVMinValueSlider()
        {
            minHSV.val[2] = vMinSlider.value;
        }

        public void OnVMaxValueSlider()
        {
            maxHSV.val[2] = vMaxSlider.value;
        }
    }
}
