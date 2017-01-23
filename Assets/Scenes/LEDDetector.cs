using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Collections;
using Vuforia;

#if UNITY_5_3 || UNITY_5_3_OR_NEWER
using UnityEngine.SceneManagement;
#endif
using OpenCVForUnity;

namespace OpenCVForUnitySample
{
    /// <summary>
    /// WebCamTexture detect face sample.
    /// </summary>
    public class LEDDetector : MonoBehaviour
    {
        [SerializeField] Slider hMinSlider;
        [SerializeField] Slider hMaxSlider;
        [SerializeField] Slider sMinSlider;
        [SerializeField] Slider sMaxSlider;
        [SerializeField] Slider vMinSlider;
        [SerializeField] Slider vMaxSlider;
        [SerializeField] Renderer renderTarget;
        [SerializeField] Renderer vuforiaRenderTarget;
        [SerializeField] GameObject targetPosDebug;

        Texture2D texture;
        WebCamTextureToMatHelper webCamTextureToMatHelper;
        Mat rgbaMat;
        Mat renderMat;
        Mat blurredMat;
        Mat hsvMat;
        Mat maskMat;
        Mat morphOutputMat;

        // HSV: 180-230, 0-100, 50-100
        Scalar minHSV = new Scalar(0, 0, 0);
        Scalar maxHSV = new Scalar(180, 255, 255);

        float focalLength = 10.0f; // 790.65f;
        float targetWidth = 62.0f * 0.0804f;  // translate mm to world units

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
            //webCamTextureToMatHelper.Init ();
#endif

            VuforiaARController.Instance.RegisterVuforiaStartedCallback(OnVuforiaStarted);
            VuforiaARController.Instance.RegisterTrackablesUpdatedCallback(OnVuforiaTrackablesUpdated);

            OnHMinValueSlider();
            OnHMaxValueSlider();
            OnSMinValueSlider();
            OnSMaxValueSlider();
            OnVMinValueSlider();
            OnVMaxValueSlider();

            targetPosDebug.SetActive(false);

            StartCoroutine(InitMats());
        }

        private void OnVuforiaStarted()
        {
            // Vuforia has started, now register camera image format
            //if (CameraDevice.Instance.SetFrameFormat(mPixelFormat, true))
            //{
            //    Debug.Log("Successfully registered pixel format " + mPixelFormat.ToString());
            //    mFormatRegistered = true;
            //}
            //else
            //{
            //    Debug.LogError("Failed to register pixel format " + mPixelFormat.ToString() +
            //        "\n the format may be unsupported by your device;" +
            //        "\n consider using a different pixel format.");
            //    mFormatRegistered = false;
            //}

            OnWebCamTextureToMatHelperInited();
        }

        IEnumerator InitMats()
        {
            while(!VuforiaRenderer.Instance.IsVideoBackgroundInfoAvailable() || !VuforiaRenderer.Instance.VideoBackgroundTexture)
            {
                yield return new WaitForSeconds(0.1f);
            }

            Texture2D bgTexture = (Texture2D)VuforiaRenderer.Instance.VideoBackgroundTexture;
            while ((bgTexture.format != TextureFormat.RGB24 && bgTexture.format != TextureFormat.RGBA32))
            {
                yield return new WaitForSeconds(0.1f);
            }

            blurredMat = new Mat();
            hsvMat = new Mat();
            maskMat = new Mat();
            morphOutputMat = new Mat();

            if (bgTexture.format == TextureFormat.RGB24)
            {
                rgbaMat = new Mat(bgTexture.height, bgTexture.width, CvType.CV_8UC3);
            }
            else if (bgTexture.format == TextureFormat.RGBA32)
            {
                rgbaMat = new Mat(bgTexture.height, bgTexture.width, CvType.CV_8UC4);
            }
            else
            {
                Debug.Log("Unsupported camera texture format detected: " + bgTexture.format.ToString());
                yield return null;
            }

            renderMat = new Mat(bgTexture.height, bgTexture.width, CvType.CV_8UC4);

            // Create our render target's Texture2D
            texture = new Texture2D(bgTexture.width, bgTexture.height, TextureFormat.RGBA32, false);
            renderTarget.material.mainTexture = texture;

            // Make sure our render target tracks Vuforia's                    
            renderTarget.transform.position = vuforiaRenderTarget.transform.position;
            renderTarget.transform.localRotation = Quaternion.identity;
            renderTarget.transform.localScale = new Vector3(
                vuforiaRenderTarget.transform.localScale.x * 2,
                vuforiaRenderTarget.transform.localScale.z * 2,
                vuforiaRenderTarget.transform.localScale.y * 2);

            colors = new Color32[bgTexture.width * bgTexture.height];         
        }

        /// <summary>
        /// Raises the web cam texture to mat helper inited event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInited ()
        {
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
        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log ("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

        // Only update when Vuforia has updated it's frame
        private void OnVuforiaTrackablesUpdated()
        {
            if (rgbaMat == null)
                return;

            if (!VuforiaRenderer.Instance.IsVideoBackgroundInfoAvailable() || !VuforiaRenderer.Instance.VideoBackgroundTexture)
            {
                return;
            }

            Texture2D bgTexture = (Texture2D)VuforiaRenderer.Instance.VideoBackgroundTexture;
            Utils.texture2DToMat(bgTexture, rgbaMat);

            Core.flip(rgbaMat, rgbaMat, 0);

            renderMat.setTo(new Scalar(0, 0, 0, 0));

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
                    Imgproc.drawContours(renderMat, contours, idx, new Scalar(255, 0, 0, 255)); 
                }

                if(circleFound[0])
                    Imgproc.circle(renderMat, center1, (int)radius1[0], new Scalar(0, 255, 0, 255), 2);

                if (circleFound[1])
                    Imgproc.circle(renderMat, center2, (int)radius2[0], new Scalar(0, 0, 255, 255), 2);

                if (circleFound[0] && circleFound[1])
                {
                    Vector3 pos = new Vector3();
                    Quaternion rot = new Quaternion();
                    GetTrackableInfo(center1, center2, ref pos, ref rot);
                }
            }

            //Core.flip(renderMat, renderMat, 0);
            Utils.matToTexture2D(renderMat, texture, colors);
        }

        void GetTrackableInfo(Point c1, Point c2, ref Vector3 pos, ref Quaternion rot)
        {
            //IEnumerable<CameraDevice.CameraField> fields = CameraDevice.Instance.GetCameraFields();
            //foreach(CameraDevice.CameraField f in fields)
            //{
            //    Debug.Log("key; " + f.Key + ", type :" + f.Type);
            //}

            /*
            f = (P * D) / W   or  D = W * f / P
            f: focal length
            P: apparent width in pixels
            W: known width (mm or inches)
            D: distance to camera (mm or inches) 
            */

            // Target's Center screen space
            float tcX = (float)((c1.x + c2.x) * 0.5);
            float tcY = (float)((c1.y + c2.y) * 0.5);
            Vector2 targetCenter = new Vector2(tcX, tcY);

            float P = Mathf.Abs((float)(c1.x - c2.x));
            float D = targetWidth * focalLength / P;

            Debug.Log("P: " + P);

            // Rotate Camera
            float halfW = VuforiaRenderer.Instance.VideoBackgroundTexture.width * 0.5f;
            float halfH = VuforiaRenderer.Instance.VideoBackgroundTexture.height * 0.5f;
            float percntX = (targetCenter.x - halfW) / VuforiaRenderer.Instance.VideoBackgroundTexture.width;
            float percntY = (targetCenter.y - halfH) / VuforiaRenderer.Instance.VideoBackgroundTexture.height;
            float dX = vuforiaRenderTarget.transform.lossyScale.x * 2 * percntX;
            float dZ = vuforiaRenderTarget.transform.lossyScale.z * 2 * percntY;    // use z for forward

            Vector3 newPt = vuforiaRenderTarget.transform.position;
            newPt += (vuforiaRenderTarget.transform.right * dX);
            newPt -= (vuforiaRenderTarget.transform.forward * dZ);

            Vector3 newForward = (newPt - Camera.main.transform.position).normalized;

            pos = Camera.main.transform.position + (newForward * D);

            // Debug target pos
            if (!targetPosDebug.activeSelf)
                targetPosDebug.SetActive(true);
            targetPosDebug.transform.position = pos;
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
