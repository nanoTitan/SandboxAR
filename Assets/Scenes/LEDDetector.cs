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
    public class OpenCVJob : ThreadedJob
    {
        Mat rgbaMat;
        Mat renderMat;
        Mat blurredMat;
        Mat hsvMat;
        Mat maskMat;
        Mat morphOutputMat;
        Mat dilateElement;
        Mat erodeElement;
        Mat hierarchy;

        // HSV: 180-230, 0-100, 50-100
        Scalar minHSV = new Scalar(0, 0, 0);
        Scalar maxHSV = new Scalar(180, 255, 255);

        float focalLength = 10.0f; // 790.65f;
        float targetWidth = 62.0f * 0.0804f;  // translate mm to world units

        Point center1 = null;
        Point center2 = null;
        bool[] circleFound = null;
        Color32[] colors = null;
        List<MatOfPoint> m_contours = null;
        Texture2D m_texture;
        Renderer m_vuforiaRenderer;

        public void SetMinHSV(float h, float s, float v)
        {
            minHSV.val[0] = h;
            minHSV.val[1] = s;
            minHSV.val[2] = v;
        }

        public void SetMaxHSV(float h, float s, float v)
        {
            maxHSV.val[0] = h;
            maxHSV.val[1] = s;
            maxHSV.val[2] = v;
        }

        public void InitOpenCVJob(Texture2D texture, Renderer vuforiaRenderer)
        {
            m_texture = texture;
            m_vuforiaRenderer = vuforiaRenderer;

            blurredMat = new Mat();
            hsvMat = new Mat();
            maskMat = new Mat();
            morphOutputMat = new Mat();
            dilateElement = new Mat();
            erodeElement = new Mat();
            hierarchy = new Mat();

            if (m_texture.format == TextureFormat.RGB24)
            {
                rgbaMat = new Mat(m_texture.height, m_texture.width, CvType.CV_8UC3);
            }
            else if (m_texture.format == TextureFormat.RGBA32)
            {
                rgbaMat = new Mat(m_texture.height, m_texture.width, CvType.CV_8UC4);
            }
            else
            {
                Debug.Log("Unsupported camera texture format detected: " + m_texture.format.ToString());
            }

            renderMat = new Mat(m_texture.height, m_texture.width, CvType.CV_8UC4);
            colors = new Color32[m_texture.width * m_texture.height];
            m_contours = new List<MatOfPoint>();
            circleFound = new bool[2] { false, false };
            center1 = new Point { x = 0, y = 0 };
            center2 = new Point { x = 0, y = 0 };

            IsDone = true;
        }

        public void UpdateMatRGBA(Texture2D texture)
        {
            Utils.fastTexture2DToMat(texture, rgbaMat);
        }

        protected override void ThreadFunction()
        {
            if (rgbaMat == null)
                return;

            Core.flip(rgbaMat, rgbaMat, 0);

            renderMat.setTo(new Scalar(0, 0, 0, 0));

            Imgproc.blur(rgbaMat, blurredMat, new Size(7, 7));
            Imgproc.cvtColor(blurredMat, hsvMat, Imgproc.COLOR_BGR2HSV);

            Core.inRange(hsvMat, minHSV, maxHSV, maskMat);

            // morphological operators
            // dilate with large element, erode with small ones
            dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24, 24));
            erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));

            Imgproc.erode(maskMat, morphOutputMat, erodeElement);
            Imgproc.erode(maskMat, morphOutputMat, erodeElement);

            Imgproc.dilate(maskMat, morphOutputMat, dilateElement);
            Imgproc.dilate(maskMat, morphOutputMat, dilateElement);

            m_contours.Clear();
            circleFound[0] = false;
            circleFound[1] = false;

            // find contours
            Imgproc.findContours(morphOutputMat, m_contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        }

        public bool GetTrackableInfo(ref Vector3 pos)
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

            if (!circleFound[0] || !circleFound[1])
            {
                return false;
            }

            // Target's Center screen space
            float tcX = (float)((center1.x + center2.x) * 0.5);
            float tcY = (float)((center1.y + center2.y) * 0.5);
            Vector2 targetCenter = new Vector2(tcX, tcY);

            float P = Mathf.Abs((float)(center1.x - center2.x));
            float D = targetWidth * focalLength / P;

            //Debug.Log("P: " + P);

            // Rotate Camera
            float halfW = VuforiaRenderer.Instance.VideoBackgroundTexture.width * 0.5f;
            float halfH = VuforiaRenderer.Instance.VideoBackgroundTexture.height * 0.5f;
            float percntX = (targetCenter.x - halfW) / VuforiaRenderer.Instance.VideoBackgroundTexture.width;
            float percntY = (targetCenter.y - halfH) / VuforiaRenderer.Instance.VideoBackgroundTexture.height;
            float dX = m_vuforiaRenderer.transform.lossyScale.x * 2 * percntX;
            float dZ = m_vuforiaRenderer.transform.lossyScale.z * 2 * percntY;    // use z for forward

            Vector3 newPt = m_vuforiaRenderer.transform.position;
            newPt += (m_vuforiaRenderer.transform.right * dX);
            newPt -= (m_vuforiaRenderer.transform.forward * dZ);

            Vector3 newForward = (newPt - Camera.main.transform.position).normalized;
            pos = Camera.main.transform.position + (newForward * D);

            return true;
        }

        // This is executed by the Unity main thread when the job is finished
        protected override void OnFinished()
        {
            /*
            // if any contour exist...
            if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
            {
                MatOfPoint currCtr = null;
                double maxArea = 0, currArea = 0;
                float[] radius1 = new float[1];
                float[] radius2 = new float[1];

                // for each contour, display it
                for (int idx = 0; idx >= 0; idx = (int)hierarchy.get(0, idx)[0])
                {
					if(idx > m_contours.Count)
						break;

                    currCtr = m_contours[idx];
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
                    Imgproc.drawContours(renderMat, m_contours, idx, new Scalar(255, 0, 0, 255));
                }

                if (circleFound[0])
                    Imgproc.circle(renderMat, center1, (int)radius1[0], new Scalar(0, 255, 0, 255), 2);

                if (circleFound[1])
                    Imgproc.circle(renderMat, center2, (int)radius2[0], new Scalar(0, 0, 255, 255), 2);
            }
            
            Utils.matToTexture2D(renderMat, m_texture, colors);
            */
        }
    }
    
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
        
        OpenCVJob m_job;
        
        // Use this for initialization
        void Start ()
        {
            VuforiaARController.Instance.RegisterVuforiaStartedCallback(OnVuforiaStarted);
            VuforiaARController.Instance.RegisterTrackablesUpdatedCallback(OnVuforiaTrackablesUpdated);

            targetPosDebug.SetActive(false);

            StartCoroutine(InitMats());
        }

        private void OnVuforiaStarted()
        {
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

            // Create our render target's Texture2D
            Texture2D newTexture = new Texture2D(bgTexture.width, bgTexture.height, TextureFormat.RGBA32, false);
            renderTarget.material.mainTexture = newTexture;

            // Make sure our render target tracks Vuforia's                    
            renderTarget.transform.position = vuforiaRenderTarget.transform.position;
            renderTarget.transform.localRotation = Quaternion.identity;
            renderTarget.transform.localScale = new Vector3(
                vuforiaRenderTarget.transform.localScale.x * 2,
                vuforiaRenderTarget.transform.localScale.z * 2,
                vuforiaRenderTarget.transform.localScale.y * 2);

            m_job = new OpenCVJob();
            m_job.InitOpenCVJob(newTexture, vuforiaRenderTarget);

            OnMinValueSlider();
            OnMaxValueSlider();
        }

        // Only update when Vuforia has updated it's frame
        private void OnVuforiaTrackablesUpdated()
        {
            StartCoroutine(TrackTargets());
        }

		private IEnumerator TrackTargets()
		{
            if (m_job == null || !VuforiaRenderer.Instance.IsVideoBackgroundInfoAvailable() || !VuforiaRenderer.Instance.VideoBackgroundTexture)
            {
                yield break;
            }

            // Check if job is completed before starting again
            if (m_job.IsDone)
            {
                m_job.UpdateMatRGBA((Texture2D)VuforiaRenderer.Instance.VideoBackgroundTexture);
                m_job.Start();
                yield return StartCoroutine(m_job.WaitFor());

                // Debug target pos
                Vector3 newPos = new Vector3();
                if(m_job.GetTrackableInfo(ref newPos))
                {
                    if (!targetPosDebug.activeSelf)
                        targetPosDebug.SetActive(true);

                    targetPosDebug.transform.position = newPos;
                }

            }
        }

        public void OnMinValueSlider()
        {
            m_job.SetMinHSV(hMinSlider.value, sMinSlider.value, vMinSlider.value);
        }

        public void OnMaxValueSlider()
        {
            m_job.SetMaxHSV(hMaxSlider.value, sMaxSlider.value, vMaxSlider.value);
        }
    }
}
