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
    public enum TrackableID
    {
        LEDRed1 = 0,
        LEDRed2,
        LEDGreen1,
        LEDGreen2,

        NumTrackableIDs // Make sure this is last for total count
    }

    public class LEDTrackable
    {
        public LEDTrackable()
        {
            Center = new Point(0, 0);
            Radius = 0;
            Area = 0;
            Found = false;
        }

        public Point Center { get; set; }
        public float Radius { get; set; }
        public double Area { get; set; }
        public bool Found { get; set; }
    }

    public class OpenCVJob : ThreadedJob
    {
        Mat rgbaMat;
        Mat rgbMat;
        Mat renderMat;
        Mat blurredMat;
        Mat m_hsvMat;
        Mat threshold;
        Mat morphOutputMat;
        Mat dilateElement;
        Mat erodeElement;
        Mat m_redHierarchy;
        Mat m_greenHierarchy;
        Mat m_linesMat;
        Mat m_hist;

        // HSV: 180-230, 0-100, 50-100
        Scalar minRedHSV = new Scalar(0, 0, 201);
        Scalar maxRedHSV = new Scalar(45, 60, 255);
        //Scalar minRedHSV = new Scalar(0, 78, 45);
        //Scalar maxRedHSV = new Scalar(8, 255, 211);
        Scalar minGreenHSV = new Scalar(63, 0, 180);
        Scalar maxGreenHSV = new Scalar(90, 42, 255);

        float m_contrast = 0.2f;
        byte m_brightness = 0;
        float focalLength = 10.0f; // 790.65f;
        float targetWidth = 62.0f * 0.0804f;  // translate mm to world units
        Vuforia.Image.PIXEL_FORMAT m_pixelFormat = Vuforia.Image.PIXEL_FORMAT.RGB888;
        LEDTrackable[] m_ledTrackableArray = null;
        Color32[] colors = null;
        List<MatOfPoint> m_redContours = null;
        List<MatOfPoint> m_greenContours = null;
        List<Mat> m_images = new List<Mat>();
        Renderer m_vuforiaRenderer;
        OpenCVForUnity.Rect m_trackWindow = new OpenCVForUnity.Rect(100, 100, 100, 100);

        int debugPrint = 0;

        public Mat RenderMat { get { return renderMat; } }
        public Texture2D Texture { get; set; }

        public Color32[] Colors
        {
            get { return colors; }
        }

public void SetMinRedHSV(float h, float s, float v)
        {
            minRedHSV.val[0] = h;
            minRedHSV.val[1] = s;
            minRedHSV.val[2] = v;
        }

        public void SetMaxRedHSV(float h, float s, float v)
        {
            maxRedHSV.val[0] = h;
            maxRedHSV.val[1] = s;
            maxRedHSV.val[2] = v;
        }

        public void SetMinGreenHSV(float h, float s, float v)
        {
            minGreenHSV.val[0] = h;
            minGreenHSV.val[1] = s;
            minGreenHSV.val[2] = v;
        }

        public void SetMaxGreenHSV(float h, float s, float v)
        {
            maxGreenHSV.val[0] = h;
            maxGreenHSV.val[1] = s;
            maxGreenHSV.val[2] = v;
        }

        public void InitOpenCVJob(Vuforia.Image image, Vuforia.Image.PIXEL_FORMAT format, Texture2D texture, Renderer vuforiaRenderer)
        {
			m_pixelFormat = format;
            Texture = texture;
            m_vuforiaRenderer = vuforiaRenderer;

            blurredMat = new Mat();
            m_hsvMat = new Mat();
			threshold = new Mat();
            morphOutputMat = new Mat();
            m_redHierarchy = new Mat();
            m_greenHierarchy = new Mat();

            if (m_pixelFormat == Vuforia.Image.PIXEL_FORMAT.RGB888)
			{
				rgbMat = new Mat(new Size(image.Width, image.Height), CvType.CV_8UC3);
				rgbaMat = null;
			}
			else
			{
				rgbMat = new Mat();
				rgbaMat = new Mat(new Size(image.Width, image.Height), CvType.CV_8UC4);
			}
            
            renderMat = new Mat(Texture.height, Texture.width, CvType.CV_8UC4);
            colors = new Color32[Texture.width * Texture.height];
            m_redContours = new List<MatOfPoint>();
            m_greenContours = new List<MatOfPoint>();

            m_ledTrackableArray = new LEDTrackable[(int)TrackableID.NumTrackableIDs];
            for(int i = 0; i < m_ledTrackableArray.Length; ++i)
                m_ledTrackableArray[i] = new LEDTrackable();

            InitMeanShiftTracking();
        }
        
        public void UpdateMat(Vuforia.Image image)
        {
            //Utils.fastTexture2DToMat(texture, rgbaMat);
            //Utils.texture2DToMat(texture, rgbaMat);

			if(m_pixelFormat == Vuforia.Image.PIXEL_FORMAT.RGB888)
			{
            	rgbMat.put(0, 0, image.Pixels);
			}
			else
			{
				rgbaMat.put(0, 0, image.Pixels);
			}
        }

        protected override void ThreadFunction()
        {
			if(m_pixelFormat == Vuforia.Image.PIXEL_FORMAT.RGB888 && rgbMat == null)
			{
				return;
			}

			if(m_pixelFormat == Vuforia.Image.PIXEL_FORMAT.RGBA8888)
			{
	            if (rgbaMat == null)
	                return;

				Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB);
			}

            // Adjust contrast and brightness
            //rgbMat.convertTo(rgbMat, -1, m_contrast, m_brightness);

            // Blur
            //Imgproc.blur(rgbMat, blurredMat, new Size(7, 7));

            renderMat.setTo(new Scalar(0, 0, 0, 0));
            m_hsvMat = new Mat();
            Imgproc.cvtColor(rgbMat, m_hsvMat, Imgproc.COLOR_RGB2HSV);

            FindCountours(minRedHSV, maxRedHSV, m_redContours, m_redHierarchy);
            FindCountours(minGreenHSV, maxGreenHSV, m_greenContours, m_greenHierarchy);
            //FindLines();
            MeanShiftTracking();
        }

        void FindCountours(Scalar min, Scalar max, List<MatOfPoint> contours, Mat hierarchy)
        {
            Core.inRange(m_hsvMat, min, max, threshold);

            // morphological operators
            // dilate with large element, erode with small ones
            dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8));
            erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

            Imgproc.erode(threshold, threshold, erodeElement);
            Imgproc.erode(threshold, threshold, erodeElement);

            Imgproc.dilate(threshold, threshold, dilateElement);
            Imgproc.dilate(threshold, threshold, dilateElement);

            contours.Clear();

            // find contours
            Imgproc.findContours(threshold, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        }

        void FindLines()
        {
            int lowThreshold = 100;
            int ratio = 3;

            Mat detectedEdges = new Mat();
            Mat grayImage = new Mat();
            Imgproc.cvtColor(rgbMat, grayImage, Imgproc.COLOR_RGB2GRAY);

            Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));
            Imgproc.Canny(detectedEdges, detectedEdges, lowThreshold, lowThreshold * ratio, 3, false);

            rgbMat.copyTo(renderMat, detectedEdges);

            // Hough Lines
            //m_linesMat = new Mat();
            //Imgproc.HoughLinesP(detectedEdges, m_linesMat, 1, System.Math.PI / 180, 50, 20, 5);

            //for (int i = 0; i < m_linesMat.cols(); i++)
            //{
            //    double[] val = m_linesMat.get(0, i);
            //    if (val == null)                        // Prevent silent crash!
            //        return;

            //    Imgproc.line(renderMat, new Point(val[0], val[1]), new Point(val[2], val[3]), new Scalar(0, 0, 255), 2);
            //}

            Mat dst = new Mat();
            Core.inRange(renderMat, minGreenHSV, maxGreenHSV, dst);
            //dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            //erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));

            renderMat.setTo(new Scalar(0, 0, 0));
            Imgproc.cvtColor(detectedEdges, renderMat, Imgproc.COLOR_GRAY2RGB);

            //Imgproc.erode(dst, dst, erodeElement);
            //Imgproc.erode(dst, dst, erodeElement);

            //Imgproc.dilate(dst, dst, dilateElement);
            //Imgproc.dilate(dst, dst, dilateElement);

            Imgproc.findContours(dst, m_greenContours, m_greenHierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        }

        bool InitMeanShiftTracking()
        {
            if (rgbMat == null || rgbMat.rows() == 0 || rgbMat.height() == 0)
                return false;

            Mat roiMat = rgbMat.submat(m_trackWindow);
            Mat hsvMat = new Mat();
            Mat mask = new Mat();
            m_hist = new Mat();
            MatOfInt histSize = new MatOfInt(256);
            MatOfFloat histRange = new MatOfFloat(0f, 256f);

            Imgproc.cvtColor(roiMat, hsvMat, Imgproc.COLOR_RGB2HSV);
            Core.inRange(hsvMat, minRedHSV, maxRedHSV, mask);

            List<Mat> images = new List<Mat> { hsvMat };
            MatOfInt channels = new MatOfInt(0);
            Imgproc.calcHist(images, channels, mask, m_hist, histSize, histRange);
            Core.normalize(m_hist, m_hist, 0, 255, Core.NORM_MINMAX);

            return true;
        }

        void MeanShiftTracking()
        {
            if (m_hist == null)
            {
                if (!InitMeanShiftTracking())
                    return;
            }
            
            Mat hsvMat = new Mat();
            MatOfFloat histRange = new MatOfFloat(0f, 256f);
            TermCriteria tc = new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 80, 1);

            Imgproc.cvtColor(rgbMat, hsvMat, Imgproc.COLOR_RGB2HSV);

            List<Mat> images = new List<Mat> { hsvMat };
            MatOfInt channels = new MatOfInt(0);
            Mat dst = new Mat();
            Imgproc.calcBackProject(images, channels, m_hist, dst, histRange, 1);
            Video.meanShift(dst, m_trackWindow, tc);

            Imgproc.rectangle(renderMat, new Point(m_trackWindow.x, m_trackWindow.y), new Point(m_trackWindow.x + m_trackWindow.width, m_trackWindow.y + m_trackWindow.height), new Scalar(255, 0, 0, 255), 2);
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

            // Target's Center screen space
            Point center = new Point();

            foreach (LEDTrackable led in m_ledTrackableArray)
            {
                if(!led.Found)
                    return false;

                center += led.Center;
            }

            // Target's Center screen space
            float oneOverNumTrackables = 1.0f / (float)TrackableID.NumTrackableIDs;
            Vector2 targetCenter = new Vector2((float)(center.x * oneOverNumTrackables), (float)(center.y * oneOverNumTrackables));

            float P = Mathf.Abs((float)(m_ledTrackableArray[0].Center.x - m_ledTrackableArray[1].Center.x));
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

            if (float.IsInfinity(pos.x) || float.IsInfinity(pos.y) || float.IsInfinity(pos.z))
                return false;

            return true;
        }

        // This is executed by the Unity main thread when the job is finished
        protected override void OnFinished()
        {
            // Reset LEDs to not found
            foreach (LEDTrackable led in m_ledTrackableArray)
                led.Found = false;

            FindLEDs(m_redContours, m_redHierarchy, m_ledTrackableArray[(int)TrackableID.LEDRed1], m_ledTrackableArray[(int)TrackableID.LEDRed2], true);
            FindLEDs(m_greenContours, m_greenHierarchy, m_ledTrackableArray[(int)TrackableID.LEDGreen1], m_ledTrackableArray[(int)TrackableID.LEDGreen2], true);
            
            Utils.matToTexture2D(renderMat, Texture, colors);
			//Utils.matToTexture2D(threshold, m_texture, colors);
        }

        void FindLEDs(List<MatOfPoint> contours, Mat hierarchy, LEDTrackable led1, LEDTrackable led2, bool doDraw)
        {
            led1.Found = false;
            led2.Found = false;

            // TODO: Change min/max HSV, blur, or erode/diolate values to lower contour count
            //if (contours.Count > 2)
            //    return;

            //Debug.Log("hierarchy.rows(): " + hierarchy.rows());

            // if any contour exist...
            if (hierarchy.rows() > 0)
            {
                MatOfPoint currCtr = null;
                led1.Area = 0;
                led2.Area = 0;
                led1.Radius = 0;
                led2.Radius = 0;
                double currArea = 0;
                float[] radiusArray = new float[1];

                // for each contour, display it
                for (int idx = 0; idx >= 0; idx = (int)hierarchy.get(0, idx)[0])
                {
                    if (idx > contours.Count)
                        break;

                    currCtr = contours[idx];
                    currArea = Imgproc.contourArea(currCtr);
                    if (currArea > led1.Area || currArea > led2.Area)
                    {
                        MatOfPoint2f c2f = new MatOfPoint2f(currCtr.toArray());
                        if (!led1.Found || (currArea > led1.Area && led1.Area <= led2.Area))
                        {
                            Imgproc.minEnclosingCircle(c2f, led1.Center, radiusArray);
                            led1.Found = true;
                            led1.Area = currArea;
                            led1.Radius = radiusArray[0];
                        }
                        else if (!led2.Found || (currArea > led2.Area))
                        {
                            Imgproc.minEnclosingCircle(c2f, led2.Center, radiusArray);
                            led2.Found = true;
                            led2.Area = currArea;
                            led2.Radius = radiusArray[0];
                        }
                    }

                    // Draw the countour
                    if(doDraw)
                        Imgproc.drawContours(renderMat, contours, idx, new Scalar(255, 0, 0, 255));
                }

                if (doDraw && led1.Found)
                    Imgproc.circle(renderMat, led1.Center, (int)led1.Radius, new Scalar(255, 0, 0, 255), 2);

                if (doDraw && led2.Found)
                    Imgproc.circle(renderMat, led2.Center, (int)led2.Radius, new Scalar(255, 0, 0, 255), 2);
            }
        }
    }
    
    public class LEDDetector : MonoBehaviour
    {
        [SerializeField] Slider hMinRedSlider;
        [SerializeField] Slider hMaxRedSlider;
        [SerializeField] Slider sMinRedSlider;
        [SerializeField] Slider sMaxRedSlider;
        [SerializeField] Slider vMinRedSlider;
        [SerializeField] Slider vMaxRedSlider;
        [SerializeField] Slider hMinGreenSlider;
        [SerializeField] Slider hMaxGreenSlider;
        [SerializeField] Slider sMinGreenSlider;
        [SerializeField] Slider sMaxGreenSlider;
        [SerializeField] Slider vMinGreenSlider;
        [SerializeField] Slider vMaxGreenSlider;
        [SerializeField] Renderer renderTarget;
        [SerializeField] Renderer vuforiaRenderTarget;
        [SerializeField] GameObject targetPosDebug;
        
        OpenCVJob[] m_jobs;
        Vector3[] m_targetPosArray;
        int m_maxTargetPosSz = 10;
        int m_currTargetPosIndx = 0;
        Vuforia.Image image = null;
        Vuforia.Image.PIXEL_FORMAT m_pixelFormat = Vuforia.Image.PIXEL_FORMAT.RGB888;
        bool m_formatRegistered = false;
		bool m_lastFrameTracked = false;

		public static bool IsMobile()
		{
			return SystemInfo.deviceModel.Contains("iPad") || SystemInfo.deviceModel.Contains("iPhone");
		}

        // Use this for initialization
        void Start ()
        {
            VuforiaARController.Instance.RegisterVuforiaStartedCallback(OnVuforiaStarted);
            VuforiaARController.Instance.RegisterOnPauseCallback(OnVuforiaPause);
            VuforiaARController.Instance.RegisterTrackablesUpdatedCallback(OnVuforiaTrackablesUpdated);
            
            targetPosDebug.SetActive(false);
        }

        private void OnVuforiaStarted()
        {
            RegisterFormat();
            StartCoroutine(InitMats());
        }

        private void OnVuforiaPause(bool paused)
        {
            if (paused)
            {
                Debug.Log("App was paused");
                UnregisterFormat();
            }
            else
            {
                Debug.Log("App was resumed");
                RegisterFormat();
            }
        }

        private void UnregisterFormat()
        {
            Debug.Log("Unregistering camera pixel format " + m_pixelFormat.ToString());
            CameraDevice.Instance.SetFrameFormat(m_pixelFormat, false);
            m_formatRegistered = false;
        }
        
        private void RegisterFormat()
        {
            // Vuforia has started, now register camera image format
			if (CameraDevice.Instance.SetFrameFormat(Vuforia.Image.PIXEL_FORMAT.RGBA8888, true))
			{
				m_pixelFormat = Vuforia.Image.PIXEL_FORMAT.RGBA8888;
				Debug.Log("Successfully registered pixel format " + m_pixelFormat.ToString());
				m_formatRegistered = true;
			}
			else if (CameraDevice.Instance.SetFrameFormat(Vuforia.Image.PIXEL_FORMAT.RGB888, true))
            {
				m_pixelFormat = Vuforia.Image.PIXEL_FORMAT.RGB888;
                Debug.Log("Successfully registered pixel format " + m_pixelFormat.ToString());
                m_formatRegistered = true;
            }
            else
            {
                Debug.LogError("Failed to register pixel format " + m_pixelFormat.ToString() +
                    "\n the format may be unsupported by your device;" +
                    "\n consider using a different pixel format.");
                m_formatRegistered = false;
            }
        }

        IEnumerator InitMats()
        {
            while (!m_formatRegistered)
            {
                yield return new WaitForSeconds(0.1f);
            }

            image = CameraDevice.Instance.GetCameraImage(m_pixelFormat);
            while (image == null || !image.IsValid())
            {
                yield return new WaitForSeconds(0.1f);
                image = CameraDevice.Instance.GetCameraImage(m_pixelFormat);
            }

            // Create our render target's Texture2D
            Texture2D newTexture = new Texture2D(image.Width, image.Height, TextureFormat.RGBA32, false);
            renderTarget.material.mainTexture = newTexture;

            // Make sure our render target tracks Vuforia's                    
            renderTarget.transform.position = vuforiaRenderTarget.transform.position;

			if(IsMobile())
				renderTarget.transform.localRotation = Quaternion.Euler(0,0,-90);
			else
				renderTarget.transform.localRotation = Quaternion.identity;

            renderTarget.transform.localScale = new Vector3(
                vuforiaRenderTarget.transform.localScale.x * 2,
                vuforiaRenderTarget.transform.localScale.z * 2,
                vuforiaRenderTarget.transform.localScale.y * 2);

			m_jobs = new OpenCVJob[1];
			for(int i = 0; i < m_jobs.Length; ++i)
			{
				m_jobs[i] = new OpenCVJob();
				m_jobs[i].InitOpenCVJob(image, m_pixelFormat, newTexture, vuforiaRenderTarget);
				m_jobs[i].Start();
			}

            OnMinRedValueSlider();
            OnMaxRedValueSlider();
            OnMinGreenValueSlider();
            OnMaxGreenValueSlider();
        }

        // Only update when Vuforia has updated it's frame
        private void OnVuforiaTrackablesUpdated()
        {	
            StartCoroutine(TrackTargets());
        }

        OpenCVJob GetAvailableJobThread()
        {
            OpenCVJob currJob = null;
            int i = 0;
            foreach (OpenCVJob job in m_jobs)
            {
                if (job.JobState == ThreadedJobState.Idle)
                {
                    currJob = job;
                    // Debug.Log("job: " + i);
                    break;
                }

                ++i;
            }

            return currJob;
        }

        void UpdateTargetPos(OpenCVJob job)
        {
            Vector3 newPos = new Vector3();
            if (job.GetTrackableInfo(ref newPos))
            {
                if (!targetPosDebug.activeSelf)
                    targetPosDebug.SetActive(true);

                if (m_targetPosArray == null)
                {
                    m_targetPosArray = new Vector3[m_maxTargetPosSz];
                    for (int indx = 0; indx < m_targetPosArray.Length; ++indx)
                        m_targetPosArray[indx] = newPos;
                }

                m_targetPosArray[m_currTargetPosIndx++] = newPos;
                if (m_currTargetPosIndx >= m_maxTargetPosSz)
                    m_currTargetPosIndx = 0;

                Vector3 newCenter = Vector3.zero;
                foreach(Vector3 v in m_targetPosArray)
                    newCenter += v;

                targetPosDebug.transform.position = newCenter / m_targetPosArray.Length;
            }
        }

		private IEnumerator TrackTargets()
		{
            if (!m_formatRegistered || m_jobs == null)
                yield break;

            //Vuforia.Image image = CameraDevice.Instance.GetCameraImage(m_pixelFormat);
            if (image == null || !image.IsValid())
                yield break;

            // Check if job is completed before starting again
            OpenCVJob currJob = GetAvailableJobThread();
            if (currJob != null)
            {
				m_lastFrameTracked = !m_lastFrameTracked;

				if(!m_lastFrameTracked)
				{
					m_lastFrameTracked = false;
					yield break;
				}

				currJob.UpdateMat(image);
				currJob.Work();
				yield return StartCoroutine(currJob.WaitFor());
                
                UpdateTargetPos(currJob);
            }
        }

        public void OnMinRedValueSlider()
        {
			foreach(OpenCVJob job in m_jobs)
            	job.SetMinRedHSV(hMinRedSlider.value, sMinRedSlider.value, vMinRedSlider.value);
        }

        public void OnMaxRedValueSlider()
        {
			foreach(OpenCVJob job in m_jobs)
            	job.SetMaxRedHSV(hMaxRedSlider.value, sMaxRedSlider.value, vMaxRedSlider.value);
        }

        public void OnMinGreenValueSlider()
        {
            foreach (OpenCVJob job in m_jobs)
                job.SetMinGreenHSV(hMinGreenSlider.value, sMinGreenSlider.value, vMinGreenSlider.value);
        }

        public void OnMaxGreenValueSlider()
        {
            foreach (OpenCVJob job in m_jobs)
                job.SetMaxGreenHSV(hMaxGreenSlider.value, sMaxGreenSlider.value, vMaxGreenSlider.value);
        }
    }
}
