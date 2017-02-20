using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Collections;
using Vuforia;
using System;

#if UNITY_5_3 || UNITY_5_3_OR_NEWER
using UnityEngine.SceneManagement;
#endif
using OpenCVForUnity;

namespace OpenCVForUnitySample
{
    public enum LEDColor
    {
        LED_RED = 0,
        LED_GREEN = 1,
        LED_BLUE = 2,
    }

    public enum TrackableID
    {
        LEDRed1 = 0,
        LEDRed2,
        LEDGreen1,
        LEDGreen2,

        NumTrackableIDs // Make sure this is last for total count
    }

    public class LEDContourInfo
    {
        public LEDContourInfo()
        {
            Center = new Vector2(0, 0);
            Radius = 0;
            Area = 0;
            ColorCode = LEDColor.LED_RED;
            MinR2R = double.MaxValue;
            MinR2G = double.MaxValue;
            MinG2G = double.MaxValue;
        }

        public Vector2 Center { get; set; }
        public float Radius { get; set; }
        public float Area { get; set; }
        public double MinR2R { get; set; }
        public double MinR2G { get; set; }
        public double MinG2G { get; set; }
        public LEDColor ColorCode { get; set; }
    }

    public class LEDTrackable
    {
        public LEDTrackable()
        {
            Center = new Vector2(0, 0);
            Radius = 0;
            Area = 0;
            Found = false;
        }

        public Vector2 Center { get; set; }
        public float Radius { get; set; }
        public float Area { get; set; }
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
        Scalar maxRedHSV = new Scalar(180, 75, 255);
        Scalar minBlueHSV = new Scalar(91, 0, 232);
        Scalar maxBlueHSV = new Scalar(116, 255, 255); 
        Scalar minGreenHSV = new Scalar(78, 0, 180);
        Scalar maxGreenHSV = new Scalar(104, 82, 255);
        
        float focalLength = 10.0f; // 790.65f;
        float targetWidth = 62.0f * 0.0804f;  // translate mm to world units
        float m_P = 0;  // Apparent width in pixels
        Vuforia.Image.PIXEL_FORMAT m_pixelFormat = Vuforia.Image.PIXEL_FORMAT.RGB888;
        List<LEDContourInfo> m_ledCountourList;
        LEDTrackable[] m_ledTrackableArray = null;
        Color32[] colors = null;
        List<MatOfPoint> m_redContours = null;
        List<MatOfPoint> m_greenContours = null;
        List<Mat> m_images = new List<Mat>();
        Renderer m_vuforiaRenderer;
        OpenCVForUnity.Rect m_trackWindow = new OpenCVForUnity.Rect(100, 100, 100, 100);
        bool m_isTracking = false;
        Vector2 m_trackCenter;

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

            //InitMeanShiftTracking();
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

            // Blur
            Imgproc.blur(rgbMat, blurredMat, new Size(7, 7));

            renderMat.setTo(new Scalar(0, 0, 0, 0));
            m_hsvMat = new Mat();
            Imgproc.cvtColor(blurredMat, m_hsvMat, Imgproc.COLOR_RGB2HSV);

            FindCountours(m_hsvMat, minRedHSV, maxRedHSV, m_redContours, m_redHierarchy);
            FindCountours(m_hsvMat, minGreenHSV, maxGreenHSV, m_greenContours, m_greenHierarchy);
            //FindLines();
            //MeanShiftTracking();
        }

        void FindCountours(Mat src, Scalar min, Scalar max, List<MatOfPoint> contours, Mat hierarchy)
        {
            Mat dst = new Mat();
            Core.inRange(src, min, max, dst);

            // morphological operators
            // dilate with large element, erode with small ones
            dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8));
            erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

            Imgproc.erode(dst, dst, erodeElement);
            Imgproc.erode(dst, dst, erodeElement);

            //Imgproc.dilate(dst, dst, dilateElement);
            Imgproc.dilate(dst, dst, dilateElement);

            contours.Clear();

            // find contours
            Imgproc.findContours(dst, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        }

        void FindLines()
        {
            int lowThreshold = 50;
            int ratio = 3;

            Mat detectedEdges = new Mat();
            Mat grayImage = new Mat();
            Mat dst = new Mat();

            Core.inRange(rgbMat, minGreenHSV, maxGreenHSV, dst);
            Imgproc.blur(dst, detectedEdges, new Size(3, 3));

            //Imgproc.cvtColor(rgbMat, grayImage, Imgproc.COLOR_RGB2GRAY);
            //Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

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

            //Mat dst = new Mat();
            //Core.inRange(renderMat, minGreenHSV, maxGreenHSV, dst);
            //dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            //erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));

            //renderMat.setTo(new Scalar(0, 0, 0));
            //Imgproc.cvtColor(detectedEdges, renderMat, Imgproc.COLOR_GRAY2RGB);

            //Imgproc.erode(dst, dst, erodeElement);
            //Imgproc.erode(dst, dst, erodeElement);

            //Imgproc.dilate(dst, dst, dilateElement);
            //Imgproc.dilate(dst, dst, dilateElement);

            //Imgproc.findContours(dst, m_greenContours, m_greenHierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
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

            if (!m_isTracking)
                return false;

            float D = targetWidth * focalLength / m_P;

            //Debug.Log("P: " + P);

            // Rotate Camera
            float halfW = VuforiaRenderer.Instance.VideoBackgroundTexture.width * 0.5f;
            float halfH = VuforiaRenderer.Instance.VideoBackgroundTexture.height * 0.5f;
            float percntX = (m_trackCenter.x - halfW) / VuforiaRenderer.Instance.VideoBackgroundTexture.width;
            float percntY = (m_trackCenter.y - halfH) / VuforiaRenderer.Instance.VideoBackgroundTexture.height;
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

            m_ledCountourList = new List<LEDContourInfo>();
            ParseContours(m_redContours, m_redHierarchy, m_ledCountourList, LEDColor.LED_RED);
            ParseContours(m_greenContours, m_greenHierarchy, m_ledCountourList, LEDColor.LED_GREEN);

            // Find the target position based on best fit for color tracking
            int redIdx1 = -1, redIdx2 = -1, greenIdx1 = -1, greenIdx2 = -1;
            m_isTracking = FindLEDs(ref redIdx1, ref redIdx2, ref greenIdx1, ref greenIdx2);
            DrawLEDs(redIdx1, redIdx2, greenIdx1, greenIdx2);

            Utils.matToTexture2D(renderMat, Texture, colors);
            //Utils.matToTexture2D(threshold, m_texture, colors);

            if (!m_isTracking)
                return;

            Vector2 c = (m_ledCountourList[redIdx1].Center + m_ledCountourList[redIdx2].Center + m_ledCountourList[greenIdx1].Center + m_ledCountourList[greenIdx2].Center) * 0.25f;
            m_trackCenter = new Vector2(c.x, c.y);

            Vector2 d = m_ledCountourList[redIdx1].Center - m_ledCountourList[redIdx2].Center;
            m_P = Mathf.Sqrt( (d.x * d.x + d.y * d.y) );
        }

        void ParseContours(List<MatOfPoint> contours, Mat hierarchy, List<LEDContourInfo> contourInfoList, LEDColor ledColor)
        {
            // Red contours
            if (hierarchy.rows() > 0)
            {
                MatOfPoint currCntr = null;
                float[] radiusArray = new float[1];

                // for each contour, display it
                for (int idx = 0; idx >= 0; idx = (int)hierarchy.get(0, idx)[0])
                {
                    if (idx > contours.Count)
                        break;
                    
                    currCntr = contours[idx];                    
                    MatOfPoint2f c2f = new MatOfPoint2f(currCntr.toArray());
                    LEDContourInfo info = new LEDContourInfo();
                    Point center = new Point();
                    Imgproc.minEnclosingCircle(c2f, center, radiusArray);
                    info.Area = (float)Imgproc.contourArea(currCntr);
                    info.Radius = radiusArray[0];
                    info.ColorCode = ledColor;
                    info.Center = new Vector2((float)center.x, (float)center.y);
                    contourInfoList.Add(info);

                    if(ledColor == LEDColor.LED_RED)
                        Imgproc.drawContours(renderMat, contours, idx, new Scalar(255, 0, 0, 255));
                    else
                        Imgproc.drawContours(renderMat, contours, idx, new Scalar(0, 255, 0, 255));
                }
            }
        }

        void DrawLEDs(int redIdx1, int redIdx2, int greenIdx1, int greenIdx2)
        {
            if(redIdx1 != -1)
                Imgproc.circle(renderMat, new Point(m_ledCountourList[redIdx1].Center.x, m_ledCountourList[redIdx1].Center.y), (int)m_ledCountourList[redIdx1].Radius, new Scalar(255, 0, 0, 255), 2);

            if(redIdx2 != -1)
                Imgproc.circle(renderMat, new Point(m_ledCountourList[redIdx2].Center.x, m_ledCountourList[redIdx2].Center.y), (int)m_ledCountourList[redIdx2].Radius, new Scalar(255, 0, 0, 255), 2);

            if(greenIdx1 != -1)
                Imgproc.circle(renderMat, new Point(m_ledCountourList[greenIdx1].Center.x, m_ledCountourList[greenIdx1].Center.y), (int)m_ledCountourList[greenIdx1].Radius, new Scalar(0, 255, 0, 255), 2);

            if(greenIdx2 != -1)
                Imgproc.circle(renderMat, new Point(m_ledCountourList[greenIdx2].Center.x, m_ledCountourList[greenIdx2].Center.y), (int)m_ledCountourList[greenIdx2].Radius, new Scalar(0, 255, 0, 255), 2);
        }

        bool FindLEDs(ref int redIdx1, ref int redIdx2, ref int greenIdx1, ref int greenIdx2)
        {
            Vector2 currCenter;
            double errorMax = double.MaxValue;
            double errorMaxArea = double.MaxValue;
            double errorWeighted = double.MaxValue;

            LEDContourInfo[] ledInfo = new LEDContourInfo[4];

            for (int i = 0; i < m_ledCountourList.Count; ++i)
            {
                if (m_ledCountourList[i].ColorCode != LEDColor.LED_RED)
                    continue;

                ledInfo[0] = m_ledCountourList[i];

                for (int j = 0; j < m_ledCountourList.Count; ++j)
                {
                    if (i == j)
                        continue;
                    
                    if (m_ledCountourList[j].ColorCode != LEDColor.LED_RED)
                        continue;

                    ledInfo[1] = m_ledCountourList[j];
                    for (int m = 0; m < m_ledCountourList.Count; ++m)
                    {
                        if (m_ledCountourList[m].ColorCode != LEDColor.LED_GREEN)
                            continue;

                        ledInfo[2] = m_ledCountourList[m];
                        for (int n = 0; n < m_ledCountourList.Count; ++n)
                        {
                            if (m == n)
                                continue;

                            if (m_ledCountourList[n].ColorCode != LEDColor.LED_GREEN)
                                continue;

                            ledInfo[3] = m_ledCountourList[n];
                            Vector2 center;
                            float errorDist = 0;
                            float errorArea = 0;
                            bool found = GetCenterInfo(ledInfo, out center, ref errorDist, ref errorArea);

                            double errW = errorDist * 2 + errorArea * 1;
                            if(found && errW < errorWeighted)
                            {
                                currCenter = center;
                                errorMax = errorDist;
                                errorMaxArea = errorArea;
                                errorWeighted = errW;
                                redIdx1 = i;
                                redIdx2 = j;
                                greenIdx1 = m;
                                greenIdx2 = n;
                            }
                        }
                    }
                }
            }

            if (errorMax == double.MaxValue || errorMaxArea == double.MaxValue)
                return false;

            Debug.Log("e: " + errorMax + ", a: " + errorMaxArea);

            return true;
        }

        public bool GetCenterInfo(LEDContourInfo[] ledInfo, out Vector2 center, ref float errorDist, ref float errorArea)
        {
            // Find center by average
            center = (ledInfo[0].Center + ledInfo[1].Center + ledInfo[2].Center + ledInfo[3].Center) * 0.25f;

            if (SamePoint(ledInfo[0], ledInfo[2]) || SamePoint(ledInfo[0], ledInfo[3]) ||
                SamePoint(ledInfo[1], ledInfo[2]) || SamePoint(ledInfo[1], ledInfo[3]) ||
                SamePoint(ledInfo[0], ledInfo[1]) || SamePoint(ledInfo[2], ledInfo[3]) )
            {
                return false;
            }

            float minArea = float.MaxValue, maxArea = 0;
            for (int i = 0; i < 4; ++i)
            {
                for (int j = i+1; j < 4; ++j)
                {
                    if (ledInfo[i].Radius > ledInfo[j].Radius)
                    {
                        if (ledInfo[i].Radius * 0.5f > ledInfo[j].Radius)
                            return false;
                    }
                    else
                    {
                        if (ledInfo[i].Radius * 0.5f > ledInfo[j].Radius)
                            return false;
                    }
                }

                if (ledInfo[i].Area > maxArea)
                    maxArea = ledInfo[i].Area;

                if (ledInfo[i].Area < minArea)
                    minArea = ledInfo[i].Area;
            }

            float l1 = (ledInfo[0].Center - ledInfo[1].Center).sqrMagnitude;   // Red to red
            float l2 = (ledInfo[2].Center - ledInfo[3].Center).sqrMagnitude;   // green to green
            float l3 = (ledInfo[0].Center - ledInfo[2].Center).sqrMagnitude;   // green to red
            float l4 = (ledInfo[1].Center - ledInfo[3].Center).sqrMagnitude;   // green to red
            float l5 = (ledInfo[0].Center - ledInfo[3].Center).sqrMagnitude;   // green to red
            float l6 = (ledInfo[1].Center - ledInfo[2].Center).sqrMagnitude;   // green to red

            // Parallel
            if (l1 < l2)
                errorDist += Math.Abs(1 - l1 / l2);
            else
                errorDist += Math.Abs(1 - l2 / l1);

            // Parallel
            if (l3 < l4)
                errorDist += Math.Abs(1 - l3 / l4);
            else
                errorDist += Math.Abs(1 - l5 / l3);

            // Diagonal
            if (l5 < l6)
                errorDist += Math.Abs(1 - l5 / l6);
            else
                errorDist += Math.Abs(1 - l6 / l5);

            // Area
            if (minArea < maxArea)
                errorArea = minArea / maxArea;
            else
                errorArea = maxArea / minArea;

            return true;
        }

        bool SamePoint(LEDContourInfo info1, LEDContourInfo info2)
        {
            float l = (info1.Center - info2.Center).sqrMagnitude;
            return l < (info1.Radius * info1.Radius) || l < (info2.Radius * info2.Radius);
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
        int m_maxTargetPosSz = 1;
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
            else
            {
                targetPosDebug.SetActive(false);
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
