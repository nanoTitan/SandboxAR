using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System;

#if UNITY_5_3 || UNITY_5_3_OR_NEWER
using UnityEngine.SceneManagement;
#endif
using OpenCVForUnity;

namespace OpenCVForUnitySample
{
	/// <summary>
	/// WebCamTexture detect face sample.
	/// </summary>
	public class LEDDetectorOCV : MonoBehaviour
	{
		[SerializeField] Slider hMinSlider;
		[SerializeField] Slider hMaxSlider;
		[SerializeField] Slider sMinSlider;
		[SerializeField] Slider sMaxSlider;
		[SerializeField] Slider vMinSlider;
		[SerializeField] Slider vMaxSlider;

		public string requestDeviceName = null;
		public int requestWidth = 640;
		public int requestHeight = 480;
		public bool requestIsFrontFacing = false;
		WebCamTexture webCamTexture;
		WebCamDevice webCamDevice;
		Color32[] colors;
		Texture2D texture;
		bool initWaiting = false;
		bool initDone = false;
		Mat rgbaMat;
		Mat blurredMat;
		Mat hsvMat;
		Mat threshold;

		// HSV: 180-230, 0-100, 50-100
		Scalar minHSV = new Scalar(0, 0, 0);
		Scalar maxHSV = new Scalar(360, 100, 100);

		// Use this for initialization
		void Start ()
		{
			init ();

			OnMinValueSlider();
			OnMaxValueSlider();
		}

		/// <summary>
		/// Init of web cam texture.
		/// </summary>
		private void init ()
		{
			blurredMat = new Mat();
			hsvMat = new Mat();
			threshold = new Mat();

			if (initWaiting)
				return;

			StartCoroutine (init_coroutine ());
		}

		/// <summary>
		/// Init of web cam texture.
		/// </summary>
		/// <param name="deviceName">Device name.</param>
		/// <param name="requestWidth">Request width.</param>
		/// <param name="requestHeight">Request height.</param>
		/// <param name="requestIsFrontFacing">If set to <c>true</c> request is front facing.</param>
		/// <param name="OnInited">On inited.</param>
		private void init (string deviceName, int requestWidth, int requestHeight, bool requestIsFrontFacing)
		{
			if (initWaiting)
				return;

			this.requestDeviceName = deviceName;
			this.requestWidth = requestWidth;
			this.requestHeight = requestHeight;
			this.requestIsFrontFacing = requestIsFrontFacing;

			StartCoroutine (init_coroutine ());
		}

		/// <summary>
		/// Init of web cam texture by coroutine.
		/// </summary>
		private IEnumerator init_coroutine ()
		{
			if (initDone)
				dispose ();

			initWaiting = true;

			if (!String.IsNullOrEmpty (requestDeviceName)) {
				//Debug.Log ("deviceName is "+requestDeviceName);
				webCamTexture = new WebCamTexture (requestDeviceName, requestWidth, requestHeight);
			} else {
				//Debug.Log ("deviceName is null");
				// Checks how many and which cameras are available on the device
				for (int cameraIndex = 0; cameraIndex < WebCamTexture.devices.Length; cameraIndex++) {
					if (WebCamTexture.devices [cameraIndex].isFrontFacing == requestIsFrontFacing) {

						//Debug.Log (cameraIndex + " name " + WebCamTexture.devices [cameraIndex].name + " isFrontFacing " + WebCamTexture.devices [cameraIndex].isFrontFacing);
						webCamDevice = WebCamTexture.devices [cameraIndex];
						webCamTexture = new WebCamTexture (webCamDevice.name, requestWidth, requestHeight);

						break;
					}
				}
			}

			if (webCamTexture == null) {
				if (WebCamTexture.devices.Length > 0) {
					webCamDevice = WebCamTexture.devices [0];
					webCamTexture = new WebCamTexture (webCamDevice.name, requestWidth, requestHeight);
				} else {
					webCamTexture = new WebCamTexture (requestWidth, requestHeight);
				}
			}

			// Starts the camera.
			webCamTexture.Play ();

			while (true) {
				// If you want to use webcamTexture.width and webcamTexture.height on iOS, you have to wait until webcamTexture.didUpdateThisFrame == 1, otherwise these two values will be equal to 16. (http://forum.unity3d.com/threads/webcamtexture-and-error-0x0502.123922/).
				#if UNITY_IOS && !UNITY_EDITOR && (UNITY_4_6_3 || UNITY_4_6_4 || UNITY_5_0_0 || UNITY_5_0_1)
				if (webCamTexture.width > 16 && webCamTexture.height > 16) {
				#else
				if (webCamTexture.didUpdateThisFrame) {
				#if UNITY_IOS && !UNITY_EDITOR && UNITY_5_2                                    
				while (webCamTexture.width <= 16) {
				webCamTexture.GetPixels32 ();
				yield return new WaitForEndOfFrame ();
				} 
				#endif
				#endif

					Debug.Log ("name " + webCamTexture.name + " width " + webCamTexture.width + " height " + webCamTexture.height + " fps " + webCamTexture.requestedFPS);
					Debug.Log ("videoRotationAngle " + webCamTexture.videoRotationAngle + " videoVerticallyMirrored " + webCamTexture.videoVerticallyMirrored + " isFrongFacing " + webCamDevice.isFrontFacing);

					initWaiting = false;
					initDone = true;

					onInited ();

					break;
				} else {
					yield return 0;
				}
			}
		}

		/// <summary>
		/// Releases all resource.
		/// </summary>
		private void dispose ()
		{
			initWaiting = false;
			initDone = false;

			if (webCamTexture != null) {
				webCamTexture.Stop ();
				webCamTexture = null;
			}
			if (rgbaMat != null) {
				rgbaMat.Dispose ();
				rgbaMat = null;
			}
		}

		/// <summary>
		/// Init completion handler of the web camera texture.
		/// </summary>
		private void onInited ()
		{
			if (colors == null || colors.Length != webCamTexture.width * webCamTexture.height)
				colors = new Color32[webCamTexture.width * webCamTexture.height];
			if (texture == null || texture.width != webCamTexture.width || texture.height != webCamTexture.height)
				texture = new Texture2D (webCamTexture.width, webCamTexture.height, TextureFormat.RGBA32, false);

			rgbaMat = new Mat (webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);

			gameObject.GetComponent<Renderer> ().material.mainTexture = texture;

			gameObject.transform.localScale = new Vector3 (webCamTexture.width, webCamTexture.height, 1);
			Debug.Log ("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);


			float width = rgbaMat.width ();
			float height = rgbaMat.height ();

			float widthScale = (float)Screen.width / width;
			float heightScale = (float)Screen.height / height;
			if (widthScale < heightScale) {
				Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
			} else {
				Camera.main.orthographicSize = height / 2;
			}
		}

		// Update is called once per frame
		void Update ()
		{
			if (initDone && webCamTexture.isPlaying && webCamTexture.didUpdateThisFrame)
			{
				Utils.webCamTextureToMat (webCamTexture, rgbaMat, colors);

				// Empty Matrix
				Mat test = new Mat(new Size(texture.width, texture.height), CvType.CV_8UC4);
				//Mat dest = new Mat(threshold.rows(), threshold.cols(), threshold.type());

				Imgproc.blur(rgbaMat, blurredMat, new Size(7, 7));
				Imgproc.cvtColor (blurredMat, hsvMat, Imgproc.COLOR_BGR2HSV);

				Core.inRange(hsvMat, minHSV, maxHSV, threshold);

				// Test Screen: hsv color clamped
				Utils.matToTexture2D(threshold, texture, colors);

				// morphological operators
				// dilate with large element, erode with small ones
				Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24, 24));
				Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12));

				Imgproc.erode(threshold, threshold, erodeElement);
				Imgproc.erode(threshold, threshold, erodeElement);

				Imgproc.dilate(threshold, threshold, dilateElement);
				Imgproc.dilate(threshold, threshold, dilateElement);

				List<MatOfPoint> contours = new List<MatOfPoint>();
				Mat hierarchy = new Mat();

				// find contours
				Imgproc.findContours(threshold, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

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
						Imgproc.drawContours(rgbaMat, contours, idx, new Scalar(255, 0, 0, 255)); 
					}

					if(circleFound[0])
						Imgproc.circle(rgbaMat, center1, (int)radius1[0], new Scalar(0, 255, 0, 255), 2);

					if (circleFound[1])
						Imgproc.circle(rgbaMat, center2, (int)radius2[0], new Scalar(0, 0, 255, 255), 2);
				}

				Utils.matToTexture2D(rgbaMat, texture, colors);

				//Utils.matToTexture2D(rgbaMat, texture, webCamTextureToMatHelper.GetBufferColors());
			}
		}

		public void OnMinValueSlider()
		{
			minHSV.val[0] = hMinSlider.value;
			minHSV.val[1] = sMinSlider.value;
			minHSV.val[2] = vMinSlider.value;
		}

		public void OnMaxValueSlider()
		{
			maxHSV.val[0] = hMaxSlider.value;
			maxHSV.val[1] = sMaxSlider.value;
			maxHSV.val[2] = vMaxSlider.value;
		}
	}
}
