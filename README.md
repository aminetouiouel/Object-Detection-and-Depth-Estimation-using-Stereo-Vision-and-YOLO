
# Object Detection and Depth Estimation using Stereo Vision and YOLO

### **Project Overview**

This project implements **object detection and distance estimation** using the **YOLOv8 model** combined with **stereo vision techniques**. It is based on a **Loitor VI stereo camera**, which captures synchronized left and right images.

The system processes these stereo image pairs through several stages:
- **Image rectification** based on feature matching and the fundamental matrix,
- **Disparity map generation** using StereoSGBM,
- **Object detection** using YOLOv8 on rectified images,
- And finally, **depth estimation** through triangulation based on camera calibration parameters.


---
###  Table of Contents

- What is Stereo Vision?
- Camera configuration
- Project Pipeline
- Detailed Pipeline Explanation
- Results Display
- Project Structure
- Getting Started
- References

  


---
###  What is Stereo Vision?

Stereo vision is a crucial field of study within digital image processing and computer vision, focusing on the extraction of depth information from two-dimensional images. The primary technique in stereo vision, binocular stereo vision, mimics the way human eyes perceive depth by capturing two images from slightly different perspectives. By applying the principle of triangulation, stereo vision systems compute the disparities between corresponding points in the left and right images, reconstructing the three-dimensional (3D) structure of a scene.

Stereo vision is widely used in various applications, such as autonomous navigation, 3D reconstruction, and augmented reality, due to its ability to generate high-resolution disparity maps. These maps visualize the differences in pixel alignment between the two images, which directly correspond to depth variations in the scene. Compared to active depth sensors like time-of-flight (ToF) cameras, stereo vision systems (passive sensors) offer greater robustness and accuracy in diverse environments, including outdoor settings.

Advancements in hardware, such as GPUs and FPGAs, have significantly improved real-time processing capabilities for stereo vision applications, making it a continuously evolving and active area of research.



---

### **Camera configuration**

The camera system used in this project is documented in the [official repository](https://github.com/loitor-vis/vi_sensor_sdk.git), which contains all essential resources for understanding and configuring the Loitor stereo vision sensor. This folder includes the Loitor User Manual, providing detailed technical specifications, connection setups, and usage instructions. Additionally, the calibration procedure is thoroughly explained in the included Loitor_Camera_Calibration_Tutorial, which guides users through intrinsic and extrinsic calibration steps, ensuring proper alignment and rectification of stereo image pairs. This calibrated information is used to improve rectification quality and disparity accuracy in the stereo vision pipeline.

---

### **Project Pipeline**

Before the stereo images are processed, the system performs **camera calibration** to extract the intrinsic and extrinsic parameters of the stereo camera (focal length, principal point,  etc.). These parameters are crucial for accurate **image rectification** and **depth estimation**.

Once calibration is done, each stereo image pair passes through the following pipeline:

1. **Image Rectification**: Aligns both images so that corresponding points lie on the same horizontal line (epipolar geometry).
2. **YOLOv8 Object Detection**: Detects and localizes objects in both rectified views.
3. **Matching Cost Computation**: Calculates pixel-wise matching differences between the left and right images.
4. **Cost Aggregation**: Refines the matching costs using local window-based smoothing to improve robustness.
5. **Disparity Selection**: Chooses the most likely disparity value for each pixel.
6. **Disparity Map Generation**: Builds a full disparity map, which is then used to estimate the **real-world distance** to each detected object through **triangulation**.

This pipeline enables the system to transform raw stereo images into a 3D scene understanding with object categories and distances.

![image](https://github.com/user-attachments/assets/4a18c6ed-d31d-49d9-a23d-7cf4d889d2c3)


---

### **Detailed Pipeline Explanation**

## **1. Camera Calibration**

- **Purpose**: Camera calibration helps in determining the intrinsic and extrinsic parameters of the cameras, ensuring correct interpretation of the images.
  
- **Process**:
  
  - Use a **chessboard pattern** to capture multiple images and calculate the camera’s intrinsic and extrinsic parameters.
  - The intrinsic parameters include the **focal length**, **principal point**, and **distortion coefficients**.
  - The extrinsic parameters define the **relative position** and **orientation** between the two stereo cameras.
  - Calibration can be done using **OpenCV** tools to get the necessary parameters for rectification and disparity calculation.
-**implementation**

-we used matlab for the calibration part of course there are several other methods to do that like python and opencv (a gentleman did a good job at explaining that, here a [link](https://github.com/TemugeB/python_stereo_camera_calibrate?tab=readme-ov-file) to his work),you can also use camera calibration tool with Ros .The official documentation [link](http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration).
- 8x6 corner checkerboard, the default cell size is 30mm.
  
![mac_calib_pattern](https://github.com/user-attachments/assets/86fa487a-969a-4a49-8a94-7f2e8788832a)
[pattern.pdf](https://github.com/user-attachments/files/19577050/pattern.pdf)

 You can print this board to A4 paper in the size of "1:1".

for our case  we use the 15-inch MacBook Pro screen as the calibration board, for in the Mac OS Preview tool,
 you can choose actual size to display PDFs. This allows the checkerboard cell size to render an exact 30mm
 and screen display also ensures absolute flatness of the calibration board.
 
If you're using a 15-inch MacBook Pro，Please use the "mac_calib_pattern. png" screen  and
 click "Actual Size."
 

  
## **2. Image Rectification**

In stereo vision, image rectification is the process of transforming a pair of stereo images so that the epipolar lines (i.e., lines along which matching points must lie) become horizontal and aligned. This simplifies the correspondence search for disparity computation: instead of scanning in 2D, the algorithm only needs to look along a single row (1D search), making the process faster and more robust.
Before Rectification: Matching points between left and right images lie along epipolar lines which can be slanted or curved, depending on the relative pose of the cameras.
After Rectification: The epipolar lines are perfectly horizontal, and any corresponding point in the left image lies on the same row in the right image.


  <p align="center">
  <img src="https://github.com/user-attachments/assets/757397c7-a4f1-4fb2-ab78-40b907565f0e" width="500" height="500" />
</p>
In this project, rectification is performed using uncalibrated stereo vision techniques:

  . SIFT is used to extract and match keypoints between left and right images.

  . The fundamental matrix is estimated via RANSAC to capture epipolar geometry.

  . Rectifying homographies are computed using cv2.stereoRectifyUncalibrated and applied with warpPerspective to align the views.

## **3. Cost Computation**
 

Cost matching is a crucial step in stereo vision, as it determines how well corresponding pixels from the left and right images match. This process is essential for **disparity estimation**, which directly impacts the accuracy of depth computation. Matching costs are calculated by comparing pixel intensities or features between the two images. Simpler methods assume that intensity values remain constant, while more advanced techniques account for radiometric variations and noise. Cost matching functions are broadly classified into **pixel-based** and **window-based** methods. **Pixel-based** techniques analyze individual pixel differences, while **window-based** methods aggregate pixel information within a defined region, improving robustness in textureless or noisy areas. However, the choice of a cost function significantly affects the balance between **accuracy, speed, and computational complexity**.  

### **Common Matching Cost Computation Algorithms**  

   1. **Sum of Absolute Differences (SAD)** : is one of the simplest and most commonly used cost matching functions in stereo vision. It computes the absolute intensity difference between corresponding pixels within a defined window, making it an efficient and fast method suitable for real-time applications. However, it is sensitive to lighting variations and textureless regions, which can affect accuracy in challenging environments.

   2. **Sum of Squared Differences (SSD)**  : is another widely used method that computes the squared intensity difference between pixels. This approach emphasizes larger intensity differences more than SAD, making it more accurate in handling noise. However, its computational cost is higher, resulting in slower performance compared to SAD, making it less ideal for real-time applications where speed is a priority.

  3. **Maximum of Absolute Differences (MAD)** : method determines disparity by selecting the maximum absolute intensity difference within a given region. This makes it computationally more efficient than both SAD and SSD, allowing for faster processing speeds. However, it is also the least accurate among the three methods, as it does not consider all pixel differences comprehensively.

  4. **Normalized Cross-Correlation (NCC)** : method is designed to compensate for variations in brightness and contrast between images, making it more robust to lighting changes than the previously mentioned methods. However, NCC is computationally expensive, requiring significant processing power, which makes it unsuitable for real-time stereo vision applications.

  5. **Rank and Census Transforms** : offer a more advanced approach by converting pixel intensities into a structured ranking format for comparison. This transformation makes them highly effective in handling noise and illumination changes, improving accuracy in difficult imaging conditions. However, their computational complexity is significantly higher, making them more demanding and less practical for real-time stereo vision systems.


For this project, we use StereoSGBM (Semi-Global Block Matching), which builds upon the **Sum of Absolute Differences (SAD)** as its core cost function. StereoSGBM enhances SAD by incorporating a global energy minimization strategy that enforces disparity smoothness across neighboring pixels, significantly improving accuracy in low-texture regions and depth discontinuities.

This approach offers a strong balance between **speed and depth accuracy**, making it well-suited for real-time or near real-time applications such as object detection and distance estimation. Although SAD-based methods are sensitive to lighting variations, StereoSGBM reduces this sensitivity by aggregating costs along multiple paths and supporting block-level comparison. In addition, the **depth map is normalized** to ensure consistent disparity scaling, which improves both visualization and distance estimation. The combination of SAD’s simplicity and SGBM’s optimization enables robust and efficient stereo matching for embedded and computationally constrained systems.


## **4. Cost Aggregation**


**Cost aggregation** is a crucial step in **stereo vision disparity map algorithms**, especially for **local methods**. The primary purpose of cost aggregation is to **reduce matching uncertainties** by refining the raw matching cost computed at each pixel. Since the initial cost matching at an individual pixel may not be reliable, cost aggregation **smooths and stabilizes disparity estimations** by considering surrounding pixels within a **support region**. This support region is typically defined by a **square or adaptive window** centered on the pixel of interest. Various **aggregation techniques** exist to enhance depth estimation accuracy while balancing computational efficiency.  

### **Common Cost Aggregation Algorithms :**  

  
  1.  **The Multiple Window (MW) Aggregation** method enhances disparity map accuracy by selecting the optimal window size from multiple candidates. Instead of using a single fixed window, it evaluates several window sizes and chooses the one that yields the lowest matching cost. This approach helps adapt to different regions in the image, improving depth estimation. However, the **computational complexity** increases significantly because multiple window comparisons must be performed for each pixel, making it less efficient for real-time applications.  

  2. **The Adaptive Window (AW) Aggregation** technique dynamically adjusts the shape and size of the support window based on the local image structure. Unlike fixed-window approaches, AW adapts to image features such as edges and object boundaries, reducing errors near depth discontinuities. This flexibility helps **preserve object boundaries** and improves disparity map quality. However, the method is **more complex** than fixed-window techniques, requiring additional processing time, which can make real-time implementations challenging.  

  3. **The Adaptive Support Weights (ASW) Aggregation** method refines cost aggregation by assigning different weights to pixels based on their **intensity similarity and spatial proximity**. This technique ensures that pixels with similar colors and closer distances to the reference pixel contribute more to the disparity calculation, effectively reducing noise and preserving object boundaries. As a result, ASW produces **highly accurate disparity maps** compared to other methods. However, the increased computational cost makes it more demanding, though modern GPU optimizations can improve its efficiency for real-time applications. 
 

In this project, we implement a Fixed Window Box Filter approach for cost aggregation. This method uniformly averages the disparity values within a square window  centered on each pixel. It helps smooth out local noise and stabilize disparity estimates by treating all neighboring pixels equally. Although this technique may reduce precision near object edges due to its lack of adaptive weighting, it offers a good trade-off between speed and accuracy, making it well-suited for real-time object detection and depth estimation in our stereo vision system.

## **5. Disparity Selection**


**Disparity selection** is a critical stage in stereo vision, as it determines the best-matching pixel offset (disparity) between the left and right images. This disparity is directly used to estimate the depth of objects in the scene. In local stereo methods, the **Winner-Takes-All (WTA)** strategy is commonly used, where for each pixel, the disparity with the lowest aggregated cost is chosen. This approach is **computationally efficient** and suitable for **real-time systems**, but may yield **inaccurate results** in regions with poor texture or occlusions due to the limited local context.

In contrast, **global methods** model disparity selection as an **energy minimization problem**, optimizing both data fidelity and spatial smoothness. Algorithms like **graph cuts**, **belief propagation**, or **dynamic programming** fall into this category. Although they provide **high-accuracy** disparity maps, their **high computational complexity** makes them less practical for real-time applications.

For this project, disparity selection is focused on the **center of each object** detected by **YOLOv8**. Around this center, a **5×5 pixel window** is extracted from the disparity map, and the **median disparity** within this patch is computed. This disparity is then used to calculate the distance to the object using **triangulation**.  
This approach provides **robustness against noise and outliers** while maintaining **real-time performance** and **low computational cost**, making it well-suited for **embedded or resource-constrained vision systems**.




---

##  Results Display

Once the system is running, several output windows will appear to visualize the stereo vision pipeline results:

- **Original Left Image with Distance**  
  Displays the original left image with YOLOv8 object detections and estimated distances (in meters) overlaid for each detected object.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b1865c7b-4af4-4c70-a1ba-d830279ddd64" width="300" height="300" />
</p>



- **Rectified Left Image**  
  Shows the rectified version of the left stereo image to confirm alignment.
  <p align="center">
  <img src="https://github.com/user-attachments/assets/e7031caa-8f03-4bc6-bc62-3831316eea2d" width="300" height="300" />
</p>


- **Rectified Right Image**  
  Displays the rectified right image.
  <p align="center">
  <img src="https://github.com/user-attachments/assets/c5e257a6-c41d-4985-b7cc-d8ba437ef29a" width="300" height="300" />
</p>
  


- **Disparity Map**  
  A grayscale image visualizing pixel disparities. Brighter areas indicate closer objects.
    <p align="center">
  <img src="https://github.com/user-attachments/assets/a6bc0b5a-31f1-4270-97d1-7c920ddbc88b" width="300" height="300" />
</p>


- **Depth Map**  
  Visualizes the calculated depth (inverse of disparity), normalized for better interpretation.
      <p align="center">
  <img src="https://github.com/user-attachments/assets/8fc8d81d-0e55-4b0e-b04e-6d79fb3ea483" width="300" height="300" />
</p>


Each object detected by YOLOv8 is annotated with its estimated 3D distance, computed using the median disparity within a small region around the object’s center. This approach balances computational efficiency with accuracy, validating the effectiveness of combining deep learning-based detection with stereo disparity for real-time distance estimation.

To further increase accuracy, the disparity can be sampled from multiple key regions within the bounding box (e.g., top, bottom, left, right edges in addition to the center). Depending on your dataset and application, this sampling strategy can be adjusted through empirical testing to better handle occlusions, object size variation, or noise in the disparity map.

For optimal results, it's also possible to apply post-processing filters such as median, bilateral, or AI-based refinement filters to reduce noise and improve disparity quality. Additionally, using a high-quality stereo camera with proper calibration greatly enhances depth accuracy and robustness, especially in challenging environments.


All windows are displayed simultaneously using OpenCV. Press any key to close them after inspection.


---

### **Project Structure**

```plaintext
StereoVisionProject/
├── ACQUISITION/ # Stereo image dataset
│ ├── image_L/ # Left camera images
│ └── image_R/ # Right camera images
│
├── rectified/ # Output folder for rectified stereo images
│ ├── rectified_left.png
│ └── rectified_right.png
│
├── yolo.py # YOLOv8 object detection module
├── Stereo.py # Stereo vision (disparity & depth computation)
├── rectification.py # Stereo rectification functions
├── main.py # Main pipeline script (from rectification to detection)
│
├── README.md # Project overview and usage guide
└── requirements.txt
```


### **Getting Started**

1. Clone the repository:
    ```bash
    git clone https://github.com/aminetouiouel/Object_Detection_and_Depth_Estimation_using_Stereo_Vision.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
3. Prepare Your Data :
```plaintext   
├── ACQUISITION/ # Stereo image dataset
│ ├── image_L/ # Left camera images
│ └── image_R/ # Right camera images
```

4. Run the project:
    ```bash
    python3 main.py
    ```


##  References & Resources

If you're interested in learning more about stereo vision, here are some useful resources:

-  **10-16-385 Computer Vision** — *Kris Kitani (CMU)*  
  A comprehensive academic course introducing core computer vision concepts including stereo vision.

-  [**Tutorial on Rectification of Stereo Images**](https://www.researchgate.net/publication/2841773_Tutorial_on_Rectification_of_Stereo_Images)  
  A foundational tutorial covering the geometric principles and implementation of stereo image rectification.

-  **Distance Measurement System Based on Binocular Stereo Vision**  
  *IOP Conference Series, 2019*  
  Presents a binocular vision system using MATLAB calibration and OpenCV matching for 3D measurement.

-  **Computer Vision Based Distance Measurement Using Stereo Camera**  
  *IEEE, 2019*  
  Describes a real-time stereo vision system using disparity maps to estimate object distance.

-  **Three-Dimensional Computer Vision: A Geometric Viewpoint – Olivier Faugeras**  
  A seminal book providing rigorous mathematical foundations for 3D vision.

-  **Stereo Vision: Algorithms and Applications – Stefano Mattoccia**  
  Covers stereo matching algorithms and their applications in real-world scenarios.

-  **Journal of Sensors (2016) – Literature Survey on Stereo Vision Disparity Map Algorithms**  
  A detailed survey of classical and modern approaches to disparity estimation.

-  **Real-Time Mobile Stereo Vision**  
  Discusses implementation of stereo vision on embedded systems for robotics.

-  **Comparison of Stereo Matching Algorithms for Disparity Map Development**  
  Benchmarks and evaluates the performance of multiple stereo algorithms.

-  **Evaluation of Stereo Image Matching**  
  Discusses evaluation metrics and test sets for stereo correspondence accuracy.

-  **Pattern Recognition for Image Processing – Pertuz et al., 2013**  
  Insights into image matching, pattern recognition, and stereo vision strategies.

-  **Diatom Autofocusing in Brightfield Microscopy**  
  While focused on microscopy, it explores focus measures applicable in stereo vision.

-  **Rectification and Epipolar Geometry – Supplementary Notes**  
  Explains concepts like the fundamental matrix and rectification transforms.

-  **Stereo Matching Important Notes**  
  Summarizes the stereo pipeline, cost aggregation, and algorithm types.

- [**Awesome Computer Vision – curated by jbhuang0604**](https://github.com/jbhuang0604/awesome-computer-vision?tab=readme-ov-file#books)  
  A comprehensive and curated list of resources, books, datasets, and papers related to computer vision, including stereo vision and 3D reconstruction.
  
## Credits

This project was created by [Mohamed Amine Touiouel](https://github.com/aminetouiouel) and [Bilal Marghich](https://github.com/BlueBerry1589).
