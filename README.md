# Nav GUI 1.0

Nav GUI 1.0 is an interactive graphical user interface application developed for the evaluation and analysis of navigation algorithms, particularly in the field of robotic navigation. This tool facilitates the processing of image and point cloud data, providing features like image loading, feature matching, point cloud generation, and action determination for navigation tasks.

## Features

- **Image Loading**: Browse and load color images from a specified directory based on user-defined IDs.
- **Feature Matching**: Utilize ORB algorithm for extracting and matching features between two images.
- **Point Cloud Generation**: Create 3D point clouds from depth images using camera intrinsic parameters and matched keypoints.
- **Point Cloud Processing and Registration**: Process point clouds with functions like downsampling, outlier removal, and normal estimation. Perform global registration using RANSAC and refinement using ICP for precise alignment.
- **Action Determination**: Determine the action for robotic navigation based on the transformation matrix from point cloud registration.
- **Save Functionality**: Save various forms of point clouds in PLY format.
- **Reset and Clear Data**: Reset the application to its initial state with a single click.
- **Structured GUI**: A user-friendly interface with distinct sections for input, display, and control.

## Prerequisites

Before running Nav GUI 1.0, ensure you have Python installed on your system. The application depends on several Python libraries listed in `requirements.txt`.

## Installation

1. Clone the repository or download the source code.
2. Navigate to the project directory in your terminal or command prompt.
3. Install the required libraries by running: `pip install -r requirements.txt`.

## Usage

To run Nav GUI 1.0:

1. Navigate to the project directory containing `nav_gui.py`.
2. Run the script with Python by executing: `python nav_gui.py`.
3. Use the GUI to browse and load images, match features, generate and process point clouds, and determine navigation actions.

### Demo Images

Demo images are provided in the `demo_images` folder for quick testing. The GUI is ready to be tested using these images. For reliable action output, use frames 10 and 20. Testing with frames 10 and 100 will not yield a reliable action, demonstrating the tool's range of functionality.

## Contributing

Contributions to Nav GUI 1.0 are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## Acknowledgements

Nav GUI 1.0 was created using several open-source libraries. We acknowledge the creators and contributors of these libraries for their invaluable work.
