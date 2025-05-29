# Enhancing Crop Disease Prediction Using Deep Learning

A cross-platform application for detecting crop diseases using YOLOv8 deep learning model.

## Features

- Real-time disease detection in crop plants
- Support for multiple crop types including Apple, Tomato, Potato, and more
- Detailed disease information and treatment recommendations
- Modern and user-friendly interface
- Fast and accurate detection using YOLOv8
- Works on both PC and mobile devices through web browser
- Camera integration for capturing images directly

## Installation

### Option 1: Install from PyPI (Recommended)

1. Make sure you have Python 3.8 or later installed

2. Install the package directly using pip:
   ```bash
   pip install crop-disease-detector
   ```

3. Run the application:
   ```bash
   crop-disease-detector
   ```

### Option 2: Install from Source

1. Clone this repository or download it as a ZIP file

2. Make sure you have Python 3.8 or later installed

3. Navigate to the project directory and install:
   ```bash
   pip install .
   ```

### Sharing the Application

To share this application with others:

1. Direct them to install using pip (Option 1 above)
2. The package includes all necessary files including the model file (best.pt)
3. No additional downloads or setup required

## Usage

### Web Application (Cross-Platform)

1. Start the web application:
   ```bash
   python app.py
   ```

2. The application will launch a web server

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

4. For mobile access, ensure your mobile device is on the same network as your PC, then navigate to your PC's IP address with port 5000:
   ```
   http://YOUR_PC_IP:5000
   ```

5. Use the interface to:
   - Upload images for disease detection
   - Capture images using your device camera
   - View detection results
   - Get detailed information about detected diseases

### Desktop Application

1. Start the desktop application:
   ```bash
   python main.py
   ```

2. The application will launch with a splash screen

3. Use the interface to:
   - Upload images for disease detection
   - View detection results
   - Get detailed information about detected diseases

## System Requirements

- Python 3.8 or later
- Windows/Linux/MacOS
- Minimum 4GB RAM
- NVIDIA GPU (recommended for faster detection)

## Dependencies

- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- Ultralytics >= 8.0.0
- PyQt5 >= 5.15.0
- OpenCV Python >= 4.5.0
- Other dependencies as listed in requirements.txt

## Support

If you encounter any issues or need assistance, please:
1. Check the requirements are correctly installed
2. Ensure the model file is present in the correct location
3. Verify your Python version is compatible

## License

This project is licensed under the MIT License - see the LICENSE file for details