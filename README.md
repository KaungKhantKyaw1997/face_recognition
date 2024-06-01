# Face Recognition

This project utilizes face recognition to identify characters from "Stranger Things" in images, operating within the realm of Computer Vision.

## Setup Instructions

1. **Clone the repository**:

   ```sh
   git clone https://github.com/KaungKhantKyaw1997/face_recognition.git
   cd face_recognition
   ```

2. **Create a virtual environment**:

   ```sh
   python3 -m venv env
   ```

3. **Activate the virtual environment**:

   - On macOS and Linux:
     ```sh
     source env/bin/activate
     ```
   - On Windows:
     ```sh
     .\env\Scripts\activate
     ```

4. **Install the required packages**:

   ```sh
   pip install face-recognition
   pip install setuptools
   pip install opencv-python
   ```

5. **Run the script**:
   ```sh
   python main.py
   ```

## Troubleshooting

If you encounter issues with missing packages or modules, ensure that all dependencies are installed correctly within the virtual environment.

- **ModuleNotFoundError: No module named 'face_recognition'**:
  Make sure you have run `pip install face-recognition` inside the activated virtual environment.

- **ModuleNotFoundError: No module named 'cv2'**:
  Make sure you have run `pip install opencv-python` inside the activated virtual environment.

## Additional Notes

- It's recommended to use a virtual environment to avoid conflicts with system-wide packages.
- Ensure your Python version is compatible with the packages used in this project.
