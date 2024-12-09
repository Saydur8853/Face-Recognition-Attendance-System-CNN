## Installation
------------------------------------

### Prerequisites

Ensure you have the following installed on your system:

-   Python 3.8 or higher
-   pip (Python package manager)
-   A webcam for capturing images

### Setup Instructions

1.  **Clone the repository**

    `git clone <repository_url>
    cd FaceRecognitionAttendance`

2.  **Set up a virtual environment**
    
    `python -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    venv\Scripts\activate     # For Windows`

3.  **Install dependencies**

    `pip install -r requirements.txt`

4.  **Run the project**
    1. 1st run capture_faces.py for Capture Faces
    2. 2nd run train_model.py for train the picture by CNN model
    3. 3rd run attendance_final.py for getting attendance.