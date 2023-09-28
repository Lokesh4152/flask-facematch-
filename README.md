# Face Matching Flask Application

This Flask application allows you to match a test image with a database of criminal faces using facial recognition.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)

## Getting Started

Follow these steps to set up and run the application:

1. Clone the repository:

   ```bash
   git clone https://github.com/lokesh4152/face-matching-app.git
Navigate to the project directory:
cd face-matching-app
Install the required dependencies (see Dependencies).

Start the Flask application:

python app.py
The application will be accessible at http://localhost:8000.

Usage
To use the Face Matching API, you can make GET and POST requests to the appropriate endpoints. Here's how you can use the API:

Matching a Face (POST Request)
You can match a test image with a database of criminal faces by sending a POST request to the /api/matchface endpoint. Provide the image URL in the request body as JSON.

Example using curl:

curl -X POST -H "Content-Type: application/json" -d '{"imgUrl": "https://example.com/test-image.jpg"}' http://localhost:8000/api/matchface
Response
The API will respond with a JSON object containing the result of the face matching operation. If a match is found, it will provide information about the matched face and the similarity score. If no match is found, it will indicate that the test image matches none.

**API Endpoints
/api/matchface (GET and POST): Match a test image with a database of criminal faces.
Dependencies
Flask: Web framework for creating the API.
DeepFace: Library for facial recognition and similarity scoring.
pandas: Data manipulation library for handling CSV data.
requests: Library for making HTTP requests.
shutil: Utility functions for file operations.
calendar: Library for working with calendars (not used in the current version).
email.mime: Library for creating email messages (not used in the current version).
cProfile: Library for profiling Python code (not used in the current version).
Make sure to install these dependencies before running the application.



Replace `"https://example.com/test-image.jpg"` with the actual URL of the test image you want to use for matching.
