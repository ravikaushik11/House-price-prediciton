# House Price Prediction

This project is a machine learning-based web application that predicts house rental prices in Bangalore based on user inputs. It includes data preprocessing, model training, and deployment using a Flask-based API for making predictions. The frontend is designed to be responsive and visually appealing with a real estate-themed background.

## Features
- Predicts annual rental prices for houses in Bangalore.
- Trained machine learning model using linear regression.
- Flask API to handle requests and return predictions.
- Responsive frontend with a user-friendly interface.
- Background image for enhanced aesthetics.
- model evaluation using RMSE, MAE, and R² scores.
- Model versioning and logging for tracking improvements.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- Flask
- NumPy
- Pandas
- Scikit-learn
- Joblib
- HTML/CSS for frontend

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/ravikaushik11/House-price-prediciton.git
cd house-price-prediction
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Flask Server
```bash
python app.py
```
The server will run on `http://127.0.0.1:5000/`.

### 5. Access the Web App
Open a browser and visit:
```
http://127.0.0.1:5000/
```

## Usage
1. Enter the location, square feet, number of bathrooms, and BHK.
2. Click on `Predict Price`.
3. The predicted rental price will be displayed.

## Notes
- Ensure the `background.jpg` file is in the correct location for the background image to display properly.
- The model is trained with specific features, so input formatting must be consistent.

## Deployment
To deploy on a cloud service (AWS, GCP, Render):
1. Containerize the app using Docker.
2. Use a cloud service to host the API and serve predictions.

## Contributions
Feel free to submit pull requests or report issues.

## Hosting
This is the deployment link "https://house-price-prediciton.onrender.com"
I have used the free version of render not sure it will work later or not, i just wanted to make show live but can't believe on the free render serivce as it might stop in some time.
