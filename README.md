# Waste Management and Recycling Prediction API

A machine learning API that predicts recycling rates for Indian cities based on waste management parameters.

## 🚀 Features

- Predict recycling rates for single or multiple waste management records
- RESTful API built with FastAPI
- Automated data preprocessing and feature engineering
- Gradient Boosting model with 97.3% R² score
- Docker containerization
- Ready for deployment on Render/AWS

## 📊 Model Performance

- **Test RMSE**: 2.71
- **Test R²**: 0.973
- **Cross-Validation RMSE**: 2.94

## 🏃‍♂️ Quick Start

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `uvicorn src.app:app --reload`
4. Visit: http://localhost:8000/docs

### Using Docker

```bash
docker build -t waste-management-api .
docker run -p 8000:8000 waste-management-api

📚 API Endpoints
GET / - API information

GET /health - Health check

POST /predict - Single prediction

POST /predict/batch - Batch predictions

GET /model/info - Model information

🚀 Deployment
Deploy to Render
Push code to GitHub

Connect repository to Render

Render will automatically deploy from render.yaml

Environment Variables
No environment variables required for basic operation.

🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

text

## Step 8: Test the API Locally

Test your API before deployment:

```bash
# Start the API locally
uvicorn src.app:app --reload

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint (using the example from the API docs)