
🏢 Employee Attrition Prediction System
A Machine Learning–Powered HR Decision Support App

This project is an end-to-end HR analytics tool designed to help teams predict employee attrition, analyze workforce trends, and generate actionable retention recommendations.
Built using Streamlit, Machine Learning, and interactive visual analytics.

🔗 Live App: Add your Streamlit Cloud link here
📁 Dataset: IBM HR Attrition Dataset (or your version)

🚀 Features
🔮 Single Employee Prediction

Enter employee details and instantly receive:

Attrition probability

Risk classification (Low / Medium / High)

HR recommendations based on the result

Summary of all factors used in the prediction

📂 Batch Prediction

Upload a CSV file to:

Process 100s of employees at once

Automatically identify high-risk employees

Download the prediction results

🎯 Recommendation Engine

Generates tailored HR suggestions using:

Job satisfaction

Work environment

Monthly income

Distance from home

Experience level

Helps HR teams take targeted retention actions.

📊 HR Dashboard

A visual dashboard providing:

Risk distribution charts

Filters for department, age, distance, job satisfaction

Top 10 high-risk employees

Overall workforce insights

🧠 Machine Learning Model

The model is trained on selected features:

Age

Monthly Income

Distance From Home

Job Satisfaction

Environment Satisfaction

Total Working Years

These features were chosen based on:

Correlation analysis

Domain relevance

Predictive performance

The final model is stored in:

final_attrition_model.pkl

🌐 Deployment

The app is deployed using Streamlit Community Cloud.

To deploy your own version:

requirements.txt
app.py
final_attrition_model.pkl


Upload these files to GitHub → Deploy on Streamlit Cloud.

🛠️ Installation (Local Development)
Clone the project:
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

📁 Project Structure
.
├── app.py                     # Main Streamlit application
├── final_attrition_model.pkl  # Trained ML model
├── requirements.txt           # Required Python packages
└── README.md                  # Documentation

✨ Future Improvements

Add authentication (HR login)

Add PDF report export for predictions

Add trend analysis (time-based attrition patterns)

Add SHAP-based explanations for model transparency

Add HR benchmark comparison charts

🙌 Credits

Developed by: Tobi
Powered by Streamlit + Machine Learning
