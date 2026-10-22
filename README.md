# 🌱 AgriBloom — AI-Powered Crop Planning & Resource Optimization

AgriBloom is an AI-powered agricultural planning application designed to help farmers make informed crop-selection decisions using soil properties and weather conditions. It combines machine learning with resource-planning tools to explore how land, water, and budget can be allocated more effectively.

🚀 **Live Demo:https://agribloom-zyoaakcgac4ujgvecjyvf9.streamlit.app/

## ✨ Features

- **🌾 AI Crop Recommendation:** Predicts suitable crops using soil nutrient levels and environmental conditions.
- **🌦️ Weather Integration:** Retrieves weather information using the Open-Meteo service when available.
- **💧 Resource Planning:** Explore land, water, and budget allocation scenarios.
- **📊 Explainable AI:** Displays model feature-importance information to help users understand which inputs influence predictions.
- **🔄 What-If Simulator:** Compare alternative resource and planning scenarios.
- **📅 Crop Calendar:** Provides a general crop-growth-stage overview.
- **📄 Downloadable Reports:** Export planning results for further review.
- **🌿 Interactive Dashboard:** A user-friendly interface built with Streamlit.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Frontend and Dashboard:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib
- **Model Serialization:** Joblib / Pickle
- **Weather Data:** Open-Meteo API
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## 🧠 Machine Learning

AgriBloom uses a trained Random Forest classification model to recommend crops based on agricultural input features.

Typical input features include:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The model was trained using a crop recommendation dataset. Its reported test accuracy was approximately **99.55%** on the evaluated dataset.

**Note:** This score reflects performance on the dataset used for evaluation. It does not guarantee the same accuracy under real-world farming conditions.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sahasra1812/agribloom.git
cd agribloom
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the project structure

```text
agribloom/
├── app.py
├── requirements.txt
├── README.md
└── ml/
    └── crop_model.pkl
```

### 5. Start the application

```bash
streamlit run app.py
```

The application will display a local URL in your terminal, usually `http://localhost:8501`.

## ☁️ Deployment

AgriBloom is deployed using Streamlit Community Cloud.

To deploy your own copy:

1. Fork or copy this repository to your GitHub account.
2. Confirm that `app.py`, `requirements.txt`, and `ml/crop_model.pkl` are present.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Connect your GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the application.

## 📊 Dataset

The project uses a crop recommendation dataset containing soil nutrient values and environmental measurements associated with crop labels.

Dataset features include N, P, K, temperature, humidity, pH, and rainfall.

The model's recommendations depend on the quality, coverage, and representativeness of its training data.

## ⚠️ Limitations

- Recommendations are based on the model's training data and may not generalize to every location or season.
- Feature importance describes model behavior; it does not establish cause and effect.
- Resource-planning outputs are estimates and should not be interpreted as guaranteed savings or profits.
- The crop calendar is a general guide, not a verified location-specific planting schedule.
- Weather information depends on the availability and coverage of the external weather service.

**AgriBloom is a decision-support prototype, not a replacement for local agricultural expertise, soil testing, or region-specific agronomic advice.**

## 🔮 Future Improvements

- Add region-specific crop recommendations.
- Incorporate local soil and historical weather data.
- Improve irrigation and fertilizer planning.
- Add multilingual support for farmers.
- Validate recommendations with agricultural experts and real farm data.
- Evaluate the model on independent datasets and field observations.

## 👩‍💻 Author

**Sahasra**  
GitHub: [@sahasra1812](https://github.com/sahasra1812)



*Built with Python, machine learning, and a vision for smarter, more sustainable agriculture. 🌱*
