# 🥗 Nutrition Optimizer: Beyond Calories

> A data-driven nutrition optimization system that goes beyond calorie counting to help users make smarter food choices based on nutrient density, clustering, and personalized meal planning.

![Streamlit App Badge](https://img.shields.io/badge/Streamlit-1.29.0-blue?logo=streamlit )
![Python Version](https://img.shields.io/badge/Python-3.8+-green?logo=python )

---

## 🌟 Overview

This project introduces a **Nutritional Efficiency Framework** using **data science and machine learning** to:

- Analyze nutritional datasets
- Detect and clean unreliable entries
- Rank foods by efficiency using a custom **Nutritional Efficiency Index (NEI)**
- Discover hidden nutritional patterns via **clustering**
- Recommend optimized food combinations tailored to dietary goals

The goal is to empower users with actionable insights into their food choices — not just how many calories they consume, but **how much nutritional value those calories deliver**.

---
### Interface

![Screenshot 2025-05-09 214014](https://github.com/user-attachments/assets/6f338acb-777d-4277-b08d-6bf57ec0b4e1)

![Screenshot 2025-05-09 214315](https://github.com/user-attachments/assets/51eb7cd5-8a16-4d6f-bae9-65ffad7ccb76)

![Screenshot 2025-05-09 214742](https://github.com/user-attachments/assets/143381c9-a29c-454f-85d1-bb618704b85e)

## 🔍 Key Features

| Feature | Description |
|--------|-------------|
| 🧪 Data Quality Assessment | Flags outliers and inconsistencies in nutritional values |
| 🏆 Nutritional Efficiency Index (NEI) | Ranks foods based on nutrient density per calorie |
| 🧠 Clustering & PCA Visualization | Groups foods into meaningful clusters based on nutrient content |
| 🥗 Food Combination Optimizer | Recommends meals that match specific health goals |
| 📊 Interactive Dashboard | Built with Streamlit for user-friendly exploration |

---

## 🎯 Supported Goals

| Goal | Focus |
|------|-------|
| 💪 Weight Loss | High protein, high fiber, low fat/carbs |
| 🏋️ Muscle Building | High protein, carbs, iron |
| ❤️ Heart Health | High fiber, moderate protein, low sodium/fat |
| 🧬 Balanced Diet | Balanced macronutrients and micronutrients |

---

## 🛠️ Technologies Used

- **Python**: Core language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Clustering (`KMeans`), Standardization (`StandardScaler`)
- **Matplotlib & Seaborn**: Visualizations
- **Streamlit**: Web interface for interactive dashboard
- **Jupyter Notebook**: For exploratory analysis and model development

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/saquib5005/nutrition-optimizer.git 
cd nutrition-optimizer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

📁 File Structure

```
nutrition-optimizer/
│
├── app.py                # Streamlit web interface
├── cleaned_nutrition_dataset.csv
├── requirements.txt      # Python dependencies
└── README.md             # You are here!
```

---

### 📜 License
This project is licensed under the MIT License. See [Licence](LICENSE) for more details.

---

### 💌 Support
If you like this project, give it a ⭐ and share it with others who might benefit from smart nutrition decisions!
