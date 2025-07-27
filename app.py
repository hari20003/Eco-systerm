import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
from flask import Flask, request, render_template
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

def load_and_preprocess():
    data = pd.read_csv("forestfires.csv")
    
    # Create binary target and remove unnecessary columns
    data['fire_occurred'] = np.where(data['area'] > 0, 1, 0)
    data = data.drop(['area', 'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI'], axis=1)
    
    # Encode categorical features
    le_month = LabelEncoder()
    le_day = LabelEncoder()
    data['month'] = le_month.fit_transform(data['month'])
    data['day'] = le_day.fit_transform(data['day'])
    
    return data, le_month, le_day

def train_model():
    data, le_month, le_day = load_and_preprocess()
    
    X = data.drop('fire_occurred', axis=1)
    y = data['fire_occurred']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_features = ['temp', 'RH', 'wind', 'rain']
    categorical_features = ['month', 'day']
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            max_depth=5))])
    
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'forest_fire_model.pkl')
    joblib.dump(le_month, 'month_encoder.pkl')
    joblib.dump(le_day, 'day_encoder.pkl')
    
    return model, le_month, le_day

# Load or train model
if not os.path.exists('forest_fire_model.pkl'):
    model, le_month, le_day = train_model()
else:
    model = joblib.load('forest_fire_model.pkl')
    le_month = joblib.load('month_encoder.pkl')
    le_day = joblib.load('day_encoder.pkl')

def predict_fire_probability(input_data):
    input_df = pd.DataFrame([input_data])
    input_df['month'] = le_month.transform([input_data['month']])[0]
    input_df['day'] = le_day.transform([input_data['day']])[0]
    cols = ['temp', 'RH', 'wind', 'rain', 'month', 'day']
    input_df = input_df[cols]
    return model.predict_proba(input_df)[0][1]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            user_input = {
                'month': request.form['month'],
                'day': request.form['day'],
                'temp': float(request.form['temp']),
                'RH': int(request.form['RH']),
                'wind': float(request.form['wind']),
                'rain': float(request.form['rain'])
            }
            
            probability = predict_fire_probability(user_input)
            probability_percent = round(probability * 100, 2)
            
            if probability > 0.7:
                warning = "🔥 HIGH FIRE DANGER!"
                alert_class = "danger"
            elif probability > 0.4:
                warning = "⚠️ Moderate fire risk"
                alert_class = "warning"
            else:
                warning = "✅ Low fire risk"
                alert_class = "success"
            
            return render_template('result.html', 
                                probability=probability_percent,
                                warning=warning,
                                alert_class=alert_class,
                                input_data=user_input)
        except Exception as e:
            return f"Error: {str(e)}", 500
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)