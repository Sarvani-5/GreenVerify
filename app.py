from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import google.generativeai as genai
from datetime import datetime

app = Flask(__name__)

# Configure Google Gemini AI
GEMINI_API_KEY = "AIzaSyA49YQ1ZwIGQp0tMdaouunx9A04F9Qn2O0"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_available = True
except Exception as e:
    print(f"Warning: Gemini AI not available: {e}")
    gemini_available = False

# ---------------------------
# Load Existing Model Files
# ---------------------------
def load_trained_model():
    """
    Loads pre-trained model files from the models directory.
    """
    try:
        # Check if all required files exist
        required_files = [
            'models/xgboost_green_certified_model.pkl',
            'models/label_encoders.pkl', 
            'models/feature_names.pkl',
            'models/scaler.pkl',
            'models/reverse_mapping.pkl'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                return None, None, None, None, f"Required file missing: {file_path}"
        
        # Load all components
        model = joblib.load('models/xgboost_green_certified_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        scaler = joblib.load('models/scaler.pkl')
        reverse_mapping = joblib.load('models/reverse_mapping.pkl')
        
        return model, feature_names, label_encoders, scaler, reverse_mapping
        
    except Exception as e:
        return None, None, None, None, f"Error loading model: {str(e)}"

# Load pre-trained model on startup
model, feature_names, label_encoders, scaler, reverse_mapping = load_trained_model()

# Set status message
if model is not None:
    train_status = "✅ Pre-trained model loaded successfully!"
else:
    train_status = "❌ Could not load pre-trained model files."

@app.route('/')
def index():
    return render_template('index.html', 
                         feature_names=feature_names, 
                         label_encoders=label_encoders,
                         train_status=train_status,
                         model=model)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not feature_names:
            return jsonify({'error': 'Model not available'})
        
        # Get form data
        inputs = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if feature in label_encoders:
                inputs[feature] = value
            else:
                inputs[feature] = float(value) if value else 0.0
        
        # Create input dataframe
        input_data = {feature: [inputs[feature]] for feature in feature_names}
        input_df = pd.DataFrame(input_data)
        
        # Check if all values are zero (not certified)
        if input_df.replace(0, np.nan).dropna(axis=1, how='all').empty:
            return jsonify({
                'warning': True,
                'message': 'This building is not certified.'
            })
        
        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:
                    input_df[col] = 0
        
        # Scale numeric features
        num_features_to_scale = [f for f in feature_names if f not in label_encoders]
        if num_features_to_scale:
            input_df[num_features_to_scale] = scaler.transform(input_df[num_features_to_scale])
        
        # Make prediction
        prediction_probs = model.predict_proba(input_df)[0]
        prediction_idx = model.predict(input_df)[0]
        prediction_label = reverse_mapping[prediction_idx]
        
        # Prepare probabilities for response - FIXED: Convert to native Python types
        probabilities = []
        for idx, prob in enumerate(prediction_probs):
            label = reverse_mapping[idx]
            probabilities.append({
                'label': int(label),  # Convert to native Python int
                'probability': float(prob)  # Convert to native Python float
            })
        
        return jsonify({
            'success': True,
            'prediction': int(prediction_label),  # Convert to native Python int
            'probabilities': probabilities,
            'confidence': float(prediction_probs[prediction_idx])  # Convert to native Python float
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

def get_greenybot_recommendations(user_inputs, prediction_rating, probabilities):
    """
    Generate personalized recommendations using Google Gemini AI
    """
    if not gemini_available:
        return "GreenyBot is currently unavailable. Please try again later."
    
    try:
        # Prepare building details for Gemini
        building_details = []
        for feature, value in user_inputs.items():
            clean_feature = feature.replace('_', ' ').title()
            building_details.append(f"- {clean_feature}: {value}")
        
        building_info = "\n".join(building_details)
        
        # Create comprehensive prompt for GreenyBot
        prompt = f"""
You are GreenyBot, an expert AI assistant specializing in GRIHA (Green Rating for Integrated Habitat Assessment) green building certification in India. You provide personalized, actionable recommendations for improving building sustainability.

BUILDING ASSESSMENT RESULTS:
- Predicted GRIHA Rating: {prediction_rating} Stars (out of 5)
- Building Details:
{building_info}

GRIHA RATING CONTEXT:
- 1 Star: Basic compliance with minimal green features
- 2 Stars: Good performance with some green initiatives
- 3 Stars: Very good performance with multiple sustainability measures
- 4 Stars: Excellent performance with comprehensive green features
- 5 Stars: Outstanding performance, benchmark for sustainable buildings

YOUR TASK AS GREENYBOT:
Please provide a comprehensive assessment in the following format:

🤖 **GreenyBot Analysis**

**Why This Rating?**
Explain why the building received this specific star rating based on the input parameters. Reference specific GRIHA criteria and requirements.

**Key Strengths** ✅
List 2-3 positive aspects of the current building design/features.

**Areas for Improvement** 🔧
Provide 3-4 specific, actionable recommendations to improve the GRIHA rating, focusing on:
- Energy efficiency measures
- Water conservation strategies  
- Sustainable materials and resources
- Indoor environmental quality improvements
- Innovation in design processes

**Benefits of Improvement** 🌟
Explain the benefits of implementing these improvements:
- Environmental impact reduction
- Cost savings potential
- Health and comfort improvements
- Certification advantages

**Next Steps** 📋
Provide 2-3 immediate actionable steps the user can take.

Keep your response informative, practical, and encouraging. Use official GRIHA guidelines and current Indian green building practices as reference. Be specific about Indian building standards and climate considerations.
"""

        # Generate response using Gemini
        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"GreenyBot encountered an error: {str(e)}. Please try again later."

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    Get GreenyBot recommendations based on prediction results
    """
    try:
        data = request.get_json()
        user_inputs = data.get('inputs', {})
        prediction_rating = data.get('prediction', 0)
        probabilities = data.get('probabilities', [])
        
        recommendations = get_greenybot_recommendations(user_inputs, prediction_rating, probabilities)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'GreenyBot error: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)