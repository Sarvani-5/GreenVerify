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
GEMINI_API_KEY = "AIzaSyA49YQ1ZwIGQp0tMdaouunx9A04F9Qn2O0"  # Move to environment variable in production
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model with correct model name
try:
    # Try different model names that are commonly available
    model_names = [
        'gemini-2.5-flash',
    ]
    
    gemini_model = None
    gemini_available = False
    
    for model_name in model_names:
        try:
            gemini_model = genai.GenerativeModel(model_name)
            # Test the model with a simple query
            test_response = gemini_model.generate_content("Test")
            gemini_available = True
            print(f"Successfully initialized Gemini model: {model_name}")
            break
        except Exception as model_error:
            print(f"Failed to initialize {model_name}: {model_error}")
            continue
    
    if not gemini_available:
        print("Warning: No Gemini models available")
        
except Exception as e:
    print(f"Warning: Gemini AI initialization failed: {e}")
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
    train_status = "Pre-trained model loaded successfully!"
else:
    train_status = "Could not load pre-trained model files."

# Store user session data (in production, use proper session management)
user_sessions = {}

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
                # Convert to float and ensure non-negative
                num_value = float(value) if value else 0.0
                inputs[feature] = max(0.0, num_value)  # Prevent negative values
        
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
        
        # Store session data
        session_id = str(hash(str(inputs)))
        user_sessions[session_id] = {
            'inputs': inputs,
            'prediction': prediction_label,
            'probabilities': prediction_probs
        }
        
        # Prepare probabilities for response
        probabilities = []
        for idx, prob in enumerate(prediction_probs):
            label = reverse_mapping[idx]
            probabilities.append({
                'label': int(label),
                'probability': float(prob)
            })
        
        return jsonify({
            'success': True,
            'prediction': int(prediction_label),
            'probabilities': probabilities,
            'confidence': float(prediction_probs[prediction_idx]),
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

def get_fallback_assessment(user_inputs, prediction_rating):
    """
    Provide fallback assessment when Gemini is not available
    """
    rating_explanations = {
        1: """This building received a 1-star GRIHA rating, indicating basic compliance with minimal green features. 
             The building meets fundamental requirements but has significant room for improvement in sustainability measures.
             Key areas likely lacking include energy efficiency systems, water conservation measures, and sustainable material usage.""",
        
        2: """This building achieved a 2-star GRIHA rating, showing good performance with some green initiatives implemented.
             While better than basic compliance, there are still substantial opportunities to enhance sustainability features
             such as improved energy systems, better water management, and enhanced indoor environmental quality.""",
        
        3: """This building earned a 3-star GRIHA rating, demonstrating very good performance with multiple sustainability measures.
             The building shows commitment to green practices with adequate energy efficiency, water conservation, and
             sustainable design elements, though improvements are still possible.""",
        
        4: """This building achieved a 4-star GRIHA rating, indicating excellent performance with comprehensive green features.
             The building demonstrates strong sustainability practices across energy, water, materials, and indoor environmental
             quality, with only minor enhancements needed to reach the highest rating.""",
        
        5: """This building earned the prestigious 5-star GRIHA rating, representing outstanding performance and serving as
             a benchmark for sustainable buildings. The building excels in all sustainability criteria including energy
             efficiency, water conservation, sustainable materials, and innovative design processes."""
    }
    
    return rating_explanations.get(prediction_rating, "Rating assessment unavailable.")

def get_initial_assessment(user_inputs, prediction_rating):
    """
    Generate initial assessment (Why This Rating section only)
    """
    if not gemini_available:
        return get_fallback_assessment(user_inputs, prediction_rating)
    
    try:
        building_details = []
        for feature, value in user_inputs.items():
            clean_feature = feature.replace('_', ' ').title()
            building_details.append(f"- {clean_feature}: {value}")
        
        building_info = "\n".join(building_details)
        
        prompt = f"""
You are GreenyBot, an expert AI assistant specializing in GRIHA (Green Rating for Integrated Habitat Assessment) green building certification in India.

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

Please provide ONLY the "Why This Rating?" section. Explain why the building received this specific star rating based on the input parameters. Reference specific GRIHA criteria and requirements. Keep it concise and informative.

Format your response as plain text without any markdown formatting, emojis, or special characters.
"""

        response = gemini_model.generate_content(prompt)
        return response.text if response and response.text else get_fallback_assessment(user_inputs, prediction_rating)

    except Exception as e:
        print(f"Gemini error in initial assessment: {e}")
        return get_fallback_assessment(user_inputs, prediction_rating)

def get_section_details(user_inputs, prediction_rating, section_type):
    """
    Generate specific section details based on section type
    """
    if not gemini_available:
        return get_fallback_section_content(prediction_rating, section_type)
    
    try:
        building_details = []
        for feature, value in user_inputs.items():
            clean_feature = feature.replace('_', ' ').title()
            building_details.append(f"- {clean_feature}: {value}")
        
        building_info = "\n".join(building_details)
        
        section_prompts = {
            'strengths': f"""
Based on the building details and {prediction_rating} star GRIHA rating, list 3-4 key strengths of this building design. Focus on what the building does well in terms of sustainability and green features. Be specific and reference GRIHA criteria.

Building Details:
{building_info}

Provide only the strengths in a clear, numbered list format.
""",
            'improvements': f"""
Based on the building details and {prediction_rating} star GRIHA rating, provide 4-5 specific, actionable recommendations to improve the GRIHA rating. Focus on:
- Energy efficiency measures
- Water conservation strategies
- Sustainable materials and resources
- Indoor environmental quality improvements
- Innovation in design processes

Building Details:
{building_info}

Provide practical recommendations that can be implemented. Use numbered list format.
""",
            'benefits': f"""
Explain the benefits of implementing green building improvements for this {prediction_rating} star rated building. Cover:
- Environmental impact reduction
- Cost savings potential
- Health and comfort improvements
- Certification advantages
- Long-term value addition

Building Details:
{building_info}

Provide clear, actionable benefits.
""",
            'next_steps': f"""
Provide 3-4 immediate actionable steps that the building owner can take to improve their {prediction_rating} star GRIHA rating. Make these steps practical, prioritized, and feasible.

Building Details:
{building_info}

List the steps in order of priority.
"""
        }
        
        prompt = section_prompts.get(section_type, "Invalid section type")
        if prompt == "Invalid section type":
            return "Invalid section requested."
            
        response = gemini_model.generate_content(prompt)
        return response.text if response and response.text else get_fallback_section_content(prediction_rating, section_type)

    except Exception as e:
        print(f"Gemini error in section details: {e}")
        return get_fallback_section_content(prediction_rating, section_type)

def get_fallback_section_content(prediction_rating, section_type):
    """
    Provide fallback content when Gemini is not available
    """
    fallback_content = {
        'strengths': {
            1: "1. Building meets basic GRIHA compliance requirements\n2. Foundation for future green improvements established\n3. Regulatory compliance achieved\n4. Potential for significant sustainability upgrades",
            2: "1. Good foundation with some green initiatives implemented\n2. Energy efficiency measures partially in place\n3. Water conservation systems showing initial results\n4. Indoor environmental quality meets standard requirements",
            3: "1. Strong energy efficiency performance with multiple systems integrated\n2. Comprehensive water conservation and management strategies\n3. Good use of sustainable materials and resources\n4. Well-designed indoor environmental quality systems",
            4: "1. Excellent energy performance with advanced efficiency systems\n2. Outstanding water conservation and recycling measures\n3. Comprehensive sustainable materials and waste management\n4. Superior indoor environmental quality with smart controls",
            5: "1. Benchmark energy performance with innovative efficiency solutions\n2. Exemplary water conservation with zero discharge systems\n3. Outstanding sustainable materials usage and circular economy principles\n4. Exceptional indoor environmental quality with advanced monitoring"
        },
        'improvements': {
            1: "1. Implement energy-efficient lighting and HVAC systems\n2. Install water conservation fixtures and rainwater harvesting\n3. Use sustainable building materials and reduce waste\n4. Improve natural lighting and ventilation systems\n5. Add renewable energy systems like solar panels",
            2: "1. Upgrade to high-performance HVAC and lighting systems\n2. Enhance water recycling and greywater treatment\n3. Increase use of recycled and locally sourced materials\n4. Improve building envelope performance\n5. Implement smart building management systems",
            3: "1. Optimize energy systems with advanced controls and monitoring\n2. Implement advanced water treatment and reuse systems\n3. Enhance material lifecycle assessment and optimization\n4. Improve indoor air quality monitoring and control\n5. Add innovative sustainable technologies",
            4: "1. Implement cutting-edge energy storage and smart grid integration\n2. Achieve water positive status with advanced treatment systems\n3. Optimize material selection for minimal environmental impact\n4. Enhance occupant comfort with personalized environmental controls\n5. Integrate IoT and AI for building optimization",
            5: "1. Maintain peak performance through regular monitoring and optimization\n2. Share best practices and mentor other projects\n3. Implement emerging technologies for continuous improvement\n4. Enhance occupant engagement and education programs\n5. Pursue additional certifications and recognition"
        },
        'benefits': {
            1: "Implementing green improvements will significantly reduce operating costs, improve occupant health and comfort, increase property value, and position the building for future regulatory compliance.",
            2: "Green building improvements will deliver substantial energy and water cost savings, enhanced indoor environmental quality, increased marketability, and improved organizational sustainability credentials.",
            3: "Further sustainability enhancements will optimize operational efficiency, maximize occupant productivity and well-being, strengthen market position, and contribute to climate action goals.",
            4: "Advanced green building features will minimize environmental impact, maximize cost savings, ensure optimal occupant experience, and establish the building as a sustainability leader.",
            5: "Maintaining this exceptional performance ensures continued leadership in sustainability, maximizes all benefits, and creates lasting positive impact on the environment and community."
        },
        'next_steps': {
            1: "1. Conduct detailed energy audit to identify improvement opportunities\n2. Install basic water conservation fixtures and LED lighting\n3. Develop waste management and recycling programs\n4. Begin planning for renewable energy installation",
            2: "1. Upgrade HVAC systems with high-efficiency equipment\n2. Implement comprehensive water management strategies\n3. Source sustainable materials for upcoming renovations\n4. Install building management system for monitoring and control",
            3: "1. Optimize existing systems through advanced controls and monitoring\n2. Implement water recycling and treatment systems\n3. Conduct material lifecycle assessments for future projects\n4. Enhance indoor environmental quality monitoring",
            4: "1. Integrate smart building technologies and IoT systems\n2. Implement advanced water treatment for reuse applications\n3. Optimize material selection processes with sustainability criteria\n4. Pursue additional green building certifications",
            5: "1. Maintain performance through regular system optimization\n2. Document and share best practices with industry\n3. Explore emerging technologies for continuous improvement\n4. Engage occupants in sustainability education and participation"
        }
    }
    
    return fallback_content.get(section_type, {}).get(prediction_rating, "Content not available for this rating.")

def get_chat_response(user_inputs, prediction_rating, question):
    """
    Generate response to user's chat question with follow-up suggestions
    """
    if not gemini_available:
        return {
            'response': f"GreenyBot is currently using offline mode. Based on your {prediction_rating}-star GRIHA rating, I can provide general guidance. However, for detailed analysis, please ensure the Gemini AI service is properly configured.",
            'suggestions': [
                "What are the key areas for improvement?",
                "How can I reduce energy consumption?",
                "What are the benefits of higher GRIHA ratings?"
            ]
        }
    
    try:
        building_details = []
        for feature, value in user_inputs.items():
            clean_feature = feature.replace('_', ' ').title()
            building_details.append(f"- {clean_feature}: {value}")
        
        building_info = "\n".join(building_details)
        
        prompt = f"""
You are GreenyBot, an expert AI assistant specializing in GRIHA green building certification in India.

CONTEXT:
- Building GRIHA Rating: {prediction_rating} Stars
- Building Details: {building_info}

USER QUESTION: {question}

Please provide a helpful, informative response related to the building's GRIHA rating and sustainability features. Be specific and reference the building's current details when relevant. Keep responses concise and practical.

After your main response, suggest 3 relevant follow-up questions that the user might want to ask, formatted as:
FOLLOW_UP_QUESTIONS:
1. [Question 1]
2. [Question 2] 
3. [Question 3]
"""

        response = gemini_model.generate_content(prompt)
        if not response or not response.text:
            return {
                'response': "I apologize, but I'm having trouble generating a response right now. Please try again later.",
                'suggestions': ["What are the key areas for improvement?", "How can I reduce energy consumption?", "What are the benefits of higher GRIHA ratings?"]
            }
            
        full_response = response.text
        
        # Extract main response and follow-up questions
        if "FOLLOW_UP_QUESTIONS:" in full_response:
            parts = full_response.split("FOLLOW_UP_QUESTIONS:")
            main_response = parts[0].strip()
            suggestions_text = parts[1].strip()
            
            # Parse follow-up questions
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    question = line.split('.', 1)[1].strip()
                    if question.startswith('[') and question.endswith(']'):
                        question = question[1:-1]
                    suggestions.append(question)
        else:
            main_response = full_response
            # Default suggestions based on rating
            if prediction_rating <= 2:
                suggestions = [
                    "How can I improve my energy efficiency score?",
                    "What water conservation measures should I implement?",
                    "Which sustainable materials would be most cost-effective?"
                ]
            elif prediction_rating == 3:
                suggestions = [
                    "How can I reach a 4-star rating?",
                    "What are the most impactful improvements I can make?",
                    "How do I optimize indoor environmental quality?"
                ]
            else:
                suggestions = [
                    "How can I maintain this high rating over time?",
                    "What innovative features could push me to 5 stars?",
                    "How do I maximize the ROI of green investments?"
                ]
        
        return {
            'response': main_response,
            'suggestions': suggestions[:3]  # Ensure max 3 suggestions
        }

    except Exception as e:
        print(f"Gemini error in chat response: {e}")
        return {
            'response': f"I encountered an error while processing your question. Please try again later. Error: {str(e)}",
            'suggestions': []
        }

@app.route('/get_initial_assessment', methods=['POST'])
def get_initial_assessment_endpoint():
    """
    Get initial GreenyBot assessment (Why This Rating section)
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in user_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        session_data = user_sessions[session_id]
        assessment = get_initial_assessment(session_data['inputs'], session_data['prediction'])
        
        return jsonify({
            'success': True,
            'assessment': assessment
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Assessment error: {str(e)}'
        })

@app.route('/get_section', methods=['POST'])
def get_section():
    """
    Get specific section details
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        section_type = data.get('section_type')
        
        if session_id not in user_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        session_data = user_sessions[session_id]
        section_content = get_section_details(
            session_data['inputs'], 
            session_data['prediction'], 
            section_type
        )
        
        return jsonify({
            'success': True,
            'content': section_content
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Section error: {str(e)}'
        })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat questions with follow-up suggestions
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question = data.get('question')
        
        if session_id not in user_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        session_data = user_sessions[session_id]
        chat_data = get_chat_response(
            session_data['inputs'], 
            session_data['prediction'], 
            question
        )
        
        return jsonify({
            'success': True,
            'response': chat_data['response'],
            'suggestions': chat_data['suggestions']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Chat error: {str(e)}',
            'suggestions': []
        })

# Add health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'gemini_available': gemini_available,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)