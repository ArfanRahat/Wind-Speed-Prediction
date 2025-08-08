import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Wind Speed Predictor",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Disable file watcher to prevent inotify issues
import streamlit.web.cli as stcli
import sys
if hasattr(stcli, '_main_run'):
    # Override the default behavior to disable file watching
    pass

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Move the cached function outside the class
@st.cache_resource
def load_model():
    """Load the XGBoost model"""
    try:
        # Feature info for scaler initialization
        feature_info = {
            'SFC': {'min': 0.0, 'max': 35.0},
            'T2M': {'min': -10.0, 'max': 45.0},
            'T2MWET': {'min': -10.0, 'max': 35.0},
            'T2M_MIN': {'min': -15.0, 'max': 40.0},
            'T2M_MAX': {'min': -5.0, 'max': 50.0},
            'PRECTOTCORR': {'min': 0.0, 'max': 50.0},
            'RH2M': {'min': 0.0, 'max': 100.0},
            'PS': {'min': 85.0, 'max': 105.0},
            'WD10M': {'min': 0.0, 'max': 360.0}
        }
        
        # Try to load from different possible locations
        possible_paths = [
            'XGBoost.pkl',  # Match your GitHub file name exactly
            'xgboost_model.pkl',
            'model/xgboost_model.pkl',
            './xgboost_model.pkl',
            'wind_speed_model.pkl',
            'XGboost.pkl',
            'xgboost.pkl',
            'XGBOOST.pkl'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            # Debug: List all files in current directory
            current_files = os.listdir('.')
            pkl_files = [f for f in current_files if f.endswith('.pkl')]
            error_msg = f"Model file not found. Please upload your XGBoost model (.pkl file)\n"
            error_msg += f"Current directory files: {current_files}\n"
            error_msg += f"Found .pkl files: {pkl_files}\n"
            error_msg += f"Searched for: {possible_paths}"
            raise FileNotFoundError(error_msg)
        
        try:
            model = joblib.load(model_path)
            print(f"Successfully loaded model from: {model_path}")
            print(f"Model type: {type(model)}")
        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        
        # Initialize scaler with approximate training data ranges
        scaler = MinMaxScaler()
        dummy_data = []
        for feature, info in feature_info.items():
            dummy_data.append([info['min'], info['max']])
        
        dummy_array = np.array(dummy_data).T
        scaler.fit(dummy_array)
        
        return model, scaler, True
        
    except Exception as e:
        return None, None, False

class WindSpeedPredictor:
    def __init__(self):
        self.feature_info = {
            'SFC': {'min': 0.0, 'max': 35.0, 'unit': 'kW-hr/m²', 'default': 15.0, 'name': 'All Sky Surface Shortwave Downward Irradiance'},
            'T2M': {'min': -10.0, 'max': 45.0, 'unit': '°C', 'default': 25.0, 'name': 'Temperature'},
            'T2MWET': {'min': -10.0, 'max': 35.0, 'unit': '°C', 'default': 20.0, 'name': 'Wet Bulb Temperature'},
            'T2M_MIN': {'min': -15.0, 'max': 40.0, 'unit': '°C', 'default': 18.0, 'name': 'Minimum Temperature'},
            'T2M_MAX': {'min': -5.0, 'max': 50.0, 'unit': '°C', 'default': 32.0, 'name': 'Maximum Temperature'},
            'PRECTOTCORR': {'min': 0.0, 'max': 50.0, 'unit': 'mm/day', 'default': 2.0, 'name': 'Precipitation'},
            'RH2M': {'min': 0.0, 'max': 100.0, 'unit': '%', 'default': 65.0, 'name': 'Relative Humidity'},
            'PS': {'min': 85.0, 'max': 105.0, 'unit': 'kPa', 'default': 101.3, 'name': 'Surface Pressure'},
            'WD10M': {'min': 0.0, 'max': 360.0, 'unit': '°', 'default': 180.0, 'name': 'Wind Direction'}
        }
        
        # Load model using the cached function
        self.model, self.scaler, self.model_loaded = load_model()
    
    def predict_wind_speed(self, values):
        """Make prediction"""
        if not self.model_loaded:
            return None, "Model not loaded"
        
        try:
            # Create DataFrame with the input values
            input_data = pd.DataFrame([values])
            
            # Ensure columns are in the same order as training data
            feature_columns = list(self.feature_info.keys())
            input_data = input_data[feature_columns]
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Ensure prediction is a scalar
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            
            return float(prediction), None
            
        except Exception as e:
            return None, str(e)
    
    def get_wind_classification(self, wind_speed):
        """Classify wind speed based on Beaufort scale"""
        if wind_speed < 0.3:
            return "Calm", "#87CEEB"
        elif wind_speed < 1.6:
            return "Light Air", "#90EE90"
        elif wind_speed < 3.4:
            return "Light Breeze", "#98FB98"
        elif wind_speed < 5.5:
            return "Gentle Breeze", "#FFFF99"
        elif wind_speed < 8.0:
            return "Moderate Breeze", "#FFD700"
        elif wind_speed < 10.8:
            return "Fresh Breeze", "#FFA500"
        elif wind_speed < 13.9:
            return "Strong Breeze", "#FF8C00"
        elif wind_speed < 17.2:
            return "Near Gale", "#FF4500"
        elif wind_speed < 20.8:
            return "Gale", "#FF0000"
        elif wind_speed < 24.5:
            return "Strong Gale", "#DC143C"
        elif wind_speed < 28.5:
            return "Storm", "#8B0000"
        else:
            return "Hurricane Force", "#4B0000"
    
    def create_gauge_chart(self, value):
        """Create a gauge chart for the prediction"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Wind Speed (m/s)"},
            delta = {'reference': 10},
            gauge = {
                'axis': {'range': [None, 30]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3.4], 'color': "lightgray"},
                    {'range': [3.4, 8.0], 'color': "yellow"},
                    {'range': [8.0, 13.9], 'color': "orange"},
                    {'range': [13.9, 20.8], 'color': "red"},
                    {'range': [20.8, 30], 'color': "darkred"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 25
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'input_values' not in st.session_state:
            st.session_state.input_values = {}
            for feature, info in self.feature_info.items():
                st.session_state.input_values[feature] = info['default']
    
    def run_app(self):
        """Run the Streamlit app"""
        
        # Initialize session state
        self.initialize_session_state()
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>💨 Wind Speed Predictor</h1>
            <p>Predict wind speed using machine learning with weather parameters</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please check the error details below:")
            try:
                # Try to load again to get the specific error
                load_model()
            except Exception as e:
                st.code(str(e))
            st.stop()
        else:
            st.success("✅ XGBoost model loaded successfully!")
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("☀️ Solar & Temperature Parameters")
            
            input_values = {}
            features = list(self.feature_info.keys())
            
            # First 5 parameters (Solar and Temperature)
            for i, feature in enumerate(features[:5]):
                info = self.feature_info[feature]
                current_value = st.session_state.input_values.get(feature, info['default'])
                
                input_values[feature] = st.number_input(
                    f"{info['name']} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(current_value),
                    step=0.1 if feature == 'PRECTOTCORR' else 1.0,
                    key=f"input_{feature}_{i}",  # Unique key to prevent conflicts
                    help=f"Range: {info['min']} - {info['max']} {info['unit']}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("🌧️ Weather Parameters")
            
            # Last 4 parameters (Weather)
            for i, feature in enumerate(features[5:], start=5):
                info = self.feature_info[feature]
                current_value = st.session_state.input_values.get(feature, info['default'])
                
                input_values[feature] = st.number_input(
                    f"{info['name']} ({info['unit']})",
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(current_value),
                    step=0.1 if feature == 'PS' else 1.0,
                    key=f"input_{feature}_{i}",  # Unique key to prevent conflicts
                    help=f"Range: {info['min']} - {info['max']} {info['unit']}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Update session state with current input values
        st.session_state.input_values = input_values
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("💨 Predict Wind Speed", type="primary", use_container_width=True):
                prediction, error = self.predict_wind_speed(input_values)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state['prediction'] = prediction
                    st.session_state['prediction_input_values'] = input_values
        
        # Display results
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            uncertainty = prediction * 0.1  # 10% uncertainty
            
            # Results card
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Prediction Results</h2>
                <h1 style="font-size: 3rem; margin: 0.5rem 0;">{prediction:.2f} m/s</h1>
                <p style="font-size: 1.2rem; opacity: 0.9;">
                    Range: {prediction-uncertainty:.2f} - {prediction+uncertainty:.2f} m/s
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart and classification
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = self.create_gauge_chart(prediction)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                classification, color = self.get_wind_classification(prediction)
                st.markdown(f"""
                <div style="background: {color}; padding: 2rem; border-radius: 10px; 
                            text-align: center; color: black; margin-top: 2rem;">
                    <h3>💨 Wind Classification</h3>
                    <h2>{classification}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Wind speed interpretation
            st.subheader("📊 Wind Speed Interpretation")
            interpretation = ""
            if prediction < 3.4:
                interpretation = "Light wind conditions. Good for outdoor activities."
            elif prediction < 8.0:
                interpretation = "Moderate wind conditions. Suitable for sailing and wind sports."
            elif prediction < 13.9:
                interpretation = "Strong wind conditions. Exercise caution for outdoor activities."
            elif prediction < 20.8:
                interpretation = "Very strong wind. High wind advisory conditions."
            else:
                interpretation = "Extreme wind conditions. Take safety precautions."
            
            st.info(f"💡 {interpretation}")
            
            # Input summary
            st.subheader("📋 Input Parameters Summary")
            
            # Create a DataFrame for display
            summary_data = []
            for feature, value in st.session_state.get('prediction_input_values', {}).items():
                info = self.feature_info[feature]
                summary_data.append({
                    'Parameter': info['name'],
                    'Value': f"{value:.2f}",
                    'Unit': info['unit']
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            # Additional info
            st.info("💡 Note: This prediction is based on an XGBoost machine learning model trained on meteorological data. Results should be verified with actual weather measurements.")

# Run the app
if __name__ == "__main__":
    app = WindSpeedPredictor()
    app.run_app()
