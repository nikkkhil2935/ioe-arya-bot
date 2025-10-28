import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration with custom theme
st.set_page_config(
    page_title="Smart Parking Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .vacant {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .occupied {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vacancy_model():
    return joblib.load('xgb_parking_vacancy_model.pkl')

@st.cache_resource
def load_vehicle_type_model():
    return joblib.load('xgb_vehicle_type_model.pkl')

@st.cache_data
def load_parking_data():
    try:
        return pd.read_csv('preprocessed_parking_data.csv')
    except:
        return None

vacancy_model = load_vacancy_model()
vehicle_type_model = load_vehicle_type_model()
parking_data = load_parking_data()


# Header with emoji and styling
st.markdown("<h1>🚗 Smart Parking Management System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1.1em;'>AI-Powered Parking Occupancy & Vehicle Type Prediction</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation and controls
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/car.png", width=100)
    st.title("🎛️ Control Panel")
    
    page = st.radio("Navigation", 
                    ["🏠 Main Dashboard", "📊 Analytics", "📈 Insights", "ℹ️ About"],
                    label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    show_probabilities = st.checkbox("Show Prediction Probabilities", value=True)
    auto_refresh = st.checkbox("Auto Refresh (Demo Mode)", value=False)
    
    if auto_refresh:
        st.info("🔄 Dashboard will refresh every 5 seconds")
        time.sleep(5)
        st.rerun()

# Main Dashboard Page
if page == "🏠 Main Dashboard":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Make Predictions")
        st.markdown('<div class="info-box">Adjust the parameters below to predict parking availability and vehicle type</div>', unsafe_allow_html=True)
    
    with col2:
        # Display current time
        current_time = datetime.now()
        st.metric("🕐 Current Time", current_time.strftime("%H:%M:%S"))
        st.metric("📅 Date", current_time.strftime("%A, %B %d, %Y"))
    
    st.markdown("---")
    
    # Input Parameters in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ⏰ Time Settings")
        entry_hour = st.slider("Entry Hour (0–23)", 0, 23, 12, 
                               help="Select the hour when the vehicle enters the parking")
        
        # Visual hour representation
        hour_emoji = "🌙" if entry_hour < 6 or entry_hour > 20 else "☀️" if entry_hour < 18 else "🌆"
        st.markdown(f"**Time Period:** {hour_emoji} {'Morning' if 6 <= entry_hour < 12 else 'Afternoon' if 12 <= entry_hour < 17 else 'Evening' if 17 <= entry_hour < 21 else 'Night'}")
    
    with col2:
        st.markdown("#### 📅 Date Settings")
        day_of_week = st.select_slider("Day of Week", 
                                       options=list(range(7)), 
                                       value=0,
                                       format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        is_weekend = 1 if day_of_week >= 5 else 0
        
        weekend_emoji = "🎉" if is_weekend else "💼"
        st.markdown(f"**Day Type:** {weekend_emoji} {'Weekend' if is_weekend else 'Weekday'}")
    
    with col3:
        st.markdown("#### ⏱️ Duration Settings")
        duration = st.number_input("Parking Duration (minutes)", 
                                   min_value=1, 
                                   max_value=1440, 
                                   value=60, 
                                   step=15,
                                   help="Expected parking duration in minutes")
        
        # Convert duration to hours and minutes
        hours = duration // 60
        minutes = duration % 60
        st.markdown(f"**Duration:** {hours}h {minutes}m")
    
    # Calculate features
    bins = [0, 6, 9, 12, 17, 20, 24]
    hour_bin = pd.cut([entry_hour], bins=bins, labels=False, right=False)[0]
    
    # Features for models
    vacancy_features = pd.DataFrame({
        "Entry_Hour": [entry_hour],
        "DayOfWeek": [day_of_week],
        "Is_Weekend": [is_weekend],
        "Hour_Bin": [hour_bin],
    })
    
    vehicle_features = pd.DataFrame({
        "Entry_Hour": [entry_hour],
        "Duration": [duration],
        "DayOfWeek": [day_of_week],
        "Is_Weekend": [is_weekend],
        "Hour_Bin": [hour_bin],
    })
    
    st.markdown("---")
    
    # Prediction Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔮 Predict Now", use_container_width=True, type="primary")
    
    if predict_button or 'predictions_made' in st.session_state:
        st.session_state.predictions_made = True
        
        # Make predictions
        vacancy_pred = vacancy_model.predict(vacancy_features)[0]
        vehicle_pred = vehicle_type_model.predict(vehicle_features)[0]
        
        # Get probabilities if enabled
        if show_probabilities:
            vacancy_proba = vacancy_model.predict_proba(vacancy_features)[0]
            vehicle_proba = vehicle_type_model.predict_proba(vehicle_features)[0]
        
        vacancy_status = "Vacant" if vacancy_pred == 1 else "Occupied"
        vehicle_type = "Two Wheeler" if vehicle_pred == 1 else "Four Wheeler"
        
        st.markdown("### 🎯 Prediction Results")
        
        # Display results in attractive boxes
        col1, col2 = st.columns(2)
        
        with col1:
            status_class = "vacant" if vacancy_pred == 1 else "occupied"
            status_emoji = "✅" if vacancy_pred == 1 else "🚫"
            st.markdown(f'<div class="prediction-box {status_class}">{status_emoji} Parking Slot Status<br/><span style="font-size: 1.5em;">{vacancy_status}</span></div>', unsafe_allow_html=True)
            
            if show_probabilities:
                # Create gauge chart for vacancy probability
                fig_vacancy = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = vacancy_proba[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Vacancy Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green" if vacancy_pred == 1 else "red"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_vacancy.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_vacancy, use_container_width=True)
        
        with col2:
            vehicle_emoji = "🏍️" if vehicle_pred == 1 else "🚗"
            st.markdown(f'<div class="prediction-box" style="background-color: #fff3cd; color: #856404; border: 2px solid #ffc107;">{vehicle_emoji} Vehicle Type<br/><span style="font-size: 1.5em;">{vehicle_type}</span></div>', unsafe_allow_html=True)
            
            if show_probabilities:
                # Create gauge chart for vehicle type probability
                fig_vehicle = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = vehicle_proba[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Two Wheeler Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_vehicle.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_vehicle, use_container_width=True)
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 💡 Contextual Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Peak hour analysis
            is_peak = 9 <= entry_hour <= 17
            st.metric("🕐 Time Category", 
                     "Peak Hours" if is_peak else "Off-Peak",
                     delta="High Traffic" if is_peak else "Low Traffic")
        
        with col2:
            # Estimated wait time
            wait_time = 0 if vacancy_pred == 1 else np.random.randint(5, 30)
            st.metric("⏳ Estimated Wait Time", 
                     f"{wait_time} min",
                     delta="Available" if wait_time == 0 else f"-{wait_time}m")
        
        with col3:
            # Parking fee estimation
            fee_per_hour = 20 if vehicle_pred == 0 else 10  # Four wheeler vs Two wheeler
            estimated_fee = (duration / 60) * fee_per_hour
            st.metric("💰 Estimated Parking Fee", 
                     f"₹{estimated_fee:.2f}",
                     delta=f"₹{fee_per_hour}/hr")

# Analytics Page
elif page == "📊 Analytics":
    st.markdown("### 📊 Parking Analytics Dashboard")
    
    if parking_data is not None:
        st.markdown('<div class="info-box">📈 Analyzing historical parking data patterns</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Records", f"{len(parking_data):,}")
        
        with col2:
            two_wheeler_pct = (parking_data['Type of Vehicle_Two Wheeler'].sum() / len(parking_data)) * 100
            st.metric("🏍️ Two Wheelers", f"{two_wheeler_pct:.1f}%")
        
        with col3:
            avg_duration = parking_data['Duration'].mean()
            st.metric("⏱️ Avg Duration", f"{avg_duration:.0f} min")
        
        with col4:
            weekend_pct = (parking_data['Is_Weekend'].sum() / len(parking_data)) * 100
            st.metric("🎉 Weekend Parking", f"{weekend_pct:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["⏰ Hourly Trends", "📅 Weekly Patterns", "🚗 Vehicle Types", "⏱️ Duration Analysis"])
        
        with tab1:
            # Hourly distribution
            hourly_data = parking_data.groupby('Entry_Hour').size().reset_index(name='Count')
            fig_hourly = px.bar(hourly_data, x='Entry_Hour', y='Count',
                               title='Parking Entries by Hour of Day',
                               labels={'Entry_Hour': 'Hour', 'Count': 'Number of Vehicles'},
                               color='Count',
                               color_continuous_scale='Blues')
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Peak hours identification
            peak_hour = hourly_data.loc[hourly_data['Count'].idxmax()]
            st.info(f"🔥 **Peak Hour:** {int(peak_hour['Entry_Hour'])}:00 with {int(peak_hour['Count'])} entries")
        
        with tab2:
            # Weekly pattern
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_data = parking_data.groupby('DayOfWeek').size().reset_index(name='Count')
            weekly_data['Day'] = weekly_data['DayOfWeek'].apply(lambda x: day_names[x])
            
            fig_weekly = px.line(weekly_data, x='Day', y='Count',
                                title='Parking Entries by Day of Week',
                                markers=True,
                                color_discrete_sequence=['#2ecc71'])
            fig_weekly.update_layout(height=400)
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            busiest_day = weekly_data.loc[weekly_data['Count'].idxmax()]
            st.info(f"📅 **Busiest Day:** {busiest_day['Day']} with {int(busiest_day['Count'])} entries")
        
        with tab3:
            # Vehicle type distribution
            vehicle_counts = pd.DataFrame({
                'Vehicle Type': ['Two Wheeler', 'Four Wheeler'],
                'Count': [parking_data['Type of Vehicle_Two Wheeler'].sum(), 
                         len(parking_data) - parking_data['Type of Vehicle_Two Wheeler'].sum()]
            })
            
            fig_vehicle = px.pie(vehicle_counts, values='Count', names='Vehicle Type',
                                title='Vehicle Type Distribution',
                                color_discrete_sequence=['#3498db', '#e74c3c'],
                                hole=0.4)
            fig_vehicle.update_layout(height=400)
            st.plotly_chart(fig_vehicle, use_container_width=True)
        
        with tab4:
            # Duration analysis
            fig_duration = px.histogram(parking_data, x='Duration',
                                       title='Parking Duration Distribution',
                                       labels={'Duration': 'Duration (minutes)', 'count': 'Frequency'},
                                       color_discrete_sequence=['#9b59b6'],
                                       nbins=50)
            fig_duration.update_layout(height=400)
            st.plotly_chart(fig_duration, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Median Duration", f"{parking_data['Duration'].median():.0f} min")
            with col2:
                st.metric("⬆️ Max Duration", f"{parking_data['Duration'].max():.0f} min")
            with col3:
                st.metric("⬇️ Min Duration", f"{parking_data['Duration'].min():.0f} min")
    else:
        st.warning("⚠️ No parking data available for analytics")

# Insights Page
elif page == "📈 Insights":
    st.markdown("### 📈 Business Insights & Recommendations")
    
    if parking_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Key Findings")
            
            # Peak hours
            hourly_counts = parking_data.groupby('Entry_Hour').size()
            peak_hours = hourly_counts.nlargest(3)
            
            st.markdown("**🔥 Peak Hours:**")
            for hour, count in peak_hours.items():
                st.markdown(f"- **{int(hour)}:00** - {int(count)} entries")
            
            # Weekend vs Weekday
            weekend_avg = parking_data[parking_data['Is_Weekend'] == 1].groupby('Entry_Hour').size().mean()
            weekday_avg = parking_data[parking_data['Is_Weekend'] == 0].groupby('Entry_Hour').size().mean()
            
            st.markdown(f"\n**📅 Average Hourly Entries:**")
            st.markdown(f"- Weekdays: {weekday_avg:.1f} vehicles/hour")
            st.markdown(f"- Weekends: {weekend_avg:.1f} vehicles/hour")
            
            # Vehicle preference
            two_wheeler_ratio = parking_data['Type of Vehicle_Two Wheeler'].mean()
            st.markdown(f"\n**🏍️ Vehicle Preference:**")
            st.markdown(f"- Two Wheelers: {two_wheeler_ratio*100:.1f}%")
            st.markdown(f"- Four Wheelers: {(1-two_wheeler_ratio)*100:.1f}%")
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            
            st.markdown("""
            <div class="info-box">
            <b>🚀 Optimization Strategies:</b><br/><br/>
            
            <b>1. Dynamic Pricing</b><br/>
            • Implement surge pricing during peak hours<br/>
            • Offer discounts during off-peak times<br/><br/>
            
            <b>2. Resource Allocation</b><br/>
            • Deploy more staff during peak hours<br/>
            • Schedule maintenance during low-traffic periods<br/><br/>
            
            <b>3. Capacity Planning</b><br/>
            • Expand two-wheeler parking zones<br/>
            • Consider reservation system for peak hours<br/><br/>
            
            <b>4. Customer Experience</b><br/>
            • Provide real-time availability updates<br/>
            • Implement mobile app for booking<br/>
            • Add loyalty programs for frequent users
            </div>
            """, unsafe_allow_html=True)
        
        # Heatmap of parking activity
        st.markdown("---")
        st.markdown("#### 🔥 Parking Activity Heatmap")
        
        heatmap_data = parking_data.groupby(['DayOfWeek', 'Entry_Hour']).size().reset_index(name='Count')
        heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Entry_Hour', values='Count').fillna(0)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_pivot.index = [day_names[i] for i in heatmap_pivot.index]
        
        fig_heatmap = px.imshow(heatmap_pivot,
                               labels=dict(x="Hour of Day", y="Day of Week", color="Entries"),
                               x=heatmap_pivot.columns,
                               y=heatmap_pivot.index,
                               color_continuous_scale='RdYlGn',
                               aspect="auto")
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    else:
        st.warning("⚠️ No parking data available for insights")

# About Page
else:
    st.markdown("### ℹ️ About Smart Parking System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Project Overview</h4>
        This Smart Parking Management System uses advanced machine learning algorithms to predict:
        <ul>
            <li>🅿️ Parking slot availability (Vacant/Occupied)</li>
            <li>🚗 Vehicle type classification (Two-Wheeler/Four-Wheeler)</li>
        </ul>
        
        The system analyzes temporal patterns including entry time, day of week, and parking duration 
        to provide accurate predictions that help optimize parking operations.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box" style="border-left-color: #9b59b6;">
        <h4>🤖 Machine Learning Models</h4>
        <b>Vacancy Prediction Model:</b><br/>
        • Algorithm: XGBoost Classifier<br/>
        • Features: Entry Hour, Day of Week, Weekend Flag, Hour Bin<br/>
        • Accuracy: ~68.6%<br/><br/>
        
        <b>Vehicle Type Model:</b><br/>
        • Algorithm: XGBoost Classifier<br/>
        • Features: Entry Hour, Duration, Day of Week, Weekend Flag, Hour Bin<br/>
        • Accuracy: ~50.7%
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="border-left-color: #e74c3c;">
        <h4>🛠️ Technologies</h4>
        • Python 3.12<br/>
        • Streamlit<br/>
        • XGBoost<br/>
        • Pandas<br/>
        • Plotly<br/>
        • Scikit-learn
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box" style="border-left-color: #2ecc71;">
        <h4>👨‍💻 Developer</h4>
        <b>Arya Sable</b><br/>
        IOE Arya Project<br/><br/>
        
        <b>GitHub:</b><br/>
        <a href="https://github.com/nikkkhil2935/ioe-arya-bot" target="_blank">
        nikkkhil2935/ioe-arya-bot
        </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features showcase
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Real-time Predictions**
        - Instant vacancy detection
        - Vehicle type classification
        - Probability scores
        """)
    
    with col2:
        st.markdown("""
        **📊 Advanced Analytics**
        - Historical data analysis
        - Trend visualization
        - Pattern recognition
        """)
    
    with col3:
        st.markdown("""
        **💡 Business Insights**
        - Peak hour identification
        - Occupancy patterns
        - Revenue optimization
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>© 2025 Smart Parking System | Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)

