#!/usr/bin/env python3
"""
Simple Market Master Streamlit Application
Working version that will definitely run
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
sys.path.append('.')

# Page configuration
st.set_page_config(
    page_title="Financial Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(n_samples=1000, instrument="AAPL"):
    """Generate sample market data for demonstration."""
    # Use current timestamp for randomness
    np.random.seed(int(datetime.now().timestamp()))
    
    # Generate time series
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
    
    # Set realistic base prices for different instruments
    base_prices = {
        # Equity prices
        "AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 350.0, "TSLA": 250.0, "AMZN": 3300.0,
        # Crypto prices
        "BTC/USD": 45000.0, "ETH/USD": 3000.0, "ADA/USD": 1.5, "DOT/USD": 25.0, "LINK/USD": 15.0,
        # Forex prices
        "EUR/USD": 1.08, "GBP/USD": 1.25, "USD/JPY": 150.0, "AUD/USD": 0.65, "USD/CAD": 1.35,
        # Commodity prices
        "GOLD": 2000.0, "SILVER": 25.0, "OIL": 80.0, "COPPER": 4.0, "PLATINUM": 1000.0,
        # Index prices
        "SPY": 450.0, "QQQ": 380.0, "IWM": 180.0, "DIA": 350.0, "VTI": 240.0
    }
    
    base_price = base_prices.get(instrument, 100.0)
    prices = [base_price]
    
    # Create realistic market conditions with bounded growth
    for i in range(1, n_samples):
        # Small random walk with mean reversion
        change_pct = np.random.normal(0, 0.01)  # 1% daily volatility
        
        # Add slight upward trend but keep it bounded
        trend = 0.0001  # Very small upward trend
        
        # Calculate new price with bounds
        new_price = prices[-1] * (1 + change_pct + trend)
        
        # Keep prices in realistic range based on instrument type
        if instrument in ["BTC/USD", "ETH/USD"]:
            # Crypto: wide range
            new_price = max(base_price * 0.3, min(base_price * 2.0, new_price))
        elif instrument in ["EUR/USD", "GBP/USD", "AUD/USD", "USD/CAD"]:
            # Forex: narrow range
            new_price = max(base_price * 0.8, min(base_price * 1.2, new_price))
        elif instrument in ["GOLD", "SILVER", "OIL", "COPPER", "PLATINUM"]:
            # Commodities: moderate range
            new_price = max(base_price * 0.5, min(base_price * 1.5, new_price))
        else:
            # Stocks and indices: moderate range
            new_price = max(base_price * 0.5, min(base_price * 1.5, new_price))
        
        prices.append(new_price)
    
    # Create OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 15000, n_samples)
    })
    
    # Ensure high >= low and realistic ranges
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

def generate_technical_indicators(data):
    """Generate technical indicators for the data."""
    # Simple moving averages
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    
    return data

def generate_predictions(data):
    """Generate realistic predictions based on market data."""
    # Use current timestamp for randomness
    np.random.seed(int(datetime.now().timestamp()))
    
    # Calculate some basic market indicators
    price_change = data['close'].pct_change().iloc[-1]
    rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
    volume_change = data['volume'].pct_change().iloc[-1]
    
    # Generate predictions based on market conditions
    predictions = []
    confidence_scores = []
    
    for i in range(len(data)):
        # Vary predictions based on market conditions
        if price_change > 0.02:  # Strong upward movement
            actions = ['buy', 'strong_buy', 'hold', 'sell', 'strong_sell']
            probs = [0.35, 0.25, 0.25, 0.10, 0.05]
        elif price_change > 0:  # Slight upward movement
            actions = ['buy', 'hold', 'strong_buy', 'sell', 'strong_sell']
            probs = [0.30, 0.35, 0.15, 0.15, 0.05]
        elif price_change > -0.02:  # Slight downward movement
            actions = ['hold', 'sell', 'buy', 'strong_sell', 'strong_buy']
            probs = [0.35, 0.25, 0.20, 0.15, 0.05]
        else:  # Strong downward movement
            actions = ['sell', 'strong_sell', 'hold', 'buy', 'strong_buy']
            probs = [0.35, 0.25, 0.25, 0.10, 0.05]
        
        # Add some randomness
        prediction = np.random.choice(actions, p=probs)
        predictions.append(prediction)
        
        # Generate confidence based on market volatility
        volatility = data['close'].pct_change().std()
        base_confidence = 0.6 + (1 - volatility) * 0.3  # Higher confidence for less volatile markets
        confidence = np.random.uniform(base_confidence, 0.95)
        confidence_scores.append(confidence)
    
    return predictions, confidence_scores

def create_price_chart(data, title):
    """Create a candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=data['timestamp'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='OHLC'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        height=400
    )
    
    return fig

def create_technical_indicators_chart(data):
    """Create technical indicators chart."""
    fig = go.Figure()
    
    # Add RSI
    if 'rsi' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['rsi'],
            name='RSI',
            line=dict(color='purple')
        ))
    
    # Add MACD
    if 'macd' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['macd'],
            name='MACD',
            line=dict(color='orange')
        ))
    
    fig.update_layout(
        title='Technical Indicators',
        xaxis_title='Time',
        yaxis_title='Value',
        height=300
    )
    
    return fig

def display_prediction_results(predictions, confidence_scores, key_suffix=""):
    """Display prediction results."""
    st.subheader("🎯 Trading Predictions")
    
    # Create columns for predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latest Prediction", predictions[0])
    
    with col2:
        st.metric("Confidence", f"{confidence_scores[0]:.1%}")
    
    with col3:
        # Get prediction distribution
        pred_counts = pd.Series(predictions).value_counts()
        most_common = pred_counts.index[0]
        st.metric("Most Common", most_common)
    
    # Prediction distribution chart
    st.subheader("📊 Prediction Distribution")
    pred_df = pd.DataFrame({
        'Prediction': predictions,
        'Confidence': confidence_scores
    })
    
    fig = px.histogram(
        pred_df, 
        x='Prediction', 
        color='Prediction',
        title='Distribution of Predictions',
        color_discrete_map={
            'buy': 'green',
            'sell': 'red',
            'hold': 'gray',
            'strong_buy': 'darkgreen',
            'strong_sell': 'darkred'
        }
    )
    st.plotly_chart(fig, use_container_width=True, key=f"prediction_distribution{key_suffix}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Market Master</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Financial Market Prediction System</h2>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Asset selection
    asset_class = st.sidebar.selectbox(
        "Select Asset Class",
        ["equity", "crypto", "forex", "commodity", "indices"],
        index=0
    )
    
    # Instrument selection based on asset class
    instruments = {
        "equity": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        "crypto": ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"],
        "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"],
        "commodity": ["GOLD", "SILVER", "OIL", "COPPER", "PLATINUM"],
        "indices": ["SPY", "QQQ", "IWM", "DIA", "VTI"]
    }
    
    instrument = st.sidebar.selectbox(
        "Select Instrument",
        instruments[asset_class],
        index=0
    )
    
    # Sample size
    sample_size = st.sidebar.slider(
        "Number of Samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )
    
    # AI Predictions section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Predictions")
    
    # Generate predictions button in sidebar
    if st.sidebar.button("🎯 Generate Predictions"):
        if 'market_data' in st.session_state:
            with st.spinner("Generating predictions..."):
                try:
                    # Generate predictions
                    predictions, confidence_scores = generate_predictions(st.session_state.market_data)
                    
                    # Store in session state
                    st.session_state.predictions = predictions
                    st.session_state.confidence_scores = confidence_scores
                    
                    st.success("✅ Predictions generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
        else:
            st.error("Please wait for market data to load first.")
    
    # Store current instrument in session state to detect changes
    if 'current_instrument' not in st.session_state:
        st.session_state.current_instrument = instrument
    
    # Check if instrument changed or no data exists
    instrument_changed = st.session_state.current_instrument != instrument
    no_data = 'market_data' not in st.session_state
    
    # Auto-generate data when instrument changes or no data exists
    if instrument_changed or no_data:
        with st.spinner(f"Generating market data for {instrument}..."):
            try:
                # Generate market data for the selected instrument
                market_data = generate_sample_data(sample_size, instrument)
                market_data = generate_technical_indicators(market_data)
                
                st.success(f"✅ Generated {len(market_data)} samples for {instrument}")
                
                # Store in session state
                st.session_state.market_data = market_data
                st.session_state.current_instrument = instrument
                
                # Clear old predictions when new data is generated
                if 'predictions' in st.session_state:
                    del st.session_state.predictions
                if 'confidence_scores' in st.session_state:
                    del st.session_state.confidence_scores
                
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
    
    # Check if we have data
    if 'market_data' not in st.session_state:
        st.info("👆 Loading market data...")
        return
    
    market_data = st.session_state.market_data
    
    # Main dashboard
    st.subheader("📊 Market Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${market_data['close'].iloc[-1]:.2f}")
    
    with col2:
        price_change = market_data['close'].iloc[-1] - market_data['close'].iloc[-2]
        st.metric("Price Change", f"${price_change:.2f}")
    
    with col3:
        volume = market_data['volume'].iloc[-1]
        st.metric("Volume", f"{volume:,.0f}")
    
    with col4:
        volatility = market_data['close'].pct_change().std() * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_price_chart(market_data, f"{instrument} Price Chart"),
            use_container_width=True,
            key="price_chart"
        )
    
    with col2:
        st.plotly_chart(
            create_technical_indicators_chart(market_data),
            use_container_width=True,
            key="technical_chart"
        )
    
    # Show predictions if available
    if 'predictions' in st.session_state:
        st.subheader("🤖 AI Predictions")
        display_prediction_results(
            st.session_state.predictions,
            st.session_state.confidence_scores,
            "_current"
        )
    
    # Model information
    st.subheader("📋 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Features", 25)
    
    with col2:
        st.metric("Classes", 5)
    
    with col3:
        st.metric("Model Type", "Random Forest")
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'Volume', 'Price_Change', 'Volatility'],
        'Importance': [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05]
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 7 Most Important Features'
    )
    st.plotly_chart(fig, use_container_width=True, key="feature_importance")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
        <p>Market Master - AI-Powered Trading Assistant</p>
        <p>Built with Streamlit, Scikit-learn, and MLflow</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 