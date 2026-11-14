"""
AI Financial Analyst Agent - Part 4: Streamlit Dashboard
The Complete Web Interface
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import tempfile

# Configure page
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #1e3a5f;
    }
    .assistant-message {
        background: #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# Import our modules
from AI_Financial_Analyst_Part1 import MarketData, fetch_news, create_candlestick_chart
from AI_Financial_Analyst_Part2 import get_llm, create_single_agent, create_multi_agent_graph, TOOLS
from AI_Financial_Analyst_Part3 import EarningsRAG, BacktestingEngine


# Initialize components
@st.cache_resource
def init_components():
    """Initialize all components."""
    market_data = MarketData()
    earnings_rag = EarningsRAG()
    return market_data, earnings_rag


market_data, earnings_rag = init_components()


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker_dashboard(ticker: str):
    """Cached fetch of fundamentals, 90-day history, and recent news."""
    fundamentals = market_data.get_fundamentals(ticker)
    history = market_data.get_stock_history(ticker, 90)
    news = fetch_news(ticker, 5)
    return fundamentals, history, news

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "AAPL"
if "agent_type" not in st.session_state:
    st.session_state.agent_type = "single"

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Key input
    api_key = st.text_input("Groq API Key", type="password", 
                           value=os.environ.get("GROQ_API_KEY", ""))
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    st.divider()
    
    # Ticker input
    ticker = st.text_input("Stock Ticker", value=st.session_state.current_ticker)
    if ticker != st.session_state.current_ticker:
        st.session_state.current_ticker = ticker.upper()
    
    # Agent selection
    agent_type = st.selectbox(
        "Agent Type",
        ["Single Agent", "Multi-Agent (Full Analysis)"],
        index=0 if st.session_state.agent_type == "single" else 1
    )
    st.session_state.agent_type = "single" if agent_type == "Single Agent" else "multi"
    
    # Advanced options
    with st.expander("Advanced Options"):
        enable_reflexion = st.checkbox("Enable Reflexion (Self-Correction)", value=True)
        time_travel_mode = st.checkbox("Time Travel Mode (Backtesting)")
        if time_travel_mode:
            backtest_date = st.date_input(
                "Simulate Date",
                value=datetime.now() - timedelta(days=365)
            )
    
    st.divider()
    
    # PDF Upload for RAG
    st.markdown("### 📄 Earnings Call Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        # Save temporarily and process
        with st.spinner("Processing PDF..."):
            temp_dir = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                chunks = earnings_rag.ingest_document(
                    temp_path,
                    {"ticker": st.session_state.current_ticker}
                )
                st.success(f"Processed {chunks} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main layout - Two columns
col1, col2 = st.columns([1, 1])

# Left column - Chat interface
with col1:
    st.markdown('<p class="main-header">🤖 AI Financial Analyst</p>', unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>You:</b> {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>Analyst:</b> {content}
                </div>
                """, unsafe_allow_html=True)
    
    # Input
    user_input = st.chat_input("Ask about stocks, markets, or upload earnings calls...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process with agent
        if api_key:
            with st.spinner("Analyzing..."):
                try:
                    if st.session_state.agent_type == "single":
                        from langchain_core.messages import HumanMessage
                        agent = create_single_agent()
                        config = {"configurable": {"thread_id": "streamlit-session"}}
                        
                        result = agent.invoke({
                            "messages": [HumanMessage(content=user_input)],
                            "current_ticker": st.session_state.current_ticker,
                            "agent_type": "single",
                            "analysis_results": {}
                        }, config)
                        
                        response = result["messages"][-1].content
                    else:
                        graph = create_multi_agent_graph()
                        result = graph.invoke({
                            "messages": [],
                            "query": user_input,
                            "market_analysis": "",
                            "news_analysis": "",
                            "final_report": "",
                            "query_type": ""
                        })
                        response = result["final_report"]
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Please enter your Groq API key in the sidebar to enable AI analysis."
            })
        
        st.rerun()

# Right column - Live data and charts
with col2:
    ticker = st.session_state.current_ticker
    
    # Stock info header
    st.markdown(f"### 📊 {ticker} Analysis")
    
    try:
        with st.spinner(f"Loading {ticker} market data..."):
            fundamentals, history, news = load_ticker_dashboard(ticker)
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Price", f"${fundamentals['current_price']}")
        with m2:
            st.metric("P/E Ratio", fundamentals['pe_ratio'])
        with m3:
            st.metric("52W High", f"${fundamentals['52_week_high']}")
        with m4:
            st.metric("52W Low", f"${fundamentals['52_week_low']}")
        
        # Candlestick chart
        fig = create_candlestick_chart(history, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # News section
        st.markdown("### 📰 Recent News")
        for item in news[:5]:
            with st.expander(item['title'][:80] + "..."):
                st.write(item.get('snippet', 'No preview available'))
                if item.get('link'):
                    st.markdown(f"[Read more]({item['link']})")
        
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>AI Financial Analyst | Built with LangGraph, Groq, and Streamlit</p>
    <p>⚠️ This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
