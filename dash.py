import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg as AR
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

# 1. Page Configuration
st.set_page_config(page_title="Microsoft 10-Year Returns Dashboard", layout="wide")





# --- MOCK DATA GENERATION (Replace with your actual df) ---


@st.cache_data
def load_my_data():
    # This only runs once! Subsequent reruns pull from memory.
    df = pd.read_csv("MSFT CSV updated.csv")
    df.columns = ["Perm No.","Date","Ticker","Company Name","Dividend Amt","Price","Volume","Stock Returns","BID","ASK","Return No Div","Index Returns"]
    df['Date'] = pd.to_datetime(df['Date']) # Pre-process once
    return df

df_data = load_my_data()

st.sidebar.header("Filter Data")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df_data["Date"].min(), df_data["Date"].max()), # Default to showing everything
    min_value=df_data["Date"].min(),
    max_value=df_data["Date"].max()
)

if len(date_range) == 2:
    start_date, end_date = date_range
    # Convert back to datetime for comparison
    mask = (df_data['Date'].dt.date >= start_date) & (df_data['Date'].dt.date <= end_date)
    df = df_data.loc[mask]
else:
    # Fallback if only one date is clicked during the selection process
    df = df_data

st.subheader(f"Analysis from {date_range[0]} to {date_range[1] if len(date_range) > 1 else ''}")
vol_window = st.sidebar.slider("Volatility Window (Days)", 5, 90, 60)
show_normal = st.sidebar.checkbox("Overlay Normal Curve", value=True)

# 3. Main Dashboard Header & KPIs
st.title("📈 Stock Analysis Dashboard")

# col1, col2, col3, col4, col5 = st.columns(5)
# col1.metric("Avg Stock Return", f"{df['Stock Returns'].mean():.4f}", "0.01%")
# col2.metric("Stock Volatility", f"{df['Stock Returns'].std():.4f}")
# col1.metric("Avg Index Return", f"{df['Index Returns'].mean():.4f}", "0.01%")
# col2.metric("Index Volatility", f"{df['Index Returns'].std():.4f}")
# col3.metric("Data Points", len(df))

# 4. Tabbed Layout for Graphs
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["Microsoft Price Time Series", "Time Series", "Distribution of Returns","Box-Plot Comparison", "MSFT vs. Index Scatterplot","Rolling Volatility","Auto Correlation Function (ACF)"])

#-----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Stock Returns"], mode='lines', zorder=1))
#fig.add_trace(go.Scatter(x=actvsfit_df_clean["Date"], y=actvsfit_df_clean["Predicted Stock"], mode='lines', zorder=2))

fig.update_layout(
    title="Actual Stock Returns Time Series",
    xaxis_title="Date",
    yaxis_title="Returns",
    showlegend =False, height =600)



fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["Date"], y=df["Index Returns"], mode='lines' , zorder=1, line = dict(color = 'red')))

fig2.update_layout(
    title="Actual Index Returns Time Series",
    xaxis_title="Date",
    yaxis_title="Returns",
    showlegend =False, height =600)



fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df["Date"], y=df["Price"], mode='lines' , zorder=1))

fig3.update_layout(
    title="Stock Price Movement Over Time",
    xaxis_title="Date",
    yaxis_title="Price",
    showlegend =False, height =600)



fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df["Date"], y=df["Stock Returns"], mode='lines', zorder=1, name = "Stock Returns"))
fig4.add_trace(go.Scatter(x=df["Date"], y=df["Index Returns"], mode='lines' , zorder=2, name = "Index Returns", opacity = 0.8, line = dict(color = 'red')))

fig4.update_layout(
    title="Actual Stock Returns vs Index Returns",
    xaxis_title="Date",
    yaxis_title="Returns",
    showlegend =True, height =600)



mu, std = df["Stock Returns"].mean(), df["Stock Returns"].std()
x_axis = np.linspace(df["Stock Returns"].min(), df["Stock Returns"].max(), 100)
y_axis = norm.pdf(x_axis, mu, std)


fig5 = go.Figure()
fig5.add_trace(go.Histogram(
    x=df["Stock Returns"],
    name="Stock Returns Distribution",
    histnorm ="probability density",
    marker_color='rgba(0, 100, 250, 0.6)', # Use rgba for transparency
    nbinsx=60                              # Suggested number of bins
))
if show_normal is True:
    
    fig5.add_trace(go.Scatter(
        x=x_axis,
        y=y_axis,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=3)))

fig5.update_layout(
    title="Distribution of Stock Returns",
    xaxis_title="Return Value",
    yaxis_title="Frequency",
    showlegend = False,
    bargap=0.1 # Adds a small gap between bars for better readability
)




mu_2, std_2 = df["Index Returns"].mean(), df["Index Returns"].std()
x_axis_2 = np.linspace(df["Index Returns"].min(), df["Index Returns"].max(), 100)
y_axis_2 = norm.pdf(x_axis_2, mu_2, std_2)


fig6 = go.Figure()
fig6.add_trace(go.Histogram(
    x=df["Index Returns"],
    name="Index Returns Distribution",
    histnorm ="probability density",
    marker_color='rgba(0, 100, 250, 0.6)', # Use rgba for transparency
    nbinsx=60                              # Suggested number of bins
))

if show_normal is True:
    fig6.add_trace(go.Scatter(
        x=x_axis_2,
        y=y_axis_2,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=3)))

fig6.update_layout(
    title="Distribution of Index Returns",
    xaxis_title="Return Value",
    yaxis_title="Frequency",
    showlegend= False,
    bargap=0.1 # Adds a small gap between bars for better readability
)


fig7 = go.Figure()
fig7.add_trace(go.Box(
    y=df["Stock Returns"],
    name="Stock",
    marker_color='royalblue',
    boxpoints= 'suspectedoutliers'# Options: 'all', 'outliers', 'suspectedoutliers', or False
))


fig7.add_trace(go.Box(
    y=df["Index Returns"],
    name="Index",
    marker_color='lightseagreen',
    boxpoints= 'outliers' # Options: 'all', 'outliers', 'suspectedoutliers', or False
))

fig7.update_layout(title="Comparison of Returns" , height =1000)



fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=df["Index Returns"], y=df["Stock Returns"], mode='markers', zorder=1))
fig8.update_layout(
    title="MSFT vs Index Returns",
    xaxis_title="Index Returns",
    yaxis_title="MSFT Returns",
    showlegend =False
)




df['Vol Stock'] = df['Stock Returns'].rolling(window=vol_window).std()
df['Vol Index'] = df['Index Returns'].rolling(window=vol_window).std()

fig9 = go.Figure()

fig9.add_trace(go.Scatter(
    x=df["Date"], 
    y=df["Vol Stock"], 
    mode='lines',
    name= f'{vol_window}-Day Rolling Vol Stock',
    line=dict(color='firebrick', width=2)
))

fig9.update_layout(
    title= f"{vol_window}-Day Rolling Volatility (Stock)",
    xaxis_title="Date",
    yaxis_title="Standard Deviation",
    template="plotly_white"
)

fig10 = go.Figure()

fig10.add_trace(go.Scatter(
    x=df["Date"], 
    y=df["Vol Index"], 
    mode='lines',
    name= f"{vol_window}-Day Rolling Vol Index",
    line=dict(color='firebrick', width=2)
))

fig10.update_layout(
    title= f"{vol_window}-Day Rolling Volatility (Index)",
    xaxis_title="Date",
    yaxis_title="Standard Deviation",
    template="plotly_white"
)

# 1. Calculate ACF values for lags 0 to 5
# We dropna() because ACF cannot handle missing values
returns_data = df["Stock Returns"].dropna()
acf_values = acf(returns_data, nlags=5)

# 2. Slice to get only lags 1-5
lags = list(range(1, 6))
values = acf_values[1:] 

# 3. Create the "Lollipop" Chart (Standard for ACF)
fig11 = go.Figure()

# Add the vertical stems
for i, val in enumerate(values):
    fig11.add_shape(type='line', x0=lags[i], y0=0, x1=lags[i], y1=val,
                  line=dict(color='royalblue', width=2))

# Add the markers at the top
fig11.add_trace(go.Scatter(
    x=lags, 
    y=values, 
    mode='markers',
    marker=dict(size=12, color='royalblue'),
    name="ACF"
))

# 4. Add Significance Thresholds (Optional)
# A common rule of thumb is 1.96 / sqrt(n)
n = len(returns_data)
threshold = 1.96 / (n**0.5)

fig11.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="95% CI")
fig11.add_hline(y=-threshold, line_dash="dash", line_color="red")

fig11.update_layout(
    title="Autocorrelation (ACF) Lags 1-5 (Stock)",
    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
    xaxis_title="Lag",
    yaxis_title="Autocorrelation",
    height=400
)



# 1. Calculate ACF values for lags 0 to 5
# We dropna() because ACF cannot handle missing values
returns_data_index = df["Index Returns"].dropna()
acf_values_index = acf(returns_data_index, nlags=5)

# 2. Slice to get only lags 1-5
lags_1 = list(range(1, 6))
values_1 = acf_values_index[1:] 

# 3. Create the "Lollipop" Chart (Standard for ACF)
fig12 = go.Figure()

# Add the vertical stems
for i_1, val_1 in enumerate(values_1):
    fig12.add_shape(type='line', x0=lags_1[i_1], y0=0, x1=lags_1[i_1], y1=val_1,
                  line=dict(color='royalblue', width=2))

# Add the markers at the top
fig12.add_trace(go.Scatter(
    x=lags_1, 
    y=values_1, 
    mode='markers',
    marker=dict(size=12, color='royalblue'),
    name="ACF"
))

# 4. Add Significance Thresholds (Optional)
# A common rule of thumb is 1.96 / sqrt(n)
n_1 = len(returns_data_index)
threshold_1 = 1.96 / (n_1**0.5)

fig12.add_hline(y=threshold_1, line_dash="dash", line_color="red", annotation_text="95% CI")
fig12.add_hline(y=-threshold_1, line_dash="dash", line_color="red")

fig12.update_layout(
    title="Autocorrelation (ACF) Lags 1-5 (Index)",
    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
    xaxis_title="Lag",
    yaxis_title="Autocorrelation",
    height=400
)


#----------------

with tab1:
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Time Series of Returns")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
            
with tab3:
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)

with tab4:
    st.plotly_chart(fig7, use_container_width=True)

with tab5:
    st.plotly_chart(fig8, use_container_width=True)


with tab6:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig9, use_container_width=True)
    with col2:
        st.plotly_chart(fig10, use_container_width=True)

with tab7:
    st.plotly_chart(fig11, use_container_width=True)
    st.plotly_chart(fig12, use_container_width=True)

# 5. Raw Data Expander
with st.expander("View Raw Data"):
    st.dataframe(df, use_container_width=True)