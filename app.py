import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration must be first!
st.set_page_config(layout='wide', page_title='Plotly Datasets Dashboard')

st.title('Plotly Built-in Datasets Dashboard')

# ===== STEP 1: LOAD & CACHE =====
@st.cache_data
def load_tips():
    return px.data.tips()

@st.cache_data
def load_iris():
    return px.data.iris()

@st.cache_data
def load_gapminder():
    return px.data.gapminder()

# ===== SIDEBAR NAVIGATION =====
st.sidebar.header('Navigation')
dataset = st.sidebar.radio('Select Dataset', ['Tips', 'Iris', 'Gapminder'])

# ===== TIPS DATASET =====
if dataset == 'Tips':
    st.header('Tips Dashboard')
    
    # STEP 1: Load & Cache
    df = load_tips()
    
    # STEP 2: Sidebar Filters
    st.sidebar.subheader('Filters')
    selected_day = st.sidebar.multiselect('Day', df['day'].unique(),
                                          default=df['day'].unique())
    selected_time = st.sidebar.selectbox('Time', ['All', 'Lunch', 'Dinner'])
    selected_sex = st.sidebar.selectbox('Sex', ['All'] + df['sex'].unique().tolist())
    
    # STEP 3: Apply Filters
    filtered = df[df['day'].isin(selected_day)]
    if selected_time != 'All':
        filtered = filtered[filtered['time'] == selected_time]
    if selected_sex != 'All':
        filtered = filtered[filtered['sex'] == selected_sex]
    
    # STEP 4: Show Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Bill', f'${filtered["total_bill"].sum():.0f}')
    col2.metric('Avg Tip', f'${filtered["tip"].mean():.2f}')
    col3.metric('Avg Bill', f'${filtered["total_bill"].mean():.2f}')
    col4.metric('Records', len(filtered))
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Scatter: Bill vs Tip')
        fig = px.scatter(filtered, x='total_bill', y='tip', color='time', size='size')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Box: Day vs Bill')
        fig = px.box(filtered, x='day', y='total_bill', color='time')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Data Preview')
    st.dataframe(filtered.head(10))

# ===== IRIS DATASET =====
elif dataset == 'Iris':
    st.header('Iris Dataset Analysis')
    
    # STEP 1: Load & Cache
    df = load_iris()
    
    # STEP 2: Sidebar Filters
    st.sidebar.subheader('Filters')
    selected_species = st.sidebar.multiselect('Species', df['species'].unique(),
                                              default=df['species'].unique())
    
    # STEP 3: Apply Filters
    filtered = df[df['species'].isin(selected_species)]
    
    # STEP 4: Show Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Records', len(filtered))
    col2.metric('Species Count', filtered['species'].nunique())
    col3.metric('Avg Petal Length', f'{filtered["petal_length"].mean():.2f}')
    col4.metric('Avg Sepal Width', f'{filtered["sepal_width"].mean():.2f}')
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Scatter: Petal')
        fig = px.scatter(filtered, x='petal_length', y='petal_width', color='species', size='sepal_length')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Histogram: Sepal Length')
        fig = px.histogram(filtered, x='sepal_length', nbins=30, color='species')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('3D Scatter Plot')
    fig = px.scatter_3d(filtered, x='sepal_length', y='sepal_width', z='petal_length', color='species')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Data Preview')
    st.dataframe(filtered.head(10))

# ===== GAPMINDER DATASET =====
elif dataset == 'Gapminder':
    st.header('Gapminder Dataset')
    
    # STEP 1: Load & Cache
    df = load_gapminder()
    
    # STEP 2: Sidebar Filters
    st.sidebar.subheader('Filters')
    selected_year = st.sidebar.slider('Select Year', 
                                      int(df['year'].min()), 
                                      int(df['year'].max()))
    selected_continent = st.sidebar.multiselect('Continent', 
                                                df['continent'].unique(),
                                                default=df['continent'].unique())
    
    # STEP 3: Apply Filters
    filtered = df[(df['year'] == selected_year) & 
                  (df['continent'].isin(selected_continent))]
    
    # STEP 4: Show Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Countries', filtered['country'].nunique())
    col2.metric('Avg Life Exp', f'{filtered["lifeExp"].mean():.1f}')
    col3.metric('Avg GDP', f'${filtered["gdpPercap"].mean():.0f}')
    col4.metric('Total Pop', f'{filtered["pop"].sum()/1e9:.2f}B')
    
    st.subheader(f'Life Expectancy vs GDP ({selected_year})')
    fig = px.scatter(filtered, x='gdpPercap', y='lifeExp', size='pop', color='continent',
                     hover_name='country', log_x=True, size_max=60)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Data Preview')
    st.dataframe(filtered.head(10))

