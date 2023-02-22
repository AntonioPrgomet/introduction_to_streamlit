
# =============================================================================
# Demonstration of how to make a Streamlit application. 
# Created by Antonio Prgomet.
# www.linkedin.com/in/antonioprgomet .
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# =============================================================================
# Loading Data and Modelling it. 
# =============================================================================

# Put your own path for reading the Excel file. 
modelling_data = pd.read_excel(r'C:\Users\Antonio Prgomet\Documents\YouTube\python\streamlit_demonstration\modelling_data.xlsx')

x = np.array(modelling_data[["x"]])
y = np.array(modelling_data[["y"]])

# Plotting the data. 
fig_data, ax_data = plt.subplots(figsize=(8,4))
ax_data.set_title('Data')
ax_data.scatter(modelling_data["x"], modelling_data["y"])
ax_data.set_xlabel('x')
ax_data.set_ylabel('y')

# Initializing and fitting our Linear Regression model.
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Creating new data that we will predict with our fitted model. 
x_new = np.linspace(0, 2, 20).reshape(-1, 1)
y_pred_lr = lin_reg.predict(x_new)

# Plotting the data and our model predictions.
fig_model, ax_model = plt.subplots()
ax_model.set_title("Linear Regression Model")
ax_model.scatter(x, y, label = "Data")
ax_model.plot(x_new, y_pred_lr, 'r', label = 'Predictions')
ax_model.set_xlabel("x")
ax_model.set_ylabel("y")
ax_model.legend()

# =============================================================================
# Creating the streamlit application. #
# =============================================================================

# Creating a navigation menu with three different sections the user can choose. 
nav = st.sidebar.radio("Navigation Menu",["Purpose", "Data & Modelling", "Next Step"])

if nav == "Purpose":
    st.title("Streamlit - Example Demonstration")
    st.header("Purpose")
    st.write("""The purpose of this example demonstration is to give you a fast 
             overview of Streamlit where details are omitted. After watching the video
             read the section "Get Started" from the excellent documentation
             available at: [https://docs.streamlit.io/library/get-started](https://docs.streamlit.io/library/get-started)
             and you will know everything you need to create your first 
             Streamlit app. """)
    st.write("Happy Coding!")
    st.write("Antonio Prgomet")
    st.write("[https://www.linkedin.com/in/antonioprgomet/](https://www.linkedin.com/in/antonioprgomet/)")

if nav == "Data & Modelling":
    st.title("Data & Machine Learning Modelling")
    st.write('In this section we will look at the data and also modell it by using Linear Regression.')
             
    st.header("Data")
    st.subheader("Scatterplot of the Data")
    st.pyplot(fig_data)
    
    st.subheader("Raw Data")
    st.write("If you want to see the raw data, check the box below.")
    if st.checkbox('Show raw data'):
        st.write(modelling_data)
        
    st.header("Machine Learning Model - Linear Regression")
    st.subheader("Visualizing the Model")
    st.pyplot(fig_model)
    
    st.subheader("Prediction Interface")
    val = st.number_input("Enter the x value you want to predict.", step = 0.25)
    val = np.array(val).reshape(1, -1)
    prediction = lin_reg.predict(val)
    if st.button("Predict"):
        st.success(f"The predicted y value is: {prediction}")
    
if nav == "Next Step":
    st.title("Read the Documentation")
    st.write("""Read the section "Get Started" from the documentation
    available at: [https://docs.streamlit.io/library/get-started](https://docs.streamlit.io/library/get-started) .""")
    
    
    

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    