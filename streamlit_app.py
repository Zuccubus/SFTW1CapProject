import streamlit as st
from RentalModel import *

# title and subtitles
st.title("NY Rental Property Type Predictor")
st.sidebar.title("Predictive Functions")
st.sidebar.write("(Fill in Info First)")

# inputs
# neighbourhoods limited to four types to
neighbourhood_select_box = st.selectbox("Neighbourhood: ",
                                        ("Brooklyn", "Manhattan", "Queens", "Harlem", "Bedford-Stuyvesant")
                                        )
lat = st.slider("Latitude", 40.50000, 40.90000, step=0.00001, format="%0.5f")
long = st.slider("Longitude", -74.20000, -73.70000, step=0.00001, format="%0.5f")
dOc_2019 = st.text_input("Number of Days Property was Occupied in 2019: ")
min_nights = st.text_input("Minimum Nights to Stay at Property: ")
num_rev = st.text_input("Number of Reviews: ")
monthly_num_rev = st.text_input("Number of Reviews per Month: ")
availability = st.text_input("Property Rental Availability During 2020: ")
price = st.number_input("Property Rental Price: ", 0, step=1, )

# neighbourhood reference values
neighbourhoods = {
    "Brooklyn": 0,
    "Manhattan": 1,
    "Queens": 2,
    "Harlem": 3,
    "Bedford-Stuyvesant": 4
}

# changing object to number value
neighbourhood = neighbourhoods[neighbourhood_select_box]
# predicting using best method
property_info = (neighbourhood, lat, long, dOc_2019,min_nights,
                 num_rev, monthly_num_rev, availability, price)

# setting up sidebar checkbox to access prediction results
if st.sidebar.checkbox("Show Predicted Property Type"):
    room_predict = best_model.predict([property_info])

    # turning number result into words
    if room_predict == 0:
        property_room_type = "Entire house/apt"
    elif room_predict == 1:
        property_room_type = "Hotel room"
    elif room_predict == 2:
        property_room_type = "Private room"
    else:
        property_room_type = "Shared room"

    # outputting result
    st.sidebar.write(property_room_type)
    # outputting prediction accuracy
    st.sidebar.write(f"Accuracy of prediction: {percentage_acc:,.2f}%")

# checkbox to show data view of inputted info
if st.sidebar.checkbox("Show Property Info"):
    st.sidebar.write(property_info)



