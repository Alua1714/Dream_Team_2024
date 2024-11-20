# Chicago Housing Price Prediction: Multi-Model Approach with Interactive Mapping

## Project Overview
This project focuses on developing a sophisticated housing price prediction system for Chicago, combining machine learning with interactive visualization tools. At its core, the system utilizes an ensemble of three specialized Random Forest models, each designed to analyze different aspects of the housing market. The first model processes traditional property characteristics, the second focuses on location-specific features, and the third analyzes temporal market trends. These models work in concert, with their predictions combined through a weighted averaging system to produce final price estimates.

## Data Processing and Feature Engineering
The project implements extensive data preprocessing, with a particular emphasis on location-based feature engineering. Using the Google Maps API, all property addresses are automatically geocoded to obtain precise latitude and longitude coordinates. This geographical data is then enriched by calculating distances to key Chicago landmarks, public transportation hubs, and important business districts. The preprocessing pipeline also generates additional features such as neighborhood demographics and historical price trends, providing the models with rich contextual information for more accurate predictions.

## Interactive Visualization System
The frontend interface centers around an embedded Google Maps implementation that offers an intuitive way to explore property data. Users can interact with a dynamic map displaying properties as color-coded markers based on their price ranges. When selecting a property, an information window appears showing the predicted price, key characteristics, and a street view image of the location. This real-time visualization system makes complex property data easily accessible, allowing users to explore housing options while understanding pricing patterns across different Chicago neighborhoods.

![datathon container aed cat - Google Chrome 17_11_2024 12_49_50](https://github.com/user-attachments/assets/a0d51763-b07b-45ab-9583-e0c37b8ecdb3)

![datathon container aed cat - Google Chrome 17_11_2024 12_47_37](https://github.com/user-attachments/assets/1519fe2c-1df5-4c0b-ab47-0953a12ed6d6)
