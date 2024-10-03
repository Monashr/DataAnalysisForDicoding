import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Air Quality Dashboard")

def load_data():
    return pd.read_csv("dashboard/main_data.csv")

all_data = load_data()

# Ensure 'year' is an integer
all_data['year'] = all_data['year'].astype(int)

unique_stations = all_data['station'].unique()
stations_with_all_option = ["All"] + list(unique_stations)

stations = st.selectbox('Select Station', stations_with_all_option, key="station")

# Slider for year selection
start_year, end_year = st.slider('Select Year Range', 
                                   min_value=int(all_data['year'].min()), 
                                   max_value=int(all_data['year'].max()), 
                                   value=(2013, 2017))

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

if stations == "All":
    pollutant_data = all_data.groupby(['year'])[pollutants].mean()
    station_data = pollutant_data.loc[start_year:end_year].reset_index()
else:
    pollutant_data = all_data.groupby(['station', 'year'])[pollutants].mean()
    station_data = pollutant_data.loc[stations].loc[start_year:end_year].reset_index()

st.dataframe(station_data, use_container_width=True)


st.title("Rainfall vs Pollutant Concentration")
stations_for_RAIN = st.selectbox('Select Station', all_data['station'].unique(), key="station_rain")

num_columns = 3
columns = st.columns(num_columns)

st.write("Select Pollutants:")

selected_pollutants = []
for i, pollutant in enumerate(pollutants):
    col = columns[i % num_columns]
    if col.checkbox(pollutant, key=f"pollutant_{pollutant}"):
        selected_pollutants.append(pollutant)

if stations_for_RAIN:
    filtered_df = all_data[all_data['station'] == stations_for_RAIN]
else:
    filtered_df = all_data


for pollutant in selected_pollutants:
    if not filtered_df.empty:
        st.subheader(f"Rainfall vs {pollutant}")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=filtered_df['RAIN'], y=filtered_df[pollutant])
        plt.title(f"Rainfall vs {pollutant}")
        plt.xlabel('Rainfall (mm)')
        plt.ylabel(f'{pollutant} Concentration')
        
        st.pyplot(plt)
        plt.clf()
    else:
        st.write("No data available for the selected station.")

st.title("Correlation Heat Map")

correlations = filtered_df[['RAIN'] + pollutants].corr()
rain_pollutant_corr = correlations['RAIN'][pollutants]

plt.figure(figsize=(10, 6))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")

st.pyplot(plt)
plt.clf()  

st.title("Average Pollutant Concentrations")

all_data['month'] = all_data['month'].astype(int)

pollutants_with_co = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
pollutants_without_co = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']

stations_for_mean = st.selectbox('Select Station', stations_with_all_option, key="station_for_mean")

if stations_for_mean == "All":
    filtered_df = all_data
else:
    filtered_df = all_data[all_data['station'] == stations_for_mean]

plt.figure(figsize=(12, 8))
for pollutant in pollutants_with_co:
    monthly_avg = filtered_df.groupby('month')[pollutant].mean() 
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label=pollutant)

plt.title('Average Monthly Concentrations of Pollutants (Including CO)')
plt.xlabel('Month')
plt.ylabel('Average Concentration')
plt.xticks(range(1, 13)) 
plt.grid()
plt.legend(title='Pollutants')
plt.tight_layout()

st.pyplot(plt)

plt.clf()

plt.figure(figsize=(12, 8))
for pollutant in pollutants_without_co:
    monthly_avg = filtered_df.groupby('month')[pollutant].mean() 
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label=pollutant)

plt.title('Average Monthly Concentrations of Pollutants (Excluding CO)')
plt.xlabel('Month')
plt.ylabel('Average Concentration')
plt.xticks(range(1, 13))
plt.grid()
plt.legend(title='Pollutants')
plt.tight_layout()


st.pyplot(plt)
plt.clf()