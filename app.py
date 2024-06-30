import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
import random
from datetime import datetime, timedelta

# Load data
ipl = pd.read_csv('ipl_data.csv')

# Dropping certain features
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)
X = df.drop(['total'], axis=1)
y = df['total']

# Label Encoding
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Compile the model
huber_loss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam', loss=huber_loss)
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))

# Streamlit UI
st.title("IPL Score Prediction")

st.sidebar.header("User Input Parameters")
venue = st.sidebar.selectbox('Select Venue', df['venue'].unique().tolist())
batting_team = st.sidebar.selectbox('Select Batting Team', df['bat_team'].unique().tolist())
bowling_team = st.sidebar.selectbox('Select Bowling Team', df['bowl_team'].unique().tolist())
striker = st.sidebar.selectbox('Select Striker', df['batsman'].unique().tolist())
bowler = st.sidebar.selectbox('Select Bowler', df['bowler'].unique().tolist())

def predict_score():
    # Encode the selected inputs
    encoded_venue = venue_encoder.transform([venue])
    encoded_batting_team = batting_team_encoder.transform([batting_team])
    encoded_bowling_team = bowling_team_encoder.transform([bowling_team])
    encoded_striker = striker_encoder.transform([striker])
    encoded_bowler = bowler_encoder.transform([bowler])

    # Create input array
    input_data = np.array([encoded_venue, encoded_batting_team, encoded_bowling_team, encoded_striker, encoded_bowler]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_score = model.predict(input_data_scaled)
    predicted_score = int(predicted_score[0, 0])

    return predicted_score

def predict_future_scores():
    # Generate synthetic data for the next 3 years
    num_future_matches_per_year = 100  # Number of future matches to predict per year
    num_future_years = 3
    num_future_matches = num_future_matches_per_year * num_future_years

    # Randomly sample from existing categorical values
    future_data = {
        'venue': random.choices(df['venue'].unique(), k=num_future_matches),
        'bat_team': random.choices(df['bat_team'].unique(), k=num_future_matches),
        'bowl_team': random.choices(df['bowl_team'].unique(), k=num_future_matches),
        'batsman': random.choices(df['batsman'].unique(), k=num_future_matches),
        'bowler': random.c
