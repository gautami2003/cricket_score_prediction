# IPL Score Prediction

This project uses a machine learning model to predict the score of an IPL (Indian Premier League) match based on various input parameters. The project is built using Python, with the Streamlit library to create a user-friendly web interface. The model is trained on historical IPL data and allows users to predict the score of a match or forecast future match scores.

## Features

- **Score Prediction:** Predict the score of an ongoing IPL match by selecting the venue, batting team, bowling team, striker, and bowler.
- **Future Score Predictions:** Generate and visualize predictions for IPL matches up to 5 years into the future.
- **User-Friendly Interface:** A simple and interactive interface built using Streamlit for easy user interaction.

## Requirements

To run this project, you'll need the following Python libraries:
        
    pip install pandas numpy scikit-learn keras tensorflow streamlit matplotlib


## Data

The data used for training the model is provided in a CSV file (`ipl_data.csv`). The data includes various features like venue, batting team, bowling team, batsman, bowler, and the total score.

## Model

- The model is a neural network built using Keras and TensorFlow.
- It includes two hidden layers with 512 and 216 neurons, respectively, and uses ReLU activation functions.
- The output layer is a single neuron with a linear activation function, predicting the final score.
- The model is compiled using the Adam optimizer and the Huber loss function.

## How to Run

1. *Clone the Repository:*

       git clone https://github.com/yourusername/ipl-score-prediction.git
        cd ipl-score-prediction

2. *Run the Application:*

       streamlit run app.py


4. *Use the Interface:*
   - Use the sidebar to select input parameters.
   - Click the "Predict Score" button to see the predicted score for a match.
   - Click the "Predict Future Scores" button to generate predictions for future IPL matches.

## Functions Overview

### `predict_score()`
- Encodes the user-selected inputs and predicts the match score using the trained model.

### `predict_future_scores()`
- Generates synthetic data for future matches and predicts their scores.
- Displays a plot showing predicted scores over time.

## Future Improvements

- Improve model accuracy by experimenting with different architectures and feature engineering.
- Include more input features such as player statistics and match conditions.
- Extend the application to include match result predictions.
