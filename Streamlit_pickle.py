import streamlit as st
import numpy as np
import pickle

# Streamlit app interface
st.title('Player Rating Predictor')

# Create the file uploader widget
uploaded_file = st.file_uploader("Upload DecisionTreeRegressor.pkl", type="pkl")

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load the model from the uploaded file
        with open(uploaded_file, 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
        
    except pickle.UnpicklingError:
        st.error("Error in unpickling the file. The file might be corrupted.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.warning('Please upload a .pkl file.')
    
    
    # Define the prediction function
def predict_player_rating(features,model):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction[0]

# Streamlit app interface
st.title('Player Rating Prediction')

# Input features from user
features_list = [ 'potential', 'value_eur', 
    'wage_eur', 'age', 'league_level', 'weak_foot', 'skill_moves', 
    'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 
    'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 
    'attacking_heading_accuracy', 'attacking_short_passing', 
    'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 
    'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 
    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
    'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 
    'mentality_positioning', 'mentality_vision', 'mentality_penalties', 
    'mentality_composure', 'defending_marking_awareness', 
    'defending_standing_tackle', 'defending_sliding_tackle', 
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
    'goalkeeping_positioning', 'goalkeeping_reflexes','cat__player_positions', 'cat__nationality_name', 'cat__preferred_foot', 
    'cat__work_rate', 'cat__ls', 'cat__st', 'cat__rs', 'cat__lw', 'cat__lf', 
    'cat__cf', 'cat__rf', 'cat__rw', 'cat__lam', 'cat__cam', 'cat__ram', 
    'cat__lm', 'cat__lcm', 'cat__cm', 'cat__rcm', 'cat__rm', 'cat__lwb', 
    'cat__ldm', 'cat__cdm', 'cat__rdm', 'cat__rwb', 'cat__lb', 'cat__lcb', 
    'cat__cb', 'cat__rcb', 'cat__rb', 'cat__gk'
]
input_data = []

for feature in features_list:
    value = st.number_input(f'Enter {feature}', value=0.0)
    input_data.append(value)

if st.button('Predict Rating'):
    rating = predict_player_rating(input_data,model)
    st.write(f'Predicted Player Rating: {rating}')
