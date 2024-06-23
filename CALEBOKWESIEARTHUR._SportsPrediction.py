#importing relevant libraries 
import pandas as pd
import numpy as np 
import pickle as pkl
import category_encoders as ce

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#importing models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#evaluation metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Fine tuninng the model 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score

# Importing streamlit
import streamlit as st



### Data Preprocessing 


#Reading the data
fifaDF = pd.read_csv('male_players (legacy).csv', low_memory = False)

fifaDF.head()

#Selecting the columns that have less than 30% null values 
greater_than = []
less_than = []
for i in fifaDF.columns:
    if((fifaDF[i].isnull().sum())< (0.3 * (fifaDF.shape[0]))):
        greater_than.append(i)
    else:
        less_than.append(i)

#Reassigning the data frame to the new dataframe 
fifaDF = fifaDF[greater_than]
fifaDF.info()

#seperating the numeric and non numeric features 
numeric_data = fifaDF.select_dtypes(include = np.number)
non_numeric = fifaDF.select_dtypes(include = ['object'])

##### Dealing with numeric data

#correlation between features listed and overall.
corr_matrix = numeric_data.corr()

corr_matrix = corr_matrix["overall"].sort_values(ascending=False)

# Identify features with correlation less than 0.4 with the target
high_corr_features = corr_matrix[abs(corr_matrix) > 0.4 ].index

low_corr_features = corr_matrix[abs(corr_matrix) < 0.4 ]
print(f'The low correlation features are:\n{low_corr_features}')

columns_to_drop = ['player_id','nationality_id','league_id','club_team_id','club_jersey_number','fifa_version','weight_kg','movement_balance','club_contract_valid_until_year','height_cm','fifa_update' ]

#droping the irrelevant columns from the numeric dataFrame
numeric_data.drop(columns = columns_to_drop, axis = 1 , inplace = True)

numeric_data.info()

#multivariate imputation 
imp = IterativeImputer(max_iter = 10, random_state = 0)
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns = numeric_data.columns)#this line learns the data and imputes the missing features 

numeric_data

##### Dealing with Non-numeric features 

non_numeric = fifaDF.select_dtypes(include = ['object'])
columns_to_drop = ['player_url','fifa_update_date','player_face_url','dob','short_name', 'long_name','league_name','club_name','club_position','club_joined_date','real_face','body_type']

#Dropping the irrelevant columns 
non_numeric.drop(columns = columns_to_drop, axis = 1 , inplace = True)

#using pipelines 
cat_pipe = Pipeline([
 ("impute", SimpleImputer(strategy="most_frequent")),
])

full_pipe = ColumnTransformer([
    ("cat", cat_pipe,make_column_selector(dtype_include = object))
])
piped = full_pipe.fit_transform(non_numeric)

non_numeric = pd.DataFrame(data = piped, columns = full_pipe.get_feature_names_out())

#using binary encoding 
encoder = ce.BinaryEncoder(cols = non_numeric.columns)

non_numeric = encoder.fit_transform(non_numeric)

non_numeric

fifaDF = pd.concat([numeric_data, non_numeric], axis= 1)

y = fifaDF['overall']

##### Scaling the data

X = fifaDF.drop('overall', axis = 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

##### Training and testing the model 

 Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size = 0.2,random_state = 42, stratify = y)

#Linear Regression
lr = LinearRegression()
lr.fit(Xtrain, Ytrain)

#Random Forest Regression 
rf = RandomForestRegressor()
rf.fit(Xtrain, Ytrain)

#Decision Tree Regression 
dt = DecisionTreeRegressor()
dt.fit(Xtrain, Ytrain)


 for model in [lr, rf, dt]:
    pkl.dump(model,open( model.__class__.__name__ +'V2.actual' +'.pkl','wb'))

#Defining a function for evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2




##### Fine-tuning the model

# Perform GridSearchCV on Descision tree
grid_search = GridSearchCV(estimator = dt, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(Xtrain, Ytrain)

# Get the best parameters and model
#best_params = grid_search.best_params_
#best_model = grid_search.best_estimator_

# Measure performance of the best model
best_predictions = best_model.predict(Xtest)
best_mse = mean_squared_error(Ytest, best_predictions)
print(f'Best Mean Squared Error: {best_mse}')

##### Cross-Validation 

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf, Xtrain, Ytrain, cv=kf, scoring='neg_mean_squared_error')

# Print the cross-validation scores and their mean
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")



# Evaluation
for model in [lr, rf, dt]:
    mse, mae, r2 = evaluate_model(model, Xtest, Ytest)
    print(f"Model: {model.__class__.__name__}, MSE: {mse}, MAE: {mae}, R2: {r2}")


#creating a function 
def process_file(file_path):
    
    # Load the dataset
    data = pd.read_csv(file_path)

    # Select columns with less than 30% null values
    greater_than = []
    less_than = []
    for i in data.columns:
        if (data[i].isnull().sum() < (0.3 * (data.shape[0]))):
            greater_than.append(i)
        else:
            less_than.append(i)

    data = data[greater_than]

    # Split the data into numeric and non-numeric
    numeric_data = data.select_dtypes(include=np.number)
    non_numeric = data.select_dtypes(include=['object'])

    # Correlation matrix to find important features 
    corr_matrix = numeric_data.corr()
    corr_matrix = corr_matrix["overall"].sort_values(ascending=False)
    high_corr_features = corr_matrix[abs(corr_matrix) > 0.4].index
    low_corr_features = corr_matrix[abs(corr_matrix) < 0.4]

    # Drop specified columns
    columns_to_drop = ['player_id','sofifa_id' ,'nationality_id', 'league_id', 'club_team_id', 'club_jersey_number', 
                       'fifa_version', 'weight_kg', 'movement_balance', 'club_contract_valid_until_year', 
                       'height_cm', 'fifa_update','nation_team_id']
    numeric_data.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    # Impute missing values in numeric data
    imp = IterativeImputer(max_iter=10, random_state=0)
    numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns=numeric_data.columns)

    # Drop specified non-numeric columns
    columns_to_drop = ['player_url', 'fifa_update_date', 'player_face_url', 'dob', 'short_name', 
                       'long_name', 'league_name', 'club_name', 'club_position', 'club_joined_date', 
                       'real_face', 'body_type']
    non_numeric.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    # Define a simple imputer for categorical data
    cat_pipe = Pipeline([
         ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    
    # Apply the imputer to non-numeric data
    full_pipe = ColumnTransformer([("cat", cat_pipe, make_column_selector(dtype_include=object))])

    non_numeric = pd.DataFrame(full_pipe.fit_transform(non_numeric), columns=non_numeric.columns)

    
    # Drop additional specified columns
    additional_columns_to_drop = ['sofifa_id','player_url','player_face_url','dob','short_name', 'long_name',
                                  'league_name','club_team_id','club_jersey_number','club_loaned_from',
                                  'nationality_id','nation_team_id','nation_jersey_number','real_face',
                                  'body_type','release_clause_eur','player_tags','player_traits',
                                  'mentality_composure','nation_position', 'goalkeeping_speed','club_joined',
                                  'club_contract_valid_until']
    non_numeric.drop(columns=additional_columns_to_drop, axis=1, inplace=True, errors='ignore')
   
    # Encode non-numeric data
    encoder = ce.BinaryEncoder(cols=non_numeric.columns)
    non_numeric = encoder.fit_transform(non_numeric)

    # Concatenate the data into a single DataFrame
    processedDF = pd.concat([numeric_data, non_numeric], axis=1)

    return processedDF


players_22 = process_file('players_22.csv')

players_22.to_csv('New_data.csv', index=False)

#selecting new X for training and Y
X_new = players_22.drop(columns=['overall'])  # Features
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)
y_new = players_22['overall']  # Target variable

# Load the best model which was the RandomForestRegressor
with open("RandomForestRegressorV2.actual.pkl", 'rb') as file:
    best_model = pkl.load(file)


with open("DecisionTreeRegressorV2.actual.pkl", 'rb') as file:
    model_2 = pkl.load(file)

# Measure performance on the new dataset
new_predictions = best_model.predict(X_new)
new_mse = mean_squared_error(y_new, new_predictions)
print(f'New Data Mean Squared Error: {new_mse}')



new_predictions = model_2.predict(X_new)
new_mse = mean_squared_error(y_new, new_predictions)
print(f'New Data Mean Squared Error: {new_mse}')

mse, mae, r2 = evaluate_model(best_model,X_new ,y_new )
    
print(f"Model: {best_model.__class__.__name__}, MSE: {mse}, MAE: {mae}, R2: {r2}")


mse, mae, r2 = evaluate_model(model_2,X_new ,y_new )
    
print(f"Model: {model_2.__class__.__name__}, MSE: {mse}, MAE: {mae}, R2: {r2}")







# Load the trained model
with open('DecisionTreeRegressorV2.actual.pkl', 'rb') as file:
    model = pkl.load(file)

    

# Define the prediction function
def predict_player_rating(features):
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
    rating = predict_player_rating(input_data)
    st.write(f'Predicted Player Rating: {rating}')


































