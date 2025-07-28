# src/models/train_model.py

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from src.models.tune_model_GridSCV import tune_random_forest

# Function to train the model using hyperparameter tuning
def train_RFmodel(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tune and train the model
    model, best_params = tune_random_forest(X_train_scaled, y_train)

    # Save model and best params
    with open('models/RFmodel_GridSCV.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    return model, X_test_scaled, y_test
