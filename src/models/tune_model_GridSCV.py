from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_random_forest(X_train, y_train):
    # Define parameter grid
    param_grid = {
        'n_estimators': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        'max_depth': [None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        'max_features': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    }

    # Initialize base model
    rf = RandomForestClassifier(random_state=42)

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Return best model and params
    return grid_search.best_estimator_, grid_search.best_params_
