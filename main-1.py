from setuptools import find_packages, setup

# Commented out: used for packaging
# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Mary Grace Lunar',
#     license='',
# )

from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import create_dummy_vars
from src.models.train_model_GridSCV import train_RFmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Step 1: Load and preprocess the data
    data_path = "data/raw/credit.csv"
    df = load_and_preprocess_data(data_path)

    # Step 2: Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Step 3: Train model with hyperparameter tuning
    model, X_test_scaled, y_test = train_RFmodel(X, y)

    # Step 4: Evaluate the model
    plot_feature_importance(model, X)
    accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)

    print(f"\nModel Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{confusion_mat}")

    # Step 5: Optionally show tuned parameters (if saved)
    if hasattr(model, 'get_params'):
        print("\nBest Parameters Used:")
        print(model.get_params())
