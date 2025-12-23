import os
import joblib
import pandas as pd

def load_models(data_dir='data'):
    """
    Load all .joblib models from the data directory.
    Returns a dictionary {model_name: model_object}
    """
    models = {}
    if not os.path.exists(data_dir):
        return models
        
    for filename in os.listdir(data_dir):
        if filename.endswith('.joblib'):
            model_name = filename.replace('.joblib', '').replace('_', ' ').title()
            model_path = os.path.join(data_dir, filename)
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return models

def predict_all(models, input_df):
    """
    Generate predictions for input_df using all models.
    Returns a DataFrame with predictions.
    """
    results = {}
    for name, model in models.items():
        try:
            pred = model.predict(input_df)[0]
            results[name] = pred
        except Exception as e:
            results[name] = f"Error: {e}"
    return results

def get_coefficients_df(models, feature_names):
    """
    Return a DataFrame of coefficients for all models.
    """
    data = {'Feature': feature_names}
    
    for name, model in models.items():
        if hasattr(model, 'coef_'):
            data[name] = model.coef_
        else:
            data[name] = [None] * len(feature_names)
    
    # Add intercept row
    intercept_row = {'Feature': 'Intercept'}
    for name, model in models.items():
        intercept = getattr(model, 'intercept_', None)
        intercept_row[name] = intercept
        
    df = pd.DataFrame(data)
    # df = pd.concat([df, pd.DataFrame([intercept_row])], ignore_index=True) # Check pandas version syntax
    # Simpler: append row manually or construct differently
    
    return df
