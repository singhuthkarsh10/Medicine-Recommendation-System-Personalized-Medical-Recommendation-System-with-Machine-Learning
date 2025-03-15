from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI(title="Medicine Recommendation API", description="API for symptom-based diagnosis and medicine recommendations")

# Define correct file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load CSV data with error handling
try:
    symptoms_df = pd.read_csv(os.path.join(BASE_DIR, "symptoms_df.csv"))
    description_df = pd.read_csv(os.path.join(BASE_DIR, "description.csv"))
    medications_df = pd.read_csv(os.path.join(BASE_DIR, "medications.csv"))
    precautions_df = pd.read_csv(os.path.join(BASE_DIR, "precautions_df.csv"))
except FileNotFoundError as e:
    raise RuntimeError(f"CSV file not found: {e}")
except Exception as e:
    raise RuntimeError(f"Error loading CSV files: {e}")

# Load or create Label Encoder
label_encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")
if os.path.exists(label_encoder_path):
    try:
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading label encoder: {e}")
else:
    try:
        label_encoder = LabelEncoder()
        label_encoder.fit(description_df["Disease"].dropna().unique())
        with open(label_encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
    except Exception as e:
        raise RuntimeError(f"Error creating label encoder: {e}")

# Load trained ML model with error handling
try:
    with open(os.path.join(BASE_DIR, "svc.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError("Trained model file 'svc.pkl' not found!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Create symptoms dictionary globally
try:
    all_symptoms = set()
    for col in symptoms_df.columns[1:]:  # Skip "Disease" column
        all_symptoms.update(symptoms_df[col].dropna().unique())
    
    # Normalize symptom names
    symptoms_dict = {symptom.strip().lower().replace(" ", "_"): idx for idx, symptom in enumerate(all_symptoms)}
    num_symptoms = len(symptoms_dict)
except Exception as e:
    raise RuntimeError(f"Error processing symptoms: {e}")

# Define Request Model
class SymptomRequest(BaseModel):
    symptoms: List[str]

@app.get("/")
def home():
    return {"message": "Welcome to the Medicine Recommendation API"}

@app.get("/symptoms")
def get_symptoms():
    """ Returns the list of available symptoms """
    return {"symptoms": list(symptoms_dict.keys())}

@app.post("/predict")
def predict_disease(request: SymptomRequest):
    try:
        symptoms = request.symptoms
        if not symptoms:
            raise HTTPException(status_code=400, detail="Please provide at least one symptom")

        # Normalize input symptoms
        normalized_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms]
        num_symptoms = model.n_features_in_  # Get correct feature count
        symptom_vector = [0] * num_symptoms
        symptom_vector = np.array(symptom_vector).reshape(1, -1)
    

        for symptom in normalized_symptoms:
            if symptom in symptoms_dict:
                symptom_vector[0, symptoms_dict[symptom]] = 1
            else:
                raise HTTPException(status_code=400, detail=f"Invalid symptom: {symptom}")

        # Make prediction
        predicted_index = model.predict(symptom_vector)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

        # Fetch disease details
        disease_info = description_df.loc[
            description_df["Disease"].str.strip().str.lower() == predicted_disease.strip().lower()
        ]
        if disease_info.empty:
            raise HTTPException(status_code=404, detail=f"Disease '{predicted_disease}' not found in dataset")

        disease_description = disease_info["Description"].values[0]

        # Fetch medications
        medication_info = medications_df.loc[
            medications_df["Disease"].str.strip().str.lower() == predicted_disease.strip().lower()
        ]
        medications = medication_info["Medication"].dropna().tolist() if not medication_info.empty else ["No specific medication found"]

        # Fetch precautions safely
        precaution_info = precautions_df.loc[
            precautions_df["Disease"].str.strip().str.lower() == predicted_disease.strip().lower()
        ]
        precautions = (
            precaution_info.iloc[:, 1:].dropna().values.flatten().tolist()
            if not precaution_info.empty and not precaution_info.iloc[:, 1:].isnull().all().all()
            else ["No precautions found"]
        )

        # Handle NaN, Inf values safely
        response = {
            "predicted_disease": predicted_disease,
            "description": disease_description,
            "medications": medications,
            "precautions": precautions,
        }

        for key, value in response.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                response[key] = None  # Replace invalid values with None

        return response

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")