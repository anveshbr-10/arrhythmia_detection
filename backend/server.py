from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn

app = FastAPI(title="Arrhythmia AI API")

# 1. CORS Setup: Allows the React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any frontend port (e.g., 5173 or 3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load your brain
print("Loading the 93% Accurate AI Brain...")
model = tf.keras.models.load_model('best_model.keras')
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]

# 3. Define the incoming data shape
class ECGPayload(BaseModel):
    heartbeat_array: list[float]

@app.get("/")
async def root():
    return {
        "message": "AI Engine is Online! 🚀", 
        "instruction": "Send your 187-point heartbeat array as a POST request to the /predict endpoint."
    }

# 4. The single API endpoint
@app.post("/predict")
async def predict_heartbeat(payload: ECGPayload):
    try:
        signal = np.array(payload.heartbeat_array)
        
        if len(signal) != 187:
            return {"error": f"Invalid data. Expected 187, got {len(signal)}"}
        
        signal_reshaped = signal.reshape((1, 187, 1))
        prediction_probs = model.predict(signal_reshaped, verbose=0)
        
        predicted_index = np.argmax(prediction_probs[0])
        confidence_score = float(np.max(prediction_probs[0]))
        
        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence_score * 100, 2),
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    # Host 0.0.0.0 is crucial so other laptops on the Wi-Fi can connect!
    uvicorn.run(app, host="0.0.0.0", port=8000)