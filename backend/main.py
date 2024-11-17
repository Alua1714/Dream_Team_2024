from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
from lightgbm import LGBMRegressor
import os
import pandas as pd
import io
import preprocess



app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "https://datathon.container.aed.cat",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
model: LGBMRegressor = pickle.load(open(os.path.join('models', 'lightgbm_model.pkl'), 'rb'))
upload_df: pd.DataFrame | None = None

@app.post("/upload/")
async def upload_file(file: UploadFile):
    global upload_df, model
    try:
        contents = await file.read()
        upload_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading the .CSV -> "+str(e))

    if upload_df is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # LOGS
    print("rows:", len(upload_df))
    print("columnes:",len(upload_df.columns))
    
    try:
        processed_df = preprocess.process_data(upload_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Processed Pipeline Failed -> "+str(e))

    # LOGS
    print("rows:", len(processed_df))
    print("columnes:",len(processed_df.columns))
    try:
        predictions = model.predict(processed_df)
        
        response_data = [
            {
                "listing_id": str(row['Listing.Id']),
                "adress": str(row['Listing.Address']),
                "Structure.YearBuilt": str(row['Structure.YearBuilt']),
                "Property.PropertyType": str(row['Property.PropertyType']),
                "Structure.BathroomsFull": str(row['Structure.BathroomsFull']),
                "Structure.BathroomsHalf": str(row['Structure.BathroomsHalf']),
                "Structure.BedroomsTotal": str(row['Structure.BedroomsTotal']),
                "Structure.Basement": str(row['Structure.Basement']),
                "location": {
                    "latitude": float(row['Listing.Location.Latitude']),
                    "longitude": float(row['Listing.Location.Longitude'])
                },
                "prediction": float(pred)
            }
            for pred, (_, row) in zip(predictions, upload_df.iterrows())
        ]
        
        return response_data
    except Exception as e:
        raise HTTPException(status_code=400, detail="Prediction Failed -> "+str(e))
    
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "uploaded_df": upload_df is not None
    }