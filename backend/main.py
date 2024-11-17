from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
from lightgbm import LGBMRegressor
import os
import pandas as pd
import io
import preprocess
from fastapi.responses import JSONResponse


app = FastAPI()

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
model: LGBMRegressor = pickle.load(open(os.path.join('model', 'lightgbm_model.pkl'), 'rb'))
upload_df: pd.DataFrame | None = None

def safe_value(val, as_float=False):
    if pd.isna(val):
        return ''
    return float(val) if as_float else str(val)

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
        predictions = model.predict(processed_df.head(20))
        print(predictions)
        response_data = [
            {
                "listing_id": safe_value(row['Listing.ListingId']),
                "adress": safe_value(row['Location.Address.UnparsedAddress']),
                "structure_yearbuilt": safe_value(row['Structure.YearBuilt']),
                "property_propertytype": safe_value(row['Property.PropertyType']),
                "structure_bathroomsfull": safe_value(row['Structure.BathroomsFull']),
                "structure_bathroomshalf": safe_value(row['Structure.BathroomsHalf']),
                "structure_bedroomstotal": safe_value(row['Structure.BedroomsTotal']),
                "structure_basement": safe_value(row['Structure.Basement']),
                "location": {
                    "latitude": safe_value(row['Location.GIS.Latitude'], as_float=True),
                    "longitude": safe_value(row['Location.GIS.Longitude'], as_float=True)
                },
                "prediction": safe_value(pred, as_float=True)
            }
            for pred, (_, row) in zip(predictions, upload_df.iterrows())
        ]
        with open('predictions.json', 'w') as f:
            f.write(str(response_data))
        
        return JSONResponse(
            content=response_data,
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Prediction Failed -> "+str(e))

    

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "uploaded_df": upload_df is not None
    }