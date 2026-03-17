from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from typing import List, Dict, Any
import os

app = FastAPI(title="Home Price Prediction API", description="API for home price predictions in State College, PA")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the combined data
predictions_data = None

def load_data():
    """Load and combine historical and future price data"""
    global predictions_data

    if predictions_data is not None:
        return predictions_data

    # Load historical data
    historical_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "zhvi_metro.csv")
    historical_df = pd.read_csv(historical_path)

    # Filter for State College, PA (assuming it's in the data)
    # Look for rows containing "State College" or similar
    state_college_rows = historical_df[
        historical_df['RegionName'].str.contains('State College', case=False, na=False)
    ]

    if state_college_rows.empty:
        # If no exact match, try to find PA metro areas
        pa_rows = historical_df[historical_df['StateName'] == 'PA']
        if not pa_rows.empty:
            # Use the first PA metro area as proxy
            state_college_row = pa_rows.iloc[0]
        else:
            raise ValueError("Could not find State College, PA data in historical dataset")
    else:
        state_college_row = state_college_rows.iloc[0]

    # Extract historical prices - columns from 2000-01-31 onwards
    date_columns = [col for col in historical_df.columns if col.startswith('20') and '-' in col]
    historical_prices = []

    for date_col in date_columns:
        price = state_college_row[date_col]
        if pd.notna(price):
            # Convert date format from YYYY-MM-DD to YYYY-MM
            date_obj = pd.to_datetime(date_col)
            year_month = f"{date_obj.year}-{date_obj.month:02d}"
            historical_prices.append({
                'date': year_month,
                'price': float(price),
                'type': 'historical'
            })

    # Load future predictions
    future_path = os.path.join(os.path.dirname(__file__), "..", "results", "future_price_predictions.csv")
    future_df = pd.read_csv(future_path)

    future_prices = []
    for _, row in future_df.iterrows():
        date_str = row['date']
        price = row['predicted_price']

        # Convert date to YYYY-MM format
        date_obj = pd.to_datetime(date_str)
        year_month = f"{date_obj.year}-{date_obj.month:02d}"

        future_prices.append({
            'date': year_month,
            'price': float(price),
            'type': 'predicted'
        })

    # Combine and sort by date
    all_prices = historical_prices + future_prices
    all_prices.sort(key=lambda x: x['date'])

    predictions_data = all_prices
    return predictions_data

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Home Price Prediction API", "status": "running"}

@app.get("/predictor")
async def get_predictor():
    """Serve the price predictor HTML page"""
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "price_predictor.html")
    return FileResponse(file_path, media_type='text/html')

@app.get("/api/predictions")
async def get_all_predictions() -> List[Dict[str, Any]]:
    """Get all historical and predicted prices"""
    data = load_data()
    return data

@app.get("/api/predictions/{year}/{month}")
async def get_prediction_by_date(year: int, month: int):
    """Get prediction for a specific year and month"""
    data = load_data()

    # Format the target date
    target_date = f"{year}-{month:02d}"

    # Find the matching prediction
    for prediction in data:
        if prediction['date'] == target_date:
            return prediction

    # If not found, return error
    raise HTTPException(
        status_code=404,
        detail=f"No prediction found for {year}-{month:02d}"
    )

@app.get("/api/chart-data")
async def get_chart_data():
    """Get data formatted for chart visualization"""
    data = load_data()

    # Separate historical and predicted data
    historical = [p for p in data if p['type'] == 'historical']
    predicted = [p for p in data if p['type'] == 'predicted']

    return {
        'historical': historical,
        'predicted': predicted,
        'all': data
    }

@app.get("/api/date-range")
async def get_date_range():
    """Get the available date range"""
    data = load_data()

    if not data:
        return {"min_date": None, "max_date": None}

    dates = [p['date'] for p in data]
    dates.sort()

    return {
        "min_date": dates[0],
        "max_date": dates[-1],
        "total_records": len(data)
    }