# Home Price Prediction API

A FastAPI-based REST API that serves home price predictions for State College, PA.

## Features

- **Historical Data**: Access 20+ years of Zillow Home Value Index (ZHVI) data
- **Future Predictions**: Get model predictions through 2031
- **Interactive Queries**: Query specific dates for price predictions
- **Chart Data**: Formatted data for visualization

## API Endpoints

### GET /
Returns basic API information.

### GET /api/predictions
Returns all historical and predicted price data.

**Response:**
```json
[
  {
    "date": "2005-07",
    "price": 160901.18,
    "type": "historical"
  },
  {
    "date": "2026-02",
    "price": 335002.45,
    "type": "predicted"
  }
]
```

### GET /api/predictions/{year}/{month}
Returns prediction for a specific year and month.

**Parameters:**
- `year`: Integer (2005-2031)
- `month`: Integer (1-12)

**Response:**
```json
{
  "date": "2026-03",
  "price": 335002.45,
  "type": "predicted"
}
```

### GET /api/date-range
Returns the available date range.

**Response:**
```json
{
  "min_date": "2005-07",
  "max_date": "2031-01",
  "total_records": 307
}
```

### GET /api/chart-data
Returns data formatted for chart visualization.

**Response:**
```json
{
  "historical": [...],
  "predicted": [...],
  "all": [...]
}
```

## Installation

1. Navigate to the API directory:
```bash
cd api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Usage with Frontend

The API is designed to work with the `price_predictor.html` frontend. Make sure:

1. The API server is running on `http://localhost:8000`
2. The HTML file can access the API (CORS is enabled)
3. Update the `API_BASE_URL` in the HTML file if needed

## Data Sources

- **Historical Data**: Zillow Home Value Index (ZHVI) CSV file
- **Predictions**: Machine learning model predictions (Random Forest, 99.98% R²)

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `404`: No data found for requested date
- `500`: Server error

## CORS

CORS is enabled to allow the HTML frontend to make requests from any origin.