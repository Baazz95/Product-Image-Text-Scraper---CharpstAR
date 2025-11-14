# Product Image Scraper API

A FastAPI service for automatically extracting product images from e-commerce websites using AI-powered analysis.

## Features

- **AI-Powered Image Selection**: Uses Google Gemini to intelligently select main product images
- **Multi-Source Scraping**: Extracts images from HTML, CSS, JavaScript, and structured data
- **Cloud Storage Integration**: Uploads images directly to Supabase Storage
- **Database Integration**: Fetches product data from Supabase and updates processing results
- **RESTful API**: Easy integration with other services

## Setup

### Prerequisites

- **Python 3.11.x** (required - Python 3.14+ is not supported due to lxml compatibility)
- pip (Python package manager)
- Access to Supabase project
- Google Gemini API key

### Quick Setup Guide

For detailed step-by-step instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

### 1. Environment Variables

Copy `env.example` to `.env` and fill in your credentials:

```bash
# Windows PowerShell
copy env.example .env

# Linux/Mac
cp env.example .env
```

Required environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key (use service_role, not anon key)
- `GEMINI_API_KEY`: Your Gemini API key
- `GEMINI_MODEL`: Gemini model to use (default: `models/gemini-2.5-pro`)

### 2. Local Development

```bash
# Create virtual environment with Python 3.11 (required)
# Windows:
py -3.11 -m venv venv
# Linux/Mac (if you have multiple Python versions):
python3.11 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Run the application
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Database Setup

Ensure your `onboarding_assets` table has these columns:
- `article_id` (text)
- `product_link` (text)
- `product_name` (text)
- `client` (text)
- `new_upload` (boolean)
- `reference` (text[] or jsonb)
- `tags` (jsonb)
- `category` (text) - **Required for classification**
- `subcategory` (text) - **Required for classification**

If `category` or `subcategory` columns don't exist, run:
```sql
ALTER TABLE onboarding_assets
ADD COLUMN IF NOT EXISTS category TEXT,
ADD COLUMN IF NOT EXISTS subcategory TEXT;
```

### 4. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t image-scraper .
docker run -p 8000:8000 --env-file .env image-scraper
```

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service and database connection.

### Process Client Images
```
POST /process-client/{client_name}
```
Processes all new uploads for a specific client.

**Example:**
```bash
curl -X POST "http://localhost:8000/process-client/MyClient"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully processed 5 products for client 'MyClient'",
  "processed_count": 5,
  "client_name": "MyClient"
}
```

### Get Client Status
```
GET /client-status/{client_name}
```
Returns the number of pending uploads for a client.

**Example:**
```bash
curl "http://localhost:8000/client-status/MyClient"
```

**Response:**
```json
{
  "client_name": "MyClient",
  "pending_uploads": 3,
  "has_pending": true
}
```

## API Documentation

Once the service is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Integration with NextJS

To call this service from your NextJS application:

```javascript
// pages/api/process-images.js
export default async function handler(req, res) {
  if (req.method === 'POST') {
    const { clientName } = req.body;
    
    try {
      const response = await fetch(`http://your-python-service:8000/process-client/${clientName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const result = await response.json();
      res.status(200).json(result);
    } catch (error) {
      res.status(500).json({ error: 'Processing failed' });
    }
  }
}
```

## Database Schema

The service expects a Supabase table called `onboarding_assets` with the following structure:

- `article_id` (string): Product SKU/ID
- `product_link` (string): Product URL
- `product_name` (string): Product name
- `client` (string): Client name
- `new_upload` (boolean): Whether this is a new upload to process
- `reference` (array): Array of image URLs (updated after processing)

## Storage Structure

Images are uploaded to Supabase Storage in the following structure:
```
assets/assets/{client_name}/{article_id}/references/image_N.jpg
```

## Error Handling

The API includes comprehensive error handling:
- Database connection errors
- Missing environment variables
- Invalid client names
- Processing failures

All errors return appropriate HTTP status codes and error messages.

## Monitoring

The service includes:
- Health check endpoint for monitoring
- Structured logging
- Docker health checks
- Error tracking and reporting

