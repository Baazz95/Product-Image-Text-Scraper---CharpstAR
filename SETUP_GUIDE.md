# Setup Guide - Step by Step

This guide will walk you through setting up the environment for the Product Image Scraper API with text extraction integration.

## Prerequisites

- **Python 3.11.x** (required - Python 3.12+ is not supported due to lxml compatibility with crawl4ai)
- pip (Python package manager)
- Access to Supabase project
- Google Gemini API key

**Important:** This project requires Python 3.11.x specifically. Python 3.14+ will fail to install dependencies because `crawl4ai` requires `lxml~=5.3`, which doesn't have pre-built wheels for newer Python versions.

## Step 1: Check Python Installation

First, verify Python is installed:

```powershell
python --version
```

You should see `Python 3.11.x`. If you see a different version (like 3.14.0), you need to install Python 3.11.x from [python.org](https://www.python.org/downloads/).

**To check if Python 3.11 is installed:**
```powershell
py -3.11 --version
```



## Step 2: Create Virtual Environment (Recommended)

It's best practice to use a virtual environment to isolate dependencies:

```powershell
# Navigate to your project directory
cd "C:\Users\CharpstAR Guest\Desktop\Michael Charpstar\Projects\image_text_scrape_classification\image_text_scraper_joined-main"

# Create virtual environment with Python 3.11 (IMPORTANT: use py -3.11 to force Python 3.11)
py -3.11 -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 3: Install Python Dependencies

With the virtual environment activated, install all required packages:

```powershell
pip install -r requirements.txt
```

This will install:
- FastAPI and uvicorn (web server)
- Supabase client
- Google Generative AI
- Playwright
- BeautifulSoup4
- And other dependencies

## Step 4: Install Playwright Browsers

Playwright needs to download browser binaries:

```powershell
playwright install chromium
```

This downloads Chromium browser (~200MB) needed for web scraping.

## Step 5: Get Your API Keys and Credentials

### 5.1 Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey) or [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new API key for Gemini
3. Copy the API key (looks like: `AIza...`)

### 5.2 Get Supabase Credentials

1. Go to your [Supabase Dashboard](https://app.supabase.com/)
2. Select your project
3. Go to **Settings** → **API**
4. Copy:
   - **Project URL** (looks like: `https://xxxxx.supabase.co`)
   - **Service Role Key** (looks like: `eyJ...` - this is the `anon` key or `service_role` key)

**Important:** Use the **service_role** key (not the anon key) for server-side operations.

## Step 6: Create .env File

Create a `.env` file in your project root with your credentials:

```powershell
# Copy the example file
copy env.example .env
```

Then edit `.env` with your actual credentials:

```env
# Supabase Configuration
SUPABASE_URL=https://ubortzfhvagvewhoppid.supabase.co
SUPABASE_KEY=...

# Gemini AI Configuration
GEMINI_API_KEY=...
GEMINI_MODEL=models/gemini-2.5-pro
```

### How to Edit .env File

**Option 1: Using Notepad**
```powershell
notepad .env
```

**Option 2: Using VS Code**
```powershell
code .env
```

**Option 3: Using PowerShell**
```powershell
# Edit the file directly
```

Replace the placeholder values with your actual credentials:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: Keep as `models/gemini-2.5-pro` (this is correct)

## Step 7: Verify Database Schema

Make sure your Supabase `onboarding_assets` table has these columns:

**Required Columns:**
- `article_id` (text) - Product SKU/ID
- `product_link` (text) - Product URL
- `product_name` (text) - Product name
- `client` (text) - Client name
- `new_upload` (boolean) - Whether to process
- `reference` (text[] or jsonb) - Array of image URLs
- `tags` (jsonb) - Tags and metadata
- `category` (text) - Product category
- `subcategory` (text) - Product subcategory

**To check your table:**
1. Go to Supabase Dashboard
2. Go to **Table Editor**
3. Select `onboarding_assets` table
4. Verify columns exist

**If `category` or `subcategory` columns don't exist:**
Run this SQL in Supabase SQL Editor:

```sql
ALTER TABLE onboarding_assets
ADD COLUMN IF NOT EXISTS category TEXT,
ADD COLUMN IF NOT EXISTS subcategory TEXT;
```

## Step 8: Verify Storage Bucket

Make sure you have a storage bucket named `assets` in Supabase:

1. Go to Supabase Dashboard
2. Go to **Storage**
3. Verify `assets` bucket exists
4. If not, create it with public access

## Step 9: Test the Setup

### 9.1 Test Environment Variables

Create a test script to verify your environment variables are loaded:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('SUPABASE_URL:', os.getenv('SUPABASE_URL')[:20] + '...' if os.getenv('SUPABASE_URL') else 'NOT SET'); print('GEMINI_API_KEY:', 'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'); print('GEMINI_MODEL:', os.getenv('GEMINI_MODEL'))"
```

You should see your credentials (partially masked).

### 9.2 Test Database Connection

Test if you can connect to Supabase:

```powershell
python -c "from dotenv import load_dotenv; from supabase import create_client; import os; load_dotenv(); supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')); result = supabase.table('onboarding_assets').select('id').limit(1).execute(); print('✓ Database connection successful!')"
```

### 9.3 Test Gemini API

Test if Gemini API key works:

```powershell
python -c "from dotenv import load_dotenv; import os; import google.generativeai as genai; load_dotenv(); genai.configure(api_key=os.getenv('GEMINI_API_KEY')); model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'models/gemini-2.5-pro')); response = model.generate_content('Say hello'); print('✓ Gemini API works! Response:', response.text[:50])"
```

## Step 10: Run the Application

### Option 1: Run as API Server

Start the FastAPI server:

```powershell
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Then visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Option 2: Run Batch Processor Directly

Process products for a specific client:

```powershell
python batch_processor.py "Client Name"
```

Replace `"Client Name"` with an actual client name from your database.

## Troubleshooting

### Issue: "GEMINI_API_KEY not set"

**Solution:** Make sure your `.env` file exists and contains `GEMINI_API_KEY=your_key`. Check that you're in the project directory.

### Issue: "SUPABASE_URL is missing"

**Solution:** Verify your `.env` file has `SUPABASE_URL` and `SUPABASE_KEY` set correctly.

### Issue: "Playwright browsers not installed"

**Solution:** Run `playwright install chromium` again.

### Issue: "Module not found" errors

**Solution:** 
1. Make sure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`

### Issue: Database connection fails

**Solution:**
1. Verify Supabase URL is correct
2. Verify Service Role Key (not anon key) is used
3. Check if Supabase project is active (free tier pauses after inactivity)
4. Verify table name is `onboarding_assets`

### Issue: "Category column doesn't exist"

**Solution:** Run the SQL migration to add columns (see Step 7).

## Quick Test Command

Test everything at once:

```powershell
# Test environment
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Env vars:', 'OK' if all([os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'), os.getenv('GEMINI_API_KEY')]) else 'MISSING')"

# Test database
python -c "from dotenv import load_dotenv; from supabase import create_client; import os; load_dotenv(); supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')); print('Database: OK' if supabase else 'FAILED')"

# Test Gemini
python -c "from dotenv import load_dotenv; import os; import google.generativeai as genai; load_dotenv(); genai.configure(api_key=os.getenv('GEMINI_API_KEY')); print('Gemini: OK')"
```

## Next Steps

Once setup is complete:

1. **Test with a single product:**
   ```powershell
   python single_product_dry_run.py
   ```

2. **Check client status:**
   ```powershell
   # Start API server first
   python main.py
   
   # Then in another terminal:
   curl http://localhost:8000/client-status/"Client Name"
   ```

3. **Process a client:**
   ```powershell
   python batch_processor.py "Client Name"
   ```

## File Structure After Setup

```
Image_scrape_classification/
├── .env                    # Your credentials (DON'T COMMIT THIS)
├── venv/                   # Virtual environment (if created)
├── main.py                 # FastAPI server
├── batch_processor.py      # Batch processing
├── text_extractor.py       # Text extraction (new)
├── category_mapper.py      # Category mapping (new)
├── anchor_selector.py      # Image scraping
├── tag_generator.py        # Image tag generation
├── requirements.txt        # Dependencies
├── env.example            # Example env file
└── ...
```

## Important Notes

1. **Never commit `.env` file** - It contains sensitive credentials
2. **Use service_role key** - Not anon key for Supabase
3. **Virtual environment** - Always activate before running
4. **Playwright browsers** - Must be installed separately
5. **Database columns** - Make sure `category` and `subcategory` exist

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify all environment variables are set
3. Check Supabase project is active
4. Verify API keys are valid
5. Check database table structure

Good luck! 🚀

