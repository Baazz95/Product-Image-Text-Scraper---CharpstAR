# Production Deployment Guide

This guide covers deploying the image scraper to production and connecting it to your live database.

## Pre-Deployment Checklist

- [ ] Test all functionality on staging/test database
- [ ] Verify environment variables are correctly configured
- [ ] Ensure API keys have appropriate rate limits
- [ ] Review and test tag generation limits (20 tags max)
- [ ] Verify measurement tag prioritization works correctly
- [ ] Test with a small batch of products first

## Step 1: Environment Configuration

### Update `.env` File for Production

**CRITICAL**: Switch from test/staging database to production database.

```bash
# Production Supabase Configuration
SUPABASE_URL=https://your-production-project.supabase.co
SUPABASE_KEY=your-production-service-role-key  # Use service_role key, NOT anon key

# Gemini API Configuration
GEMINI_API_KEY=your-production-gemini-api-key
GEMINI_MODEL=models/gemini-2.5-pro  # or gemini-2.5-flash for faster processing

# Optional: Logging Level
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

### Important Notes:

1. **Service Role Key**: Always use the `service_role` key, not the `anon` key. The service role key has full database access required for updates.

2. **API Rate Limits**: 
   - Gemini API has rate limits. Monitor usage in Google Cloud Console.
   - Consider using `gemini-2.5-flash` for faster, cheaper processing if quality is acceptable.

3. **Database Permissions**: Ensure the service role key has:
   - `SELECT` on `onboarding_assets` table
   - `UPDATE` on `onboarding_assets` table
   - `INSERT` permission on Supabase Storage bucket

## Step 2: Database Schema Verification

Ensure your production `onboarding_assets` table has all required columns:

```sql
-- Check existing columns
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'onboarding_assets';

-- Add missing columns if needed
ALTER TABLE onboarding_assets
ADD COLUMN IF NOT EXISTS category TEXT,
ADD COLUMN IF NOT EXISTS subcategory TEXT,
ADD COLUMN IF NOT EXISTS likely_null BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS reference TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Verify reference column can store arrays
-- If it's JSONB, you may need to migrate:
-- ALTER TABLE onboarding_assets ALTER COLUMN reference TYPE TEXT[] USING reference::TEXT[];
```

## Step 3: Storage Bucket Configuration

### Verify Supabase Storage Setup

1. **Bucket Exists**: Ensure `assets` bucket exists in Supabase Storage
2. **Permissions**: Service role key should have full access
3. **Path Structure**: Images will be stored as:
   ```
   assets/assets/{client_name}/{article_id}/references/image_N.jpg
   ```

### Storage Policies (if using RLS)

If Row Level Security is enabled, ensure policies allow service role access:

```sql
-- Allow service role to upload files
CREATE POLICY IF NOT EXISTS "Service role can upload files"
ON storage.objects
FOR INSERT
TO service_role
WITH CHECK (bucket_id = 'assets');

-- Allow service role to read files
CREATE POLICY IF NOT EXISTS "Service role can read files"
ON storage.objects
FOR SELECT
TO service_role
USING (bucket_id = 'assets');
```

## Step 4: Tag Generation Configuration

### Current Tag Settings

- **Maximum Tags**: 20 tags per product (hard limit)
- **Priority Order**:
  1. Measurement/dimension tags (height, width, depth, weight, size)
  2. Material tags
  3. Style/design tags
  4. Color tags
  5. Functional attributes
  6. Brand/model information

### Tag Generation Process

1. **Image Tags**: Generated from product images using Gemini Vision API
   - Limited to 20 tags total
   - Measurement tags prioritized
   - Uses `gemini-2.5-flash` or `gemini-2.5-pro` model

2. **Text Tags**: Extracted from product page HTML
   - Dimensions extracted separately and added as measurement tags
   - Product type included if available

3. **Combined Tags**: Final `all_tags` array combines:
   - Measurement tags from text extraction (highest priority)
   - Image tags (already limited to 20)
   - Product type
   - Final limit: 20 tags total

## Step 5: Running the Scraper

### Option A: Direct Python Execution

```bash
# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Run batch processor
python process_csv_batch.py your_file.csv

# Or process from database
python -m batch_processor --client-name "YourClientName"
```

### Option B: Docker Deployment

```bash
# Build Docker image
docker build -t image-scraper:latest .

# Run with production .env
docker run --env-file .env.production image-scraper:latest

# Or use docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Option C: API Server (FastAPI)

```bash
# Start API server
python main.py

# Process client via API
curl -X POST "http://localhost:8000/process-client/YourClientName"
```

## Step 6: Monitoring and Logging

### Log Files

- Check console output for processing status
- Logs include:
  - Product processing progress
  - Image extraction results
  - Tag generation status
  - Classification results
  - Database update confirmations

### Key Metrics to Monitor

1. **Processing Rate**: Products processed per hour
2. **Success Rate**: Percentage of successful classifications
3. **Image Extraction**: Number of images extracted per product
4. **Tag Generation**: Average tags per product (should be ≤20)
5. **API Costs**: Monitor Gemini API usage

### Error Handling

The scraper handles:
- 404 pages (flagged as `likely_null`)
- Invalid URLs (skipped with warning)
- API failures (logged, product skipped)
- Database connection issues (retries with backoff)

## Step 7: Testing Production Deployment

### Small Batch Test

1. **Test with 5-10 products** first:
   ```bash
   python process_csv_batch.py test_batch.csv
   ```

2. **Verify Results**:
   - Check database for updated records
   - Verify images uploaded to storage
   - Confirm tags are generated (≤20 tags)
   - Check measurement tags are prioritized
   - Verify category/subcategory classifications

3. **Check Logs**:
   - No critical errors
   - All products processed successfully
   - Tags generated correctly

### Full Production Run

Once small batch test passes:

1. **Process all pending products**:
   ```bash
   python -m batch_processor --client-name "YourClientName"
   ```

2. **Monitor Progress**:
   - Watch console output
   - Check database periodically
   - Monitor API usage

3. **Verify Final Results**:
   - All products have `new_upload=False`
   - Images uploaded correctly
   - Tags populated
   - Categories assigned

## Step 8: Production Considerations

### Rate Limiting

- **Gemini API**: Monitor rate limits in Google Cloud Console
- **Supabase**: Check database connection pool limits
- **Image Downloads**: May be rate-limited by source websites

### Cost Management

- **Gemini API Costs**: 
  - `gemini-2.5-flash`: ~$0.075 per 1M tokens
  - `gemini-2.5-pro`: ~$1.25 per 1M tokens
  - Image analysis uses more tokens

- **Storage Costs**: Monitor Supabase Storage usage

### Performance Optimization

- **Batch Processing**: Process multiple products concurrently
- **Image Limits**: Currently limited to 5 images per product
- **Tag Limits**: Hard limit of 20 tags prevents bloat

### Backup Strategy

- **Database**: Ensure Supabase backups are enabled
- **Code**: Version control all changes
- **Environment**: Keep `.env` files secure (use secrets management)

## Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Check `.env` file exists and has all required variables
   - Verify no typos in variable names

2. **"Database connection failed"**
   - Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
   - Check network connectivity
   - Ensure service role key has proper permissions

3. **"No images extracted"**
   - Check if product page loads correctly
   - Verify images exist on the page
   - Check for JavaScript-rendered images (may need wait time)

4. **"Tags not generated"**
   - Verify `GEMINI_API_KEY` is valid
   - Check API rate limits
   - Review error logs for specific failures

5. **"Category classification failed"**
   - Check if domain is in `get_domain_expected_category` mapping
   - Verify text extraction is working
   - Review classification logs

## Rollback Plan

If issues occur:

1. **Stop Processing**: Kill running processes
2. **Review Logs**: Identify root cause
3. **Fix Issues**: Update code/environment as needed
4. **Test Again**: Run small batch test
5. **Resume**: Continue processing once fixed

## Support

For issues or questions:
- Check logs for detailed error messages
- Review this deployment guide
- Check code comments for implementation details
- Test with small batches before full deployment

