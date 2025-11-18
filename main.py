from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from batch_processor import process_batch, fetch_new_uploads, get_assets_table_name
from tag_generator import AITagGenerator, TagResult
import logging
from typing import Set, List, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Product Image Scraper API",
    description="API for processing product images from e-commerce websites",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ProcessClientRequest(BaseModel):
    client_name: str

class TagGenerationRequest(BaseModel):
    product_name: str
    product_url: str
    image_urls: List[str]
    max_images: int = 5

class BatchTagGenerationRequest(BaseModel):
    products: List[Dict[str, Any]]
    max_images_per_product: int = 5

# Global tracking for active processing
active_clients: Set[str] = set()

# Initialize Supabase client
def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is missing")
    if not supabase_key:
        raise HTTPException(status_code=500, detail="SUPABASE_KEY is missing")
    return create_client(supabase_url, supabase_key)

@app.get("/")
async def root():
    return {"message": "Product Image Scraper API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        supabase = get_supabase_client()
        # Test database connection
        table_name = get_assets_table_name()
        supabase.table(table_name).select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/process-client/{client_name:path}")
async def process_client_images(client_name: str):
    """
    Process product images for a specific client
    
    Args:
        client_name: The name of the client to process (can contain spaces)
        
    Returns:
        JSON response with processing results
    """
    try:
        # Check if client is already being processed
        if client_name in active_clients:
            raise HTTPException(
                status_code=409, 
                detail=f"Client '{client_name}' is already being processed. Please wait for completion."
            )
        
        # Add client to active processing set
        active_clients.add(client_name)
        
        try:
            # Validate required environment variables
            if not os.getenv('GEMINI_API_KEY'):
                raise HTTPException(status_code=500, detail="GEMINI_API_KEY is required")
            
            # Get Supabase client
            supabase = get_supabase_client()
            
            # Fetch new uploads for the client
            new_rows = fetch_new_uploads(client_name, supabase)
        
            if not new_rows:
                return {
                    "status": "no_data", 
                    "message": f"No new uploads found for client '{client_name}'",
                    "processed_count": 0
                }
            
            # Process the batch
            await process_batch(new_rows, client_name, supabase)
            
            return {
                "status": "success",
                "message": f"Successfully processed {len(new_rows)} products for client '{client_name}'",
                "processed_count": len(new_rows),
                "client_name": client_name
            }
            
        finally:
            # Always remove client from active processing set
            active_clients.discard(client_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error processing client {client_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/client-status/{client_name:path}")
async def get_client_status(client_name: str):
    """
    Get the status of pending uploads for a client
    
    Args:
        client_name: The name of the client
        
    Returns:
        JSON response with client status
    """
    try:
        supabase = get_supabase_client()
        new_rows = fetch_new_uploads(client_name, supabase)
        
        return {
            "client_name": client_name,
            "pending_uploads": len(new_rows),
            "has_pending": len(new_rows) > 0
        }
        
    except Exception as e:
        logging.error(f"Error getting status for client {client_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get client status: {str(e)}")

@app.post("/generate-tags")
async def generate_tags(request: TagGenerationRequest):
    """
    Generate AI-powered tags for a single product
    
    Args:
        request: TagGenerationRequest with product details and image URLs
        
    Returns:
        JSON response with generated tags
    """
    try:
        # Validate required environment variables
        if not os.getenv('GEMINI_API_KEY'):
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is required")
        
        # Initialize tag generator
        tag_generator = AITagGenerator(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Generate tags
        result = await tag_generator.generate_tags_for_product(
            product_name=request.product_name,
            product_url=request.product_url,
            image_urls=request.image_urls,
            max_images=request.max_images
        )
        
        if result.error_message:
            raise HTTPException(status_code=500, detail=f"Tag generation failed: {result.error_message}")
        
        return {
            "status": "success",
            "product_name": result.product_name,
            "product_url": result.product_url,
            "tags": result.tags,
            "generated_tags": getattr(result, 'generated_tags', result.tags),
            "category_tags": result.category_tags,
            "style_tags": result.style_tags,
            "material_tags": result.material_tags,
            "color_tags": result.color_tags,
            "brand_tags": result.brand_tags,
            "confidence_scores": result.confidence_scores,
            "processing_time": result.processing_time,
            "images_analyzed": len(result.image_urls)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error generating tags: {e}")
        raise HTTPException(status_code=500, detail=f"Tag generation failed: {str(e)}")

@app.post("/generate-tags-batch")
async def generate_tags_batch(request: BatchTagGenerationRequest):
    """
    Generate AI-powered tags for multiple products in batch
    
    Args:
        request: BatchTagGenerationRequest with list of products
        
    Returns:
        JSON response with generated tags for all products
    """
    try:
        # Validate required environment variables
        if not os.getenv('GEMINI_API_KEY'):
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is required")
        
        # Initialize tag generator
        tag_generator = AITagGenerator(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Generate tags for all products
        results = await tag_generator.generate_tags_batch(
            products=request.products,
            max_images_per_product=request.max_images_per_product
        )
        
        # Format results
        formatted_results = []
        for result in results:
            if result.error_message:
                formatted_results.append({
                    "product_name": result.product_name,
                    "product_url": result.product_url,
                    "status": "error",
                    "error_message": result.error_message
                })
            else:
                formatted_results.append({
                    "product_name": result.product_name,
                    "product_url": result.product_url,
                    "status": "success",
                    "tags": result.tags,
                    "generated_tags": getattr(result, 'generated_tags', result.tags),
                    "category_tags": result.category_tags,
                    "style_tags": result.style_tags,
                    "material_tags": result.material_tags,
                    "color_tags": result.color_tags,
                    "brand_tags": result.brand_tags,
                    "confidence_scores": result.confidence_scores,
                    "processing_time": result.processing_time,
                    "images_analyzed": len(result.image_urls)
                })
        
        return {
            "status": "success",
            "total_products": len(results),
            "successful": len([r for r in results if not r.error_message]),
            "failed": len([r for r in results if r.error_message]),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error generating tags in batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch tag generation failed: {str(e)}")

@app.get("/products-with-tags/{client_name:path}")
async def get_products_with_tags(client_name: str, limit: int = 50):
    """
    Get products that have generated tags for a specific client
    
    Args:
        client_name: The name of the client
        limit: Maximum number of products to return
        
    Returns:
        JSON response with products and their tags
    """
    try:
        supabase = get_supabase_client()
        
        # Query products with generated tags
        table_name = get_assets_table_name()
        response = (
            supabase.table(table_name)
            .select("article_id, product_name, product_link, tags, category_tags, style_tags, material_tags, color_tags, brand_tags, reference")
            .eq("client", client_name)
            .not_.is_("tags", "null")
            .limit(limit)
            .execute()
        )
        
        return {
            "client_name": client_name,
            "total_products": len(response.data),
            "products": response.data
        }
        
    except Exception as e:
        logging.error(f"Error getting products with tags for client {client_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get products with tags: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
