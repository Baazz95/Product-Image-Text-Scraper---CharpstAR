import argparse
import asyncio
import json
import os
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from dataclasses import asdict
from ftplib import FTP
from io import BytesIO

from dotenv import load_dotenv
from supabase import create_client, Client

from anchor_selector import learn_domain_selector, DownloadResult, DomainProfile
from tag_generator import AITagGenerator, TagResult
from text_extractor import extract_all_text_data, TextExtractionResult
from category_mapper import map_product_type_to_category, validate_category, validate_subcategory
import aiohttp
import base64

# Load environment variables at module level
load_dotenv()

# --- Helper Functions ---

def sanitize_for_storage_path(name: str) -> str:
    """
    Sanitize a name (client name, article ID) for use in Supabase Storage paths.
    Removes or replaces special characters that aren't allowed in storage paths.
    
    Args:
        name: Name to sanitize
        
    Returns:
        Sanitized name safe for use in storage paths
    """
    import re
    import unicodedata
    
    # Normalize unicode characters (convert ö to o, etc.)
    name = unicodedata.normalize('NFKD', name)
    
    # Remove special characters, keep only alphanumeric, spaces, hyphens, underscores
    # Replace spaces and other invalid chars with underscores
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Ensure it's not empty
    if not name:
        name = 'unknown'
    
    return name


def is_product_id_preserved_in_redirect(original_url: str, final_url: str, product_id: str = None) -> bool:
    """
    Check if the product ID from the original URL appears in the redirected URL.
    
    This allows redirects where the product ID is preserved (e.g., short URL → full product page)
    while still catching redirects to completely different products.
    
    Works purely from URLs - extracts product ID from original URL and checks if it's in final URL.
    The product_id parameter is optional and used as a fallback if extraction fails.
    
    Args:
        original_url: The original product URL
        final_url: The URL after redirect
        product_id: Optional product ID (article_id/sku) - used as fallback if extraction fails
        
    Returns:
        True if product ID from original URL is found in final_url, False otherwise
    """
    if not final_url or not original_url:
        return False
    
    import re
    
    # Extract numeric product ID from original URL (common patterns: /p/123456, /product/123456, etc.)
    # Try multiple patterns to catch different URL structures
    numeric_id_match = re.search(r'/(?:p|product|produkt|artikel|item)/?(\d+)', original_url, re.IGNORECASE)
    
    if numeric_id_match:
        numeric_id = numeric_id_match.group(1)
        # Check if this numeric ID appears anywhere in the final URL
        if numeric_id in final_url:
            logging.debug(f"Product ID '{numeric_id}' extracted from original URL is present in final URL")
            return True
    
    # Fallback: if product_id parameter provided and extraction failed, check if it's in final URL
    if product_id:
        if product_id in final_url:
            logging.debug(f"Product ID '{product_id}' from parameter is present in final URL")
            return True
    
    # If we can't find the product ID in the final URL, it's not preserved
    logging.debug(f"Product ID from original URL not found in final URL")
    return False


def get_bunnycdn_public_url(storage_path: str) -> str:
    """
    Construct public URL for BunnyCDN storage object using Pull Zone hostname.
    
    Args:
        storage_path: Path to the object in storage (e.g., "scraped_product_images/client/article/image.jpg")
        
    Returns:
        Public URL to access the stored object on BunnyCDN via Pull Zone
    """
    # Use Pull Zone hostname for public access (CDN URL)
    # Default to scraper.b-cdn.net, but can be overridden via environment variable
    pull_zone_url = os.getenv("BUNNYCDN_PULL_ZONE_URL", "https://scraper.b-cdn.net")
    # Ensure storage_path doesn't have leading slash
    storage_path = storage_path.lstrip('/')
    return f"{pull_zone_url}/{storage_path}"


def get_supabase_public_url(storage_path: str, supabase_url: str = None) -> str:
    """
    Construct public URL for Supabase Storage object.
    DEPRECATED: Use get_bunnycdn_public_url instead.
    
    Args:
        storage_path: Path to the object in storage (e.g., "scraped_product_images/client/article/image.jpg")
                     Note: This should NOT include the bucket name, just the path within the bucket
        supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
        
    Returns:
        Public URL to access the stored object
    """
    if supabase_url is None:
        supabase_url = os.getenv("SUPABASE_URL")
    
    if not supabase_url:
        raise ValueError("SUPABASE_URL not set in environment variables")
    
    # Remove trailing slash if present
    base_url = supabase_url.rstrip('/')
    # Storage path is relative to bucket, bucket name is "assets"
    return f"{base_url}/storage/v1/object/public/assets/{storage_path}"

# --- Database Interaction with Supabase ---

def get_assets_table_name() -> str:
    """
    Get the assets table name from environment variable, default to 'assets'.
    Allows temporary override for testing (e.g., 'onboarding_assets').
    """
    return os.getenv("SUPABASE_TABLE_NAME", "assets")

def fetch_new_uploads(client_name: str, supabase: Client) -> List[Dict[str, Any]]:
    """
    Fetches all rows for a given client from the assets table
    where 'is_scraped' is false (products that need to be scraped).
    """
    logging.info(f"Fetching unscraped products for client: {client_name}")
    try:
        table_name = get_assets_table_name()
        response = (
            supabase.table(table_name)
            .select("article_id, product_link, product_name, client, is_scraped, preview_image") # Include preview_image for image-based classification
            .eq("client", client_name)
            .eq("is_scraped", False)
            .execute()
        )

        # --- Enhanced Debug Logging ---
        logging.info(f"Supabase raw response received. Data length: {len(response.data)}")
        if response.data:
            logging.info(f"First item returned from Supabase: {response.data[0]}")
        # --- End Enhanced Debug Logging ---
        
        # Map Supabase columns to the keys our script expects
        rows = [
            {
                "sku": item.get("article_id"),
                "product_url": item.get("product_link"),
                "product_name": item.get("product_name"),
                "preview_image": item.get("preview_image"),  # Include preview_image for image-based classification
            }
            for item in response.data
        ]
        logging.info(f"Found {len(rows)} unscraped products to process.")
        return rows
    except Exception as e:
        logging.error(f"Error fetching data from Supabase: {e}")
        return []


def dampen_confidence_scores(confidence_scores: Dict[str, float], dampening_factor: float = 0.5) -> Dict[str, float]:
    """
    Dampen confidence scores by multiplying by a factor.
    Used when likely_null is True to reduce confidence in image-based tags.
    
    Args:
        confidence_scores: Dictionary of confidence scores (e.g., {'category': 0.9, 'style': 0.8})
        dampening_factor: Factor to multiply scores by (default 0.5 = reduce by 50%)
        
    Returns:
        Dictionary with dampened confidence scores
    """
    if not confidence_scores:
        return {}
    
    dampened = {}
    for key, value in confidence_scores.items():
        dampened[key] = max(0.0, min(1.0, value * dampening_factor))
    
    return dampened


async def update_single_item(
    row_data: Dict[str, Any], 
    result: DownloadResult, 
    tag_result: Optional[TagResult] = None,
    text_result: Optional[TextExtractionResult] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    classification_confidence: float = 0.0,
    classification_source: str = "unknown",
    likely_null: bool = False,
    supabase: Client = None
):
    """
    Updates a single item in the database immediately after processing.
    
    Args:
        likely_null: Whether the product link is likely null (404/dead link or redirect to different page).
                     This should be calculated in the calling function to include redirect detection.
    """
    sku = row_data.get("sku")
    if not sku:
        logging.warning("Skipping database update - missing SKU")
        return
    
    logging.info(f"Updating database for SKU: {sku}")
    logging.info(f"Result status: {result.status}, saved_count: {result.saved_count}")
    
    # Use the likely_null value passed from the caller (which includes redirect detection)
    # Don't recalculate here, as redirect detection happens in process_batch
    
    # Build comprehensive update payload
    base_update = {"is_scraped": True}  # Mark as scraped after successful processing
    # Only save image references if we actually uploaded images (likely_null = False)
    # This ensures we don't save image URLs from dead/404 pages
    if result.kept_links and not likely_null:
        base_update["reference"] = result.kept_links
    elif result.kept_links and likely_null:
        # Clear kept_links if likely_null to prevent saving bad images
        logging.warning(f"Not saving {len(result.kept_links)} image URLs due to likely_null=True")
        result.kept_links = []
    
    # Set likely_null flag
    base_update["likely_null"] = likely_null
    if likely_null:
        logging.warning(f"Product link flagged as likely_null (status: {result.status})")
    
    # Add category and subcategory if available
    if category:
        base_update["category"] = category
    if subcategory:
        base_update["subcategory"] = subcategory
    
    # Build tags JSONB with all data
    # When likely_null=True, don't save any tag data (keep tags column unpopulated)
    if likely_null:
        logging.warning(f"Page is likely_null (redirect or dead link) - skipping tag data save for SKU {sku}")
        base_update["tags"] = None  # Keep tags column unpopulated
    else:
        tags_data = {}
        # Normal flow: save both image and text data
        # Add image tags
        if tag_result and not tag_result.error_message:
            tags_data["image_tags"] = tag_result.tags
            tags_data["image_category_tags"] = tag_result.category_tags
            tags_data["image_style_tags"] = tag_result.style_tags
            tags_data["image_material_tags"] = tag_result.material_tags
            tags_data["image_color_tags"] = tag_result.color_tags
            tags_data["image_brand_tags"] = tag_result.brand_tags
            tags_data["image_confidence_scores"] = tag_result.confidence_scores
        
        # Add text data
        if text_result and not text_result.error_message:
            tags_data["text_product_type"] = text_result.product_type
            tags_data["text_product_type_confidence"] = text_result.product_type_confidence
            tags_data["text_specifications"] = text_result.specifications
            tags_data["text_dimensions"] = text_result.dimensions
            if text_result.price:
                tags_data["text_price"] = {
                    "value": text_result.price,
                    "currency": text_result.currency,
                    "confidence": text_result.price_confidence
                }
            tags_data["text_language"] = text_result.language_detected
        
        # Add classification metadata
        tags_data["classification_confidence"] = classification_confidence
        tags_data["classification_source"] = classification_source
        
        # Combine all tags (for backward compatibility)
        # Priority: measurement tags from text > image tags > product type
        all_tags = []
        
        # 1. Add measurement/dimension tags from text extraction (highest priority)
        if text_result and text_result.dimensions:
            dims = text_result.dimensions
            measurement_tags = []
            if dims.get('height'):
                measurement_tags.append(f"height-{dims['height']}{dims.get('unit', 'cm')}")
            if dims.get('width'):
                measurement_tags.append(f"width-{dims['width']}{dims.get('unit', 'cm')}")
            if dims.get('depth'):
                measurement_tags.append(f"depth-{dims['depth']}{dims.get('unit', 'cm')}")
            all_tags.extend(measurement_tags)
    
        # 2. Add image tags (already prioritized and limited to 20 in tag_generator)
        if tag_result and not tag_result.error_message:
            all_tags.extend(tag_result.tags)
        
        # 3. Add product type if not already included
        if text_result and text_result.product_type:
            if text_result.product_type not in all_tags:
                all_tags.append(text_result.product_type)
        
        # Remove duplicates and limit to 20 total (measurement tags prioritized)
        all_tags = list(dict.fromkeys(all_tags))
        if len(all_tags) > 20:
            # Keep measurement tags, trim others
            measurement_count = sum(1 for tag in all_tags if any(kw in tag.lower() for kw in ['height', 'width', 'depth', 'weight', 'size', 'dimension', 'cm', 'mm', 'm', 'kg', 'g', 'inch', 'in', 'ft']))
            if measurement_count < 20:
                measurement_tags = [t for t in all_tags if any(kw in t.lower() for kw in ['height', 'width', 'depth', 'weight', 'size', 'dimension', 'cm', 'mm', 'm', 'kg', 'g', 'inch', 'in', 'ft'])]
                other_tags = [t for t in all_tags if t not in measurement_tags]
                all_tags = measurement_tags + other_tags[:20 - measurement_count]
            else:
                # Keep first 20 measurement tags if they exceed limit
                measurement_tags = [t for t in all_tags if any(kw in t.lower() for kw in ['height', 'width', 'depth', 'weight', 'size', 'dimension', 'cm', 'mm', 'm', 'kg', 'g', 'inch', 'in', 'ft'])]
                all_tags = measurement_tags[:20]
        
        tags_data["all_tags"] = all_tags
        
        # Update tags column with all data
        base_update["tags"] = tags_data
    
    # Log update payload
    if likely_null:
        logging.info(f"Update payload: category={category}, subcategory={subcategory}, tags=None (likely_null=True)")
    else:
        logging.info(f"Update payload: category={category}, subcategory={subcategory}, tags keys={list(tags_data.keys())}")
    
    try:
        table_name = get_assets_table_name()
        (
            supabase.table(table_name)
            .update(base_update)
            .eq("article_id", sku)
            .execute()
        )
        logging.info(f"✓ Database updated for SKU {sku}")
        if category:
            logging.info(f"  Category: {category}/{subcategory} (confidence: {classification_confidence:.2f}, source: {classification_source})")
    except Exception as e:
        logging.error(f"Error updating database for SKU {sku}: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        
        # Try updating without category/subcategory if they don't exist
        try:
            fallback_update = {"is_scraped": True, "likely_null": likely_null}  # Mark as scraped even if processing failed
            # Only save image references if we actually uploaded images (likely_null = False)
            if result.kept_links and not likely_null:
                fallback_update["reference"] = result.kept_links
            fallback_update["tags"] = tags_data
            
            table_name = get_assets_table_name()
            (
                supabase.table(table_name)
                .update(fallback_update)
                .eq("article_id", sku)
                .execute()
            )
            logging.info(f"✓ Database updated for SKU {sku} (without category/subcategory columns)")
        except Exception as e2:
            logging.error(f"Failed to update database for SKU {sku}: {e2}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")


async def upload_images_to_bunnycdn_ftp(
    image_urls: List[str], 
    client_name: str, 
    article_id: str
) -> List[str]:
    """
    Upload images directly from URLs to BunnyCDN via FTP without local disk storage.
    Returns list of public URLs for uploaded images.
    
    Args:
        image_urls: List of image URLs to download and upload
        client_name: Client name for organizing files
        article_id: Article ID for organizing files
        
    Returns:
        List of public URLs for successfully uploaded images
    """
    uploaded_urls = []
    
    if not image_urls:
        logging.warning("No image URLs provided for upload")
        return uploaded_urls
    
    # BunnyCDN FTP credentials (from environment variables)
    ftp_host = os.getenv("BUNNYCDN_FTP_HOST", "storage.bunnycdn.com")
    ftp_port = int(os.getenv("BUNNYCDN_FTP_PORT", "21"))
    ftp_username = os.getenv("BUNNYCDN_FTP_USERNAME")
    ftp_password = os.getenv("BUNNYCDN_FTP_PASSWORD")
    
    # If BunnyCDN credentials are not set, return empty list (will fall back to Supabase)
    if not ftp_username or not ftp_password:
        logging.info("BunnyCDN credentials not set - will fall back to Supabase Storage")
        return []
    
    logging.info(f"Starting FTP upload of {len(image_urls)} images to BunnyCDN for client: {client_name}, article: {article_id}")
    
    # Download all images first
    downloaded_images = []
    async with aiohttp.ClientSession() as session:
        for i, image_url in enumerate(image_urls, 1):
            try:
                # Detect file extension and content type from URL
                file_extension = os.path.splitext(urlparse(image_url).path)[1].lower()
                if not file_extension:
                    file_extension = '.jpg'  # Default fallback
                
                filename = f"image_{i}{file_extension}"
                logging.info(f"Downloading {filename} from: {image_url}")
                
                async with session.get(image_url, timeout=30) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        if not image_data:
                            logging.error(f"Downloaded image data is empty: {image_url}")
                            continue
                        
                        # Validate image data before upload
                        if len(image_data) < 100:  # Very small files are likely not valid images
                            logging.warning(f"Image data too small ({len(image_data)} bytes) for {image_url}, skipping")
                            continue
                        
                        # Sanitize client_name and article_id for storage paths
                        sanitized_client = sanitize_for_storage_path(client_name)
                        sanitized_article_id = sanitize_for_storage_path(article_id)
                        # Create storage path
                        storage_path = f"scraped_product_images/{sanitized_client}/{sanitized_article_id}/{filename}"
                        
                        downloaded_images.append({
                            'filename': filename,
                            'data': image_data,
                            'storage_path': storage_path
                        })
                    else:
                        logging.warning(f"Failed to download image {image_url}: HTTP {response.status}")
            except Exception as e:
                logging.error(f"Error downloading image {image_url}: {e}")
                import traceback
                logging.error(f"Download error traceback: {traceback.format_exc()}")
    
    # Upload all downloaded images via FTP
    if downloaded_images:
        # Run FTP operations in executor since ftplib is synchronous
        loop = asyncio.get_event_loop()
        uploaded_urls = await loop.run_in_executor(
            None,
            _upload_to_bunnycdn_ftp_sync,
            downloaded_images,
            ftp_host,
            ftp_port,
            ftp_username,
            ftp_password
        )
    
    logging.info(f"Successfully uploaded {len(uploaded_urls)}/{len(image_urls)} images to BunnyCDN")
    return uploaded_urls


def _upload_to_bunnycdn_ftp_sync(
    downloaded_images: List[Dict[str, Any]],
    ftp_host: str,
    ftp_port: int,
    ftp_username: str,
    ftp_password: str
) -> List[str]:
    """
    Synchronous FTP upload function (runs in executor).
    
    Args:
        downloaded_images: List of dicts with 'filename', 'data', and 'storage_path'
        ftp_host: FTP hostname
        ftp_port: FTP port
        ftp_username: FTP username
        ftp_password: FTP password
        
    Returns:
        List of public URLs for successfully uploaded images
    """
    uploaded_urls = []
    ftp = None
    
    try:
        # Connect to BunnyCDN FTP
        logging.info(f"Connecting to BunnyCDN FTP: {ftp_host}:{ftp_port}")
        ftp = FTP()
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_username, ftp_password)
        logging.info("✓ Connected to BunnyCDN FTP")
        
        # Capture the starting directory (where FTP login puts us)
        try:
            starting_directory = ftp.pwd()
            logging.debug(f"Starting FTP directory: {starting_directory}")
        except Exception:
            # If pwd() fails, assume we're at root or can't determine
            starting_directory = None
            logging.debug("Could not determine starting FTP directory, will navigate from current location")
        
        # Upload each image
        for img_info in downloaded_images:
            filename = img_info['filename']
            image_data = img_info['data']
            storage_path = img_info['storage_path']
            
            try:
                # Reset to starting directory before processing each file
                # This ensures we always navigate from the same base path
                if starting_directory:
                    try:
                        ftp.cwd(starting_directory)
                    except Exception as reset_error:
                        logging.warning(f"Could not reset to starting directory {starting_directory}: {reset_error}")
                        # Continue anyway - might still work from current location
                
                # Create directory structure if needed
                path_parts = storage_path.split('/')
                directory_parts = [p for p in path_parts[:-1] if p]  # All parts except filename, skip empty
                
                # Navigate/create directory structure from starting directory
                for part in directory_parts:
                    try:
                        # Try to change to directory
                        ftp.cwd(part)
                    except Exception:
                        # Directory doesn't exist, create it
                        try:
                            ftp.mkd(part)
                            ftp.cwd(part)
                            logging.debug(f"Created directory: {part}")
                        except Exception as mkd_error:
                            # Directory might have been created by another process
                            # Try to change to it again
                            try:
                                ftp.cwd(part)
                            except Exception:
                                logging.warning(f"Could not create/access directory {part}: {mkd_error}")
                                raise
                
                # Upload file data
                image_file = BytesIO(image_data)
                ftp.storbinary(f'STOR {path_parts[-1]}', image_file)
                
                # Construct public URL
                public_url = get_bunnycdn_public_url(storage_path)
                uploaded_urls.append(public_url)
                logging.info(f"✓ Uploaded {filename} to BunnyCDN: {public_url}")
                
            except Exception as upload_error:
                logging.error(f"Failed to upload {filename}: {upload_error}")
                import traceback
                logging.error(f"Upload error traceback: {traceback.format_exc()}")
        
    except Exception as e:
        logging.error(f"FTP connection/upload error: {e}")
        import traceback
        logging.error(f"FTP error traceback: {traceback.format_exc()}")
    finally:
        if ftp:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass
    
    return uploaded_urls


async def upload_images_to_storage_direct(
    image_urls: List[str], 
    client_name: str, 
    article_id: str, 
    supabase: Client = None
) -> Tuple[List[str], str]:
    """
    Upload images directly from URLs to storage.
    Tries BunnyCDN FTP first, falls back to Supabase Storage if BunnyCDN credentials not set.
    Returns tuple of (list of public URLs, storage_type) where storage_type is 'bunnycdn' or 'supabase'.
    """
    # Try BunnyCDN FTP first
    bunnycdn_urls = await upload_images_to_bunnycdn_ftp(image_urls, client_name, article_id)
    
    # If BunnyCDN returned URLs, use those
    if bunnycdn_urls:
        return bunnycdn_urls, 'bunnycdn'
    
    # Fall back to Supabase Storage if BunnyCDN not configured
    if not supabase:
        logging.warning("Supabase client not provided, cannot upload to Supabase Storage")
        return [], 'none'
    
    logging.info(f"Falling back to Supabase Storage for {len(image_urls)} images")
    supabase_urls = await upload_images_to_supabase_storage(image_urls, client_name, article_id, supabase)
    return supabase_urls, 'supabase'


async def upload_images_to_supabase_storage(
    image_urls: List[str], 
    client_name: str, 
    article_id: str, 
    supabase: Client
) -> List[str]:
    """
    Upload images directly from URLs to Supabase Storage.
    Returns list of public URLs for uploaded images.
    """
    uploaded_urls = []
    
    if not image_urls:
        logging.warning("No image URLs provided for upload")
        return uploaded_urls
    
    logging.info(f"Starting Supabase Storage upload of {len(image_urls)} images for client: {client_name}, article: {article_id}")
    
    for i, image_url in enumerate(image_urls, 1):
        try:
            # Detect file extension and content type from URL
            file_extension = os.path.splitext(urlparse(image_url).path)[1].lower()
            if not file_extension:
                file_extension = '.jpg'  # Default fallback
            
            # Map extensions to content types
            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg', 
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.svg': 'image/svg+xml'
            }
            content_type = content_type_map.get(file_extension, 'image/jpeg')
            
            # Create storage path
            filename = f"image_{i}{file_extension}"
            sanitized_client = sanitize_for_storage_path(client_name)
            sanitized_article_id = sanitize_for_storage_path(article_id)
            storage_path = f"scraped_product_images/{sanitized_client}/{sanitized_article_id}/{filename}".strip('/')
            
            logging.info(f"Downloading and uploading {filename} from: {image_url}")
            
            # Download image data directly from URL
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        if not image_data:
                            logging.error(f"Downloaded image data is empty: {image_url}")
                            continue
                        
                        # Validate image data before upload
                        if len(image_data) < 100:
                            logging.warning(f"Image data too small ({len(image_data)} bytes) for {image_url}, skipping")
                            continue
                        
                        # Upload directly to Supabase Storage
                        try:
                            upload_response = supabase.storage.from_("assets").upload(
                                path=storage_path,
                                file=image_data,
                                file_options={"content-type": content_type, "upsert": "true"}
                            )
                            
                            if upload_response:
                                public_url = get_supabase_public_url(storage_path)
                                uploaded_urls.append(public_url)
                                logging.info(f"✓ Uploaded {filename} to Supabase Storage: {public_url}")
                            else:
                                logging.warning(f"✗ Failed to upload {filename} - no response from Supabase")
                                
                        except Exception as upload_error:
                            if "409" in str(upload_error) or "Duplicate" in str(upload_error):
                                # File exists, try to update it instead
                                logging.info(f"File {filename} already exists, updating instead...")
                                try:
                                    update_response = supabase.storage.from_("assets").update(
                                        path=storage_path,
                                        file=image_data,
                                        file_options={"content-type": content_type}
                                    )
                                    if update_response:
                                        public_url = get_supabase_public_url(storage_path)
                                        uploaded_urls.append(public_url)
                                        logging.info(f"✓ Updated {filename} in Supabase Storage: {public_url}")
                                    else:
                                        logging.warning(f"✗ Failed to update {filename} - no response from Supabase")
                                except Exception as update_error:
                                    logging.error(f"Failed to update existing file {filename}: {update_error}")
                            else:
                                logging.error(f"Upload error for {filename}: {upload_error}")
                    else:
                        logging.warning(f"Failed to download image from {image_url}: HTTP {response.status}")
                        
        except Exception as e:
            logging.error(f"Error processing {image_url}: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
    
    logging.info(f"Supabase Storage upload complete. {len(uploaded_urls)}/{len(image_urls)} images uploaded successfully")
    return uploaded_urls


# --- End Database Interaction ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOMAIN_PROFILES_FILE = "domain_profiles.json"
OUTPUT_DIR = "output"
RESULTS_FILE = "results.jsonl"


def load_domain_profiles() -> Dict[str, DomainProfile]:
    if not os.path.exists(DOMAIN_PROFILES_FILE):
        return {}
    try:
        with open(DOMAIN_PROFILES_FILE, "r") as f:
            data = json.load(f)
            # Handle old format gracefully
            if "domains" in data:
                data = data["domains"]
            return {domain: DomainProfile(**profile_data) for domain, profile_data in data.items()}
    except (json.JSONDecodeError, TypeError):
        logging.warning(f"Could not parse {DOMAIN_PROFILES_FILE}. Starting with empty profiles.")
        return {}

def save_domain_profiles(profiles: Dict[str, DomainProfile]):
    with open(DOMAIN_PROFILES_FILE, "w") as f:
        # Use asdict for dataclass serialization
        dump_data = {domain: asdict(profile) for domain, profile in profiles.items()}
        json.dump(dump_data, f, indent=2)


def append_result(result_data: Dict[str, Any]):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result_data) + "\n")


def get_domain_expected_category(domain: str) -> Optional[str]:
    """
    Get expected category based on domain name (strong signal).
    
    Returns:
        Expected category or None if domain doesn't strongly suggest a category
    """
    domain_lower = domain.lower()
    
    # Strong domain signals
    domain_mappings = {
        'carpetshop.co.il': 'Home Decor',
        'synsam.se': 'Eyewear',
        'mobelmastarna.se': 'Furniture',
        'soffadirekt.se': 'Furniture',
        'sono.se': 'Furniture',  # Office furniture
    }
    
    for domain_pattern, expected_category in domain_mappings.items():
        if domain_pattern in domain_lower:
            return expected_category
    
    return None


async def validate_individual_images(
    image_urls: List[str],
    category: str,
    subcategory: Optional[str],
    product_name: Optional[str],
    api_key: str
) -> List[int]:
    """
    Validate each image individually against category/subcategory using Gemini Vision API.
    Returns list of indices for images that match the category/subcategory.
    
    Args:
        image_urls: List of image URLs to validate
        category: Product category (e.g., "Furniture", "Eyewear")
        subcategory: Product subcategory (e.g., "Sofas", "Sunglasses") or None
        product_name: Product name for context
        api_key: Gemini API key
        
    Returns:
        List of indices (0-based) for images that match the category/subcategory
    """
    if not image_urls or len(image_urls) == 0:
        return []  # No images to validate
    
    if len(image_urls) == 1:
        return [0]  # Single image - assume valid (less risk, and we need at least one)
    
    matching_indices = []
    
    try:
        import google.generativeai as genai
        from tag_generator import AITagGenerator
        import io
        from PIL import Image
        
        # Download images for validation
        tag_generator = AITagGenerator(api_key=api_key)
        images_for_validation = []  # List of (original_index, image) tuples
        
        for idx, img_url in enumerate(image_urls[:20]):  # Limit to 20 images for cost control
            try:
                image_data = await tag_generator.download_image_for_analysis(img_url)
                if image_data:
                    image = Image.open(io.BytesIO(image_data))
                    images_for_validation.append((idx, image))  # Store original index
            except Exception as e:
                logging.warning(f"Failed to download image for validation {img_url}: {e}")
                continue
        
        if len(images_for_validation) == 0:
            logging.warning("No images could be downloaded for validation, keeping all images")
            return list(range(len(image_urls)))  # Default to keeping all if we can't validate
        
        # Build validation prompt - focus on PRIMARY product, allow logos/text overlays
        subcategory_text = f" (specifically {subcategory})" if subcategory else ""
        product_context = f"\nPRODUCT NAME: {product_name}" if product_name else ""
        
        validation_prompt = f"""Analyze this product image and determine if the PRIMARY/MAIN product shown matches the expected category.

EXPECTED PRODUCT TYPE: {category}{subcategory_text}{product_context}

TASK:
Check if the PRIMARY/MAIN product in this image is of type: {category}{subcategory_text}

CRITERIA:
- ✓ VALID: The PRIMARY product shown is {category}{subcategory_text}
- ✓ VALID: Logo/text overlays are OK as long as the main product matches
- ✓ VALID: Different angles/views of the same product type
- ✗ INVALID: The PRIMARY product is NOT {category}{subcategory_text}
- ✗ INVALID: Image shows navigation, headers, or page elements without a product
- ✗ INVALID: Image shows a completely different product category

Focus on the MAIN product in the image. Logos, watermarks, or text overlays in corners are acceptable as long as the primary product matches.

Return ONLY valid JSON:
{{
  "matches_category": <true or false>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<brief explanation>"
}}

IMPORTANT:
- "matches_category": true if the PRIMARY product matches {category}{subcategory_text}
- "confidence": How confident you are (0.0 = not confident, 1.0 = very confident)
- Focus on the main product, not logos/text overlays"""
        
        # Initialize Gemini model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Validate each image individually
        for idx, image in images_for_validation:
            try:
                response = model.generate_content(
                    [validation_prompt, image],
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    }
                )
            except Exception:
                # Fallback if response_mime_type not supported
                response = model.generate_content(
                    [validation_prompt, image],
                    generation_config={
                        "temperature": 0.1,
                    }
                )
            
            ai_response = response.text.strip()
            # Remove markdown code blocks if present
            if ai_response.startswith("```"):
                lines = ai_response.split("\n")
                ai_response = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
            
            result = json.loads(ai_response)
            matches = result.get("matches_category", False)
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "")
            
            if matches:
                matching_indices.append(idx)
                logging.debug(f"✓ Image {idx} matches {category}/{subcategory} (confidence: {confidence:.2f}) - {reason}")
            else:
                logging.info(f"✗ Image {idx} does not match {category}/{subcategory} (confidence: {confidence:.2f}) - {reason}")
        
        logging.info(f"Image validation: {len(matching_indices)}/{len(images_for_validation)} images match {category}/{subcategory}")
        return matching_indices
        
    except Exception as e:
        logging.error(f"Error validating individual images against category: {e}")
        import traceback
        logging.error(f"Validation error traceback: {traceback.format_exc()}")
        # On error, default to keeping all images (don't block on validation failure)
        return list(range(len(image_urls)))


async def classify_from_preview_image(
    preview_image_url: str,
    product_name: str,
    api_key: str
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Classify product category and subcategory from preview image using Gemini Vision API.
    
    Args:
        preview_image_url: URL of the preview image
        product_name: Product name for context
        api_key: Gemini API key
        
    Returns:
        Tuple of (category, subcategory, confidence) or (None, None, 0.0) if fails
    """
    try:
        import google.generativeai as genai
        from category_mapper import CATEGORIES_WITH_SUBCATEGORIES, VALID_CATEGORIES, validate_category, validate_subcategory
        from tag_generator import AITagGenerator
        
        # Use tag generator's image download method
        tag_generator = AITagGenerator(api_key=api_key)
        image_data = await tag_generator.download_image_for_analysis(preview_image_url)
        
        if not image_data:
            logging.warning(f"Failed to download preview image: {preview_image_url}")
            return None, None, 0.0
        
        # Prepare image for Gemini
        import base64
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image_data))
        
        # Build category taxonomy
        taxonomy_text = "\n".join([
            f"{i+1}. {category}:\n   - {', '.join(subcategories)}"
            for i, (category, subcategories) in enumerate(CATEGORIES_WITH_SUBCATEGORIES.items())
        ])
        
        # Create classification prompt
        prompt = f"""Analyze this product image and classify it into the appropriate category and subcategory.

PRODUCT NAME: {product_name}

APPROVED TAXONOMY (choose ONE category and ONE subcategory):
{taxonomy_text}

TASK:
1. Identify what type of product this is from the image
2. Match it to the most appropriate category and subcategory from the taxonomy above
3. Use the product name as additional context if helpful

Return ONLY valid JSON:
{{
  "category": "<one of the approved categories exactly as listed>",
  "subcategory": "<one of the approved subcategories for the chosen category exactly as listed>",
  "confidence": <float between 0.0 and 1.0>
}}

IMPORTANT: 
- "category" MUST be exactly one of: {', '.join(VALID_CATEGORIES)}
- "subcategory" MUST be one of the subcategories listed for the chosen category"""
        
        # Initialize Gemini model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')  # Use flash for faster/cheaper image analysis
        
        try:
            response = model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                }
            )
        except Exception:
            # Fallback if response_mime_type not supported
            response = model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": 0.1,
                }
            )
        
        ai_response = response.text.strip()
        # Remove markdown code blocks if present
        if ai_response.startswith("```"):
            lines = ai_response.split("\n")
            ai_response = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        
        result = json.loads(ai_response)
        category = result.get("category")
        subcategory = result.get("subcategory")
        confidence = float(result.get("confidence", 0.0))
        
        # Validate category is in approved list
        if category:
            if not validate_category(category):
                logging.warning(f"Gemini returned invalid category from image: {category}, using 'Others'")
                category = "Others"
                subcategory = None
                confidence = min(confidence, 0.5)
            elif subcategory and not validate_subcategory(subcategory, category):
                logging.warning(f"Gemini returned invalid subcategory '{subcategory}' for category '{category}', setting to None")
                subcategory = None
        
        if category:
            logging.info(f"Image-based classification: '{product_name}' → Category: '{category}', Subcategory: '{subcategory}' (confidence: {confidence:.2f})")
        
        return category, subcategory, confidence
        
    except Exception as e:
        logging.error(f"Error classifying from preview image: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, 0.0


def classify_product_weighted(
    text_result: TextExtractionResult,
    image_category_tags: List[str],
    image_category_confidence: float = 0.0,
    likely_null: bool = False,
    domain: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], float, str]:
    """
    Classify product into category and subcategory using text-based classification only.
    
    Image categories are ignored due to poor accuracy. Only text-based classification
    is used, with domain-based fallback when text classification fails.
    
    Args:
        text_result: Text extraction result with product type and confidence
        image_category_tags: Category tags from image analysis (ignored, kept for backward compatibility)
        image_category_confidence: Confidence score from image analysis (ignored, kept for backward compatibility)
        likely_null: If True, log that this is a likely 404 page
        domain: Domain name for domain-based fallback classification
        
    Returns:
        Tuple of (category, subcategory, final_confidence, source)
        source can be: "text", "domain", or "unknown"
    """
    # Get text-based category (now returned directly from Gemini)
    text_category = None
    text_subcategory = None
    text_confidence = text_result.category_confidence if hasattr(text_result, 'category_confidence') else text_result.product_type_confidence
    
    # Use category and subcategory directly if available (new approach), otherwise fall back to mapping
    if hasattr(text_result, 'category') and text_result.category:
        text_category = text_result.category
        text_subcategory = getattr(text_result, 'subcategory', None)  # Get subcategory if available
        if not validate_category(text_category):
            text_category = None
            text_subcategory = None
        elif text_subcategory and not validate_subcategory(text_subcategory, text_category):
            # Invalid subcategory, but keep category
            text_subcategory = None
    elif text_result.product_type:
        # Fallback: map product_type to category (backward compatibility)
        text_category, text_subcategory = map_product_type_to_category(text_result.product_type)
        if not validate_category(text_category):
            text_category = None
            text_subcategory = None
        elif not validate_subcategory(text_subcategory, text_category):
            text_subcategory = None  # No subcategories
    
    # Domain-based fallback: check if domain strongly suggests a category
    domain_expected_category = None
    if domain:
        domain_expected_category = get_domain_expected_category(domain)
        if domain_expected_category:
            logging.info(f"Domain {domain} strongly suggests category: {domain_expected_category}")
    
    # Classification logic: text-only approach (image categories are stored in DB but not used for classification)
    if text_category:
        # Text has category - use full text confidence
        final_confidence = text_confidence
        final_category = text_category
        final_subcategory = text_subcategory  # Use text subcategory (may be None)
        source = "text"
        
        # Validate against domain if available (for logging/debugging)
        if domain_expected_category and text_category != domain_expected_category:
            logging.warning(f"Domain validation: text category '{text_category}' doesn't match domain expectation '{domain_expected_category}'. Using text classification.")
        
        if likely_null:
            logging.info(f"Using text-only categorization (likely_null=True): {text_category}/{text_subcategory} (confidence: {final_confidence:.2f})")
    else:
        # No text category - try domain fallback
        if domain_expected_category:
            logging.info(f"No category from text, using domain fallback: {domain_expected_category}")
            final_category = domain_expected_category
            final_subcategory = None  # Can't determine subcategory from domain alone
            final_confidence = 0.6  # Moderate confidence for domain-only classification
            source = "domain"
        else:
            final_confidence = 0.0
            final_category = None
            final_subcategory = None
            source = "unknown"
    
    return final_category, final_subcategory, final_confidence, source


async def process_batch(rows: List[Dict[str, Any]], client_name: str, supabase: Client) -> None:
    """
    Processes each product row using the simplified crawl4AI + AI confirmation approach.
    Updates the database immediately after processing each product.
    Now includes AI-powered tag generation for products with successfully extracted images.
    """
    if not rows:
        logging.info("No new rows to process.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        logging.error("GEMINI_API_KEY is required for image extraction")
        return

    # Initialize tag generator
    tag_generator = AITagGenerator(api_key=api_key)

    # Process rows one-by-one with AI-powered image extraction
    for i, row in enumerate(rows, 1):
        sku = row.get('sku')
        url = row.get('product_url')
        name = row.get('product_name')
        preview_image = row.get('preview_image')
        
        # Handle products with empty product_link - use preview_image for classification
        if not url:
            if preview_image:
                logging.info(f"[{i}/{len(rows)}] Processing: {name or 'Unknown Product'} - No URL, using preview_image for classification")
                try:
                    # Classify from preview image
                    category, subcategory, confidence = await classify_from_preview_image(
                        preview_image_url=preview_image,
                        product_name=name or 'Unknown Product',
                        api_key=api_key
                    )
                    
                    # Update database with classification only (no images, no tags)
                    # Mark as scraped since we attempted to process it
                    if category:
                        base_update = {
                            "is_scraped": True,
                            "category": category,
                            "subcategory": subcategory,
                            "likely_null": True,  # Mark as likely_null since no URL
                            "tags": None  # No tags for image-only classification
                        }
                        
                        table_name = get_assets_table_name()
                        (
                            supabase.table(table_name)
                            .update(base_update)
                            .eq("article_id", sku)
                            .execute()
                        )
                        logging.info(f"✓ Classified from preview image: {category}/{subcategory} (confidence: {confidence:.2f})")
                    else:
                        # Mark as scraped and likely_null even if classification failed
                        table_name = get_assets_table_name()
                        (
                            supabase.table(table_name)
                            .update({"is_scraped": True, "likely_null": True})
                            .eq("article_id", sku)
                            .execute()
                        )
                        logging.warning(f"✗ Could not classify from preview image for {name}")
                    
                    continue  # Skip to next product
                except Exception as e:
                    logging.error(f"Error processing preview image for {name}: {e}")
                    # Mark as scraped and likely_null on error
                    try:
                        table_name = get_assets_table_name()
                        (
                            supabase.table(table_name)
                            .update({"is_scraped": True, "likely_null": True})
                            .eq("article_id", sku)
                            .execute()
                        )
                    except:
                        pass
                    continue
            else:
                logging.warning(f"Skipping row {i} with missing 'product_url' and no 'preview_image': {row}")
                # Mark as scraped and likely_null
                try:
                    (
                        supabase.table("assets")
                        .update({"is_scraped": True, "likely_null": True})
                        .eq("article_id", sku)
                        .execute()
                    )
                except:
                    pass
                continue
            
        try:
            # Validate URL format
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logging.warning(f"Skipping row {i} with invalid URL (missing scheme/netloc): {url}")
                continue
            if parsed.scheme not in ['http', 'https']:
                logging.warning(f"Skipping row {i} with invalid URL scheme: {url}")
                continue
        except Exception:
            logging.warning(f"Skipping row {i} with invalid URL: {row}")
            continue

        logging.info(f"[{i}/{len(rows)}] Processing: {name or 'Unknown Product'} - {url}")
        
        try:
            result = await learn_domain_selector(
                url=url,
                product_name=name or 'Unknown Product',
                output_dir=os.path.join(OUTPUT_DIR, sku or f'product_{i}'),
                api_key=api_key,
            )
            
            # Check if product link is likely null (404/dead link or redirect to different page)
            # Only upload images if likely_null is False (active page)
            likely_null = result.status in ['NOT_FOUND', 'FETCH_FAILED', 'INVALID_URL']
            
            # Check for redirects to different pages (path changed)
            # Only flag as likely_null if the product ID is NOT preserved in the redirect
            if result.final_url:
                from anchor_selector import urls_have_different_paths
                if urls_have_different_paths(url, result.final_url):
                    # Check if product ID is preserved in the redirect
                    if sku:
                        product_id_preserved = is_product_id_preserved_in_redirect(url, result.final_url, sku)
                        logging.debug(f"Checking product ID preservation: sku='{sku}', original_url='{url}', final_url='{result.final_url}', preserved={product_id_preserved}")
                        if product_id_preserved:
                            logging.info(f"URL redirected but product ID preserved: {url} → {result.final_url}, allowing redirect")
                            # Don't flag as likely_null - this is a valid redirect (e.g., short URL → full product page)
                        else:
                            logging.warning(f"URL redirected to different page (product ID '{sku}' not found in final URL): {url} → {result.final_url}, flagging as likely_null")
                            likely_null = True
                    else:
                        logging.warning(f"URL redirected but no SKU available to check: {url} → {result.final_url}, flagging as likely_null")
                        likely_null = True
            
            # Upload images directly to Supabase Storage from URLs
            # ONLY if likely_null is False (we have an active product page)
            storage_urls = []
            storage_type = 'none'
            tag_result = None
            original_image_urls = result.kept_links.copy() if result.kept_links else []  # Store original URLs for validation
            
            if likely_null:
                logging.warning(f"Skipping image upload for {name or 'Unknown Product'} - page is likely_null (status: {result.status})")
                logging.warning("Images will not be saved to prevent training data contamination from dead/404 pages")
                # Clear kept_links to prevent any image processing
                result.kept_links = []
            elif result.kept_links:
                logging.info(f"Page is active (likely_null=False), uploading {len(result.kept_links)} images")
                
                storage_urls, storage_type = await upload_images_to_storage_direct(
                    result.kept_links, 
                    client_name, 
                    sku or f'product_{i}',  # sku is the article_id
                    supabase
                )
                
                # Update result with storage URLs
                result.kept_links = storage_urls
                logging.info(f"Updated result.kept_links with {len(storage_urls)} storage URLs")
            else:
                logging.info(f"No image URLs to upload (kept_links: {len(result.kept_links)})")
            
            # Generate tags if we have successfully extracted images AND page is active
            # Skip tag generation if likely_null is True (to avoid wasting API calls on bad images from redirects/404s)
            if likely_null:
                logging.warning(f"Skipping tag generation for {name or 'Unknown Product'} - page is likely_null (redirect or dead link)")
                tag_result = None
            elif result.status == 'OK' and result.kept_links:
                logging.info(f"Generating AI tags for {name or 'Unknown Product'}")
                try:
                    tag_result = await tag_generator.generate_tags_for_product(
                        product_name=name or 'Unknown Product',
                        product_url=url,
                        image_urls=result.kept_links,
                        max_images=5  # Limit to 5 images to control costs
                    )
                    
                    if tag_result.error_message:
                        logging.warning(f"Tag generation failed for {name}: {tag_result.error_message}")
                    else:
                        logging.info(f"✓ Generated {len(tag_result.tags)} combined tags for {name}")
                        logging.info(f"  Categories: {tag_result.category_tags}")
                        logging.info(f"  Styles: {tag_result.style_tags}")
                        logging.info(f"  Materials: {tag_result.material_tags}")
                        logging.info(f"  Colors: {tag_result.color_tags}")
                except Exception as e:
                    logging.error(f"Error generating tags for {name}: {e}")
                    tag_result = None
            
            # Extract text data from HTML (reuse HTML from image scraper)
            # IMPORTANT: Always attempt text extraction, even for 404 pages (likely_null=True)
            # Classification can work from URL and product_name alone, which is critical for 404 pages
            text_result = None
            category = None
            subcategory = None
            classification_confidence = 0.0
            classification_source = "unknown"
            
            # Always try text extraction - it can work with minimal/empty HTML for 404 pages
            # The detect_product_type function prioritizes domain/URL/product_name when HTML is minimal
            html_for_extraction = result.html or ""  # Use empty string if HTML is None
            logging.info(f"Extracting text data for {name or 'Unknown Product'} (likely_null={likely_null}, html_length={len(html_for_extraction) if html_for_extraction else 0})")
            
            try:
                text_result = await extract_all_text_data(
                    html=html_for_extraction,
                    url=url,
                    product_name=name or 'Unknown Product'
                )
                
                if text_result.error_message:
                    logging.warning(f"Text extraction failed for {name}: {text_result.error_message}")
                else:
                    logging.info(f"✓ Extracted text data for {name}")
                    logging.info(f"  Product Type: {text_result.product_type} (confidence: {text_result.product_type_confidence:.2f})")
                    if text_result.category:
                        logging.info(f"  Category: {text_result.category}/{text_result.subcategory} (confidence: {text_result.category_confidence:.2f})")
                    if text_result.specifications:
                        logging.info(f"  Specifications: {len(text_result.specifications)} items")
                    if text_result.price:
                        logging.info(f"  Price: {text_result.price} {text_result.currency or ''}")
            except Exception as e:
                logging.error(f"Error extracting text data for {name}: {e}")
                text_result = None
            
            # Classify product using weighted approach (60% text, 40% image)
            # For testing: store raw product_type in subcategory column
            raw_product_type = None
            category = None
            subcategory = None
            classification_confidence = 0.0
            classification_source = "unknown"
            
            if text_result and text_result.product_type:
                raw_product_type = text_result.product_type
            
            # likely_null is already determined above (before image upload)
            # Use it for confidence dampening in categorization
            # Always try to classify if we have text_result (even when likely_null=True, we should still classify from text)
            # Only skip if we have neither text nor image data
            if text_result or (tag_result and tag_result.category_tags):
                image_category_tags = tag_result.category_tags if tag_result else []
                
                # Get image category confidence - dampen if likely_null
                if tag_result and tag_result.confidence_scores:
                    image_category_confidence = tag_result.confidence_scores.get('category', 0.0)
                    # Dampen image confidence for categorization if likely_null
                    # Note: This only affects the image portion of categorization, not text-based categorization
                    if likely_null:
                        original_confidence = image_category_confidence
                        image_category_confidence = image_category_confidence * 0.5
                        logging.warning(f"Dampened image category confidence for classification due to likely_null: {original_confidence:.2f} -> {image_category_confidence:.2f}")
                else:
                    image_category_confidence = 0.0
                
                # Log image confidence for debugging
                if tag_result and tag_result.confidence_scores:
                    logging.info(f"Image confidence scores available: {tag_result.confidence_scores}")
                    logging.info(f"Using image category confidence: {image_category_confidence}")
                else:
                    logging.warning(f"No image confidence scores available (tag_result exists: {tag_result is not None})")
                
                # Extract domain from URL for domain-based validation
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower() if parsed_url.netloc else None
                
                # Note: Classification uses text_result (product_name, product_link) primarily.
                # Image categories are ignored due to poor accuracy.
                category, subcategory, classification_confidence, classification_source = classify_product_weighted(
                    text_result=text_result or TextExtractionResult(),
                    image_category_tags=image_category_tags,
                    image_category_confidence=image_category_confidence,
                    likely_null=likely_null,
                    domain=domain
                )
                
                # Use the subcategory from classification (already set above)
                # No need to override with raw_product_type since we now have proper subcategories
                
                if category:
                    logging.info(f"✓ Classified product: {category}/{subcategory} (confidence: {classification_confidence:.2f}, source: {classification_source})")
                    
                    # Validate images individually and filter to keep only matching images
                    # Only validate if we have category, multiple original images, storage URLs were uploaded, and not already flagged as likely_null
                    if original_image_urls and len(original_image_urls) > 1 and storage_urls and not likely_null:
                        matching_indices = await validate_individual_images(
                            image_urls=original_image_urls,  # Use original URLs for validation
                            category=category,
                            subcategory=subcategory,
                            product_name=name,
                            api_key=api_key
                        )
                        
                        if len(matching_indices) == 0:
                            # No images match - flag as likely_null
                            logging.warning(f"No images match classified category {category}/{subcategory}. Clearing storage URLs to prevent incorrect image storage.")
                            storage_urls = []  # Clear storage URLs (images already uploaded but won't be saved to DB)
                            result.kept_links = []  # Clear kept_links
                            likely_null = True  # Flag as problematic
                        elif len(matching_indices) < len(original_image_urls):
                            # Some images don't match - filter to keep only matching ones
                            # Ensure indices are valid and within bounds
                            valid_indices = [i for i in matching_indices if 0 <= i < len(original_image_urls) and i < len(storage_urls)]
                            
                            if len(valid_indices) == 0:
                                # No valid matching indices - flag as likely_null
                                logging.warning(f"No valid matching images after filtering. Clearing storage URLs.")
                                storage_urls = []
                                result.kept_links = []
                                likely_null = True
                            else:
                                filtered_original_urls = [original_image_urls[i] for i in valid_indices]
                                filtered_storage_urls = [storage_urls[i] for i in valid_indices]
                                
                                logging.info(f"Filtered images: {len(valid_indices)}/{len(original_image_urls)} match {category}/{subcategory}. Keeping {len(filtered_storage_urls)} matching images.")
                                
                                # Update with filtered lists
                                original_image_urls = filtered_original_urls
                                storage_urls = filtered_storage_urls
                                result.kept_links = filtered_storage_urls
                        else:
                            # All images match - no filtering needed
                            logging.info(f"All {len(matching_indices)} images match {category}/{subcategory}")
                else:
                    logging.warning(f"✗ Could not classify product (text: {text_result.product_type if text_result else 'N/A'}, image: {image_category_tags})")
            # Note: If classification failed, category and subcategory will both be None
            # This is expected behavior - we don't want to store invalid classifications
            
            # Update database immediately after processing this product
            await update_single_item(row, result, tag_result, text_result, category, subcategory, classification_confidence, classification_source, likely_null, supabase)
            
            # Log result summary
            if result.saved_count > 0:
                logging.info(f"✓ Successfully extracted {result.saved_count} images for {name or 'Unknown Product'}")
                if storage_urls:
                    storage_type_name = storage_type.capitalize() if storage_type != 'none' else 'Storage'
                    logging.info(f"✓ Uploaded {len(storage_urls)} images to {storage_type_name}")
                if tag_result and not tag_result.error_message:
                    logging.info(f"✓ Generated {len(tag_result.generated_tags)} AI tags")
            else:
                logging.warning(f"✗ No images extracted for {name or 'Unknown Product'} - Status: {result.status}")
            
            # Save result to JSONL
            result_dict = asdict(result)
            result_dict.pop("profile", None)
            result_dict.pop("html", None)  # Don't store HTML in JSONL (too large)
            result_data = {**result_dict, "sku": sku, "product_url": url, "product_name": name}
            
            # Add tag information to JSONL if available
            if tag_result and not tag_result.error_message:
                result_data.update({
                    "tags": tag_result.tags,  # Combined tags
                    "image_tags": tag_result.tags,
                    "image_category_tags": tag_result.category_tags
                })
            
            # Add text data to JSONL if available
            if text_result and not text_result.error_message:
                result_data.update({
                    "text_product_type": text_result.product_type,
                    "text_product_type_confidence": text_result.product_type_confidence,
                    "text_specifications_count": len(text_result.specifications) if text_result.specifications else 0,
                    "text_price": text_result.price,
                    "text_currency": text_result.currency
                })
            
            # Add classification results
            if category:
                result_data.update({
                    "category": category,
                    "subcategory": subcategory,
                    "classification_confidence": classification_confidence,
                    "classification_source": classification_source
                })
            
            append_result(result_data)
            
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            # Create error result
            error_result = DownloadResult(
                status='ERROR',
                ai_used=True,
                profile=DomainProfile(domain=urlparse(url).netloc.lower()),
                candidates_found=0,
                candidates_kept=0,
                saved_count=0,
                kept_links=[],
                html=None
            )
            # Update database immediately for failed items too (will set likely_null=True due to ERROR status)
            await update_single_item(row, error_result, None, None, None, None, 0.0, "error", True, supabase)

    logging.info(f"Batch processing complete. Processed {len(rows)} products.")


def main():
    parser = argparse.ArgumentParser(description="Batch process product image scraping for a specific client using Supabase.")
    parser.add_argument("client_name", type=str, help="The name of the client to process from the 'assets' table.")
    parser.add_argument("--dry-run", action="store_true", help="Run the script without updating the database.")
    args = parser.parse_args()

    load_dotenv()

    # Check for all required environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if not all([supabase_url, supabase_key, gemini_api_key]):
        logging.error("Missing required environment variables. Ensure SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY are set in your .env file.")
        return

    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)

    client_name = args.client_name
    new_rows = fetch_new_uploads(client_name, supabase)

    if not new_rows:
        logging.info(f"No new uploads found for client '{client_name}'. Exiting.")
        return
        
    # Run the async processing (database updates happen immediately during processing)
    asyncio.run(process_batch(new_rows, client_name, supabase))

    if args.dry_run:
        logging.info("\n--- DRY RUN COMPLETE ---")
        logging.info("No changes were made to the database.")
    
    logging.info("Script finished.")


if __name__ == "__main__":
    main()
