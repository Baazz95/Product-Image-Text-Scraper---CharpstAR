"""
Process products from a CSV file

This script reads product URLs from a CSV file and processes them through
the image scraping and classification pipeline.

CSV Format (required columns):
- article_id: Product SKU/ID
- product_link: Product URL to scrape
- product_name: Product name (optional)
- client: Client name

Example CSV:
article_id,product_link,product_name,client
SKU001,https://example.com/product1,Product 1,TestClient
SKU002,https://example.com/product2,Product 2,TestClient
"""

import argparse
import asyncio
import csv
import logging
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

from batch_processor import process_batch, update_single_item, get_assets_table_name
from anchor_selector import learn_domain_selector, DownloadResult
from tag_generator import AITagGenerator, TagResult
from text_extractor import extract_all_text_data, TextExtractionResult
from category_mapper import validate_category

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def read_csv_products(csv_path: str) -> List[Dict[str, Any]]:
    """
    Read products from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of product dictionaries with keys: sku, product_url, product_name, client
    """
    products = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        required_columns = ['article_id', 'product_link', 'client']
        if not all(col in reader.fieldnames for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}. Found: {', '.join(reader.fieldnames)}")
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
            article_id = row.get('article_id', '').strip()
            product_link = row.get('product_link', '').strip()
            product_name = row.get('product_name', '').strip() or None
            client = row.get('client', '').strip()
            
            if not article_id:
                logging.warning(f"Row {row_num}: Missing article_id, skipping")
                continue
            if not product_link:
                logging.warning(f"Row {row_num}: Missing product_link, skipping")
                continue
            if not client:
                logging.warning(f"Row {row_num}: Missing client, skipping")
                continue
            
            products.append({
                'sku': article_id,
                'product_url': product_link,
                'product_name': product_name,
                'client': client
            })
    
    logging.info(f"Read {len(products)} products from CSV file: {csv_path}")
    return products


async def ensure_product_in_db(product: Dict[str, Any], client_name: str, supabase: Client) -> bool:
    """
    Ensure product exists in database, insert if it doesn't exist
    
    Returns:
        True if product exists or was inserted, False on error
    """
    try:
        table_name = get_assets_table_name()
        # Check if product exists
        response = supabase.table(table_name).select("article_id").eq("article_id", product['sku']).execute()
        
        if response.data:
            # Product exists, update new_upload flag
            supabase.table(table_name).update({
                "new_upload": True,
                "product_link": product['product_url'],
                "product_name": product.get('product_name'),
                "client": client_name
            }).eq("article_id", product['sku']).execute()
            return True
        else:
            # Product doesn't exist, insert it
            supabase.table(table_name).insert({
                "article_id": product['sku'],
                "product_link": product['product_url'],
                "product_name": product.get('product_name'),
                "client": client_name,
                "new_upload": True
            }).execute()
            logging.info(f"✓ Inserted product into database: {product['sku']}")
            return True
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            # Product exists, just update
            try:
                table_name = get_assets_table_name()
                supabase.table(table_name).update({
                    "new_upload": True,
                    "product_link": product['product_url'],
                    "product_name": product.get('product_name'),
                    "client": client_name
                }).eq("article_id", product['sku']).execute()
                return True
            except Exception as e2:
                logging.error(f"Error updating product {product['sku']}: {e2}")
                return False
        else:
            logging.error(f"Error ensuring product {product['sku']} in database: {e}")
            return False


async def process_csv_products(products: List[Dict[str, Any]], supabase: Client, insert_to_db: bool = True) -> None:
    """
    Process products from CSV
    
    Args:
        products: List of product dictionaries
        supabase: Supabase client
        insert_to_db: If True, ensure products exist in database first (default: True)
    """
    if not products:
        logging.info("No products to process.")
        return
    
    # Group products by client
    products_by_client = {}
    for product in products:
        client = product['client']
        if client not in products_by_client:
            products_by_client[client] = []
        products_by_client[client].append(product)
    
    logging.info(f"Processing products for {len(products_by_client)} client(s)")
    
    # Process each client's products
    for client_name, client_products in products_by_client.items():
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing {len(client_products)} products for client: {client_name}")
            logging.info(f"{'='*60}\n")
            
            if insert_to_db:
                # Ensure all products exist in database
                logging.info(f"Ensuring {len(client_products)} products exist in database...")
                for product in client_products:
                    await ensure_product_in_db(product, client_name, supabase)
            
            # Process the products
            # Convert to format expected by process_batch (remove 'client' key)
            rows = [
                {
                    'sku': p['sku'],
                    'product_url': p['product_url'],
                    'product_name': p.get('product_name')
                }
                for p in client_products
            ]
            
            await process_batch(rows, client_name, supabase)
            
            logging.info(f"✓ Completed processing for client: {client_name}\n")
        except Exception as e:
            logging.error(f"✗ Error processing client '{client_name}': {e}")
            logging.exception(f"Full error traceback for client '{client_name}':")
            # Continue to next client instead of stopping
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Process products from a CSV file through the image scraping and classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Format (required columns):
  - article_id: Product SKU/ID
  - product_link: Product URL to scrape
  - product_name: Product name (optional)
  - client: Client name

Example CSV:
  article_id,product_link,product_name,client
  SKU001,https://example.com/product1,Product 1,TestClient
  SKU002,https://example.com/product2,Product 2,TestClient
        """
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to CSV file containing products to process"
    )
    parser.add_argument(
        "--no-insert-db",
        action="store_true",
        help="Skip inserting products into database (products must already exist in database)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (not implemented yet, but will skip database updates)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not all([supabase_url, supabase_key, gemini_api_key]):
        logging.error("Missing required environment variables. Ensure SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY are set in your .env file.")
        return
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        logging.error(f"CSV file not found: {args.csv_file}")
        return
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Read products from CSV
    try:
        products = read_csv_products(args.csv_file)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return
    
    if not products:
        logging.warning("No valid products found in CSV file.")
        return
    
    # Process products
    try:
        asyncio.run(process_csv_products(products, supabase, insert_to_db=not args.no_insert_db))
        logging.info("\n" + "="*60)
        logging.info("CSV batch processing complete!")
        logging.info("="*60)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()

