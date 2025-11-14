import os
import json
import asyncio
import aiohttp
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import google.generativeai as genai
import logging

@dataclass
class TagResult:
    """Result of tag generation for a product"""
    product_name: str
    product_url: str
    image_urls: List[str]
    tags: List[str] = field(default_factory=list)  # Combined list of all tags
    generated_tags: List[str] = field(default_factory=list)  # Deprecated, kept for compatibility
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    category_tags: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    material_tags: List[str] = field(default_factory=list)
    color_tags: List[str] = field(default_factory=list)
    brand_tags: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time: float = 0.0

class AITagGenerator:
    """AI-powered tag generator for product images"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
        # For testing purposes, allow initialization without API key
        
    async def download_image_for_analysis(self, image_url: str) -> Optional[bytes]:
        """Download image data for AI analysis"""
        try:
            # Check if this is a Supabase Storage URL
            is_supabase_url = 'supabase.co' in image_url or 'supabase' in image_url.lower()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            # For Supabase Storage URLs, use simpler headers (no Sec-Fetch headers that might cause issues)
            if not is_supabase_url:
                headers.update({
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Sec-Fetch-Dest': 'image',
                    'Sec-Fetch-Mode': 'no-cors',
                    'Sec-Fetch-Site': 'cross-site'
                })
                # Add Referer for external URLs to help with hotlink protection
                from urllib.parse import urlparse
                parsed = urlparse(image_url)
                if parsed.netloc:
                    headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"
            
            timeout = aiohttp.ClientTimeout(total=30)  # Increased timeout
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.get(image_url, allow_redirects=True) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        # Check if image is not too large (Vision API has limits)
                        if len(image_data) < 20 * 1024 * 1024:  # 20MB limit
                            return image_data
                        else:
                            logging.warning(f"Image too large for analysis: {len(image_data)} bytes (max 20MB)")
                    else:
                        logging.warning(f"Failed to download image {image_url}: HTTP {response.status}")
                        # Log response text for debugging
                        try:
                            error_text = await response.text()
                            if error_text:
                                logging.debug(f"Error response: {error_text[:200]}")
                        except:
                            pass
        except aiohttp.ClientError as e:
            logging.warning(f"Network error downloading image {image_url}: {e}")
        except Exception as e:
            logging.warning(f"Error downloading image {image_url}: {e}")
        return None

    async def generate_tags_for_product(
        self, 
        product_name: str, 
        product_url: str, 
        image_urls: List[str],
        max_images: int = 5
    ) -> TagResult:
        """
        Generate comprehensive tags for a product based on its images and metadata
        
        Args:
            product_name: Name of the product
            product_url: URL of the product page
            image_urls: List of image URLs to analyze
            max_images: Maximum number of images to analyze (to control costs)
            
        Returns:
            TagResult object with generated tags and metadata
        """
        import time
        start_time = time.time()
        
        result = TagResult(
            product_name=product_name,
            product_url=product_url,
            image_urls=image_urls
        )
        
        try:
            # Limit images to analyze to control costs
            images_to_analyze = image_urls[:max_images]
            
            if not images_to_analyze:
                result.error_message = "No images provided for analysis"
                return result
            
            # Download images for analysis
            downloaded_images = []
            for i, image_url in enumerate(images_to_analyze):
                image_data = await self.download_image_for_analysis(image_url)
                if image_data and len(image_data) <= 5 * 1024 * 1024:  # 5MB limit per image
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    downloaded_images.append({
                        "url": image_url,
                        "base64": base64_image,
                        "index": i + 1
                    })
                elif image_data and len(image_data) > 5 * 1024 * 1024:
                    logging.warning(f"Image {image_url} too large, skipping")
            
            if not downloaded_images:
                result.error_message = "No images could be downloaded for analysis"
                return result
            
            # Generate tags using AI
            tags = await self._analyze_images_with_ai(
                product_name, product_url, downloaded_images
            )
            
            # Parse and organize the tags
            result.generated_tags = tags.get('all_tags', [])
            result.category_tags = tags.get('category_tags', [])
            result.style_tags = tags.get('style_tags', [])
            result.material_tags = tags.get('material_tags', [])
            result.color_tags = tags.get('color_tags', [])
            result.brand_tags = tags.get('brand_tags', [])
            result.confidence_scores = tags.get('confidence_scores', {})
            
            # Log confidence scores for debugging
            if result.confidence_scores:
                logging.info(f"Image confidence scores: {result.confidence_scores}")
            else:
                logging.warning(f"No confidence scores returned from AI for {product_name}")

            # Combine all tags into a single list with prioritization
            # Priority: measurement/dimension tags > material > style > color > functional
            all_tags = []
            
            # 1. Measurement/dimension tags (highest priority) - extract from all_tags if they contain measurement keywords
            measurement_keywords = ['height', 'width', 'depth', 'length', 'weight', 'size', 'dimension', 'cm', 'mm', 'm', 'kg', 'g', 'inch', 'in', 'ft']
            measurement_tags = []
            other_tags = []
            
            # Check all tag categories for measurement-related tags
            for tag_list in [result.generated_tags, result.category_tags, result.style_tags, result.material_tags, result.color_tags, result.brand_tags]:
                for tag in tag_list:
                    tag_lower = tag.lower()
                    if any(keyword in tag_lower for keyword in measurement_keywords):
                        measurement_tags.append(tag)
                    else:
                        other_tags.append(tag)
            
            # Prioritize: measurement tags first, then others
            all_tags.extend(measurement_tags)
            all_tags.extend(other_tags)
            
            # Remove duplicates while preserving order
            all_tags = list(dict.fromkeys(all_tags))
            
            # Limit to 20 tags total (keep measurement tags, trim others if needed)
            if len(all_tags) > 20:
                # Keep all measurement tags, limit others
                measurement_count = len(measurement_tags)
                max_others = max(0, 20 - measurement_count)
                if measurement_count < 20:
                    result.tags = measurement_tags + other_tags[:max_others]
                else:
                    # If measurement tags exceed 20, keep first 20 measurement tags
                    result.tags = measurement_tags[:20]
            else:
                result.tags = all_tags
            
        except Exception as e:
            result.error_message = f"Error generating tags: {str(e)}"
            logging.error(f"Error generating tags for {product_name}: {e}")
        
        result.processing_time = time.time() - start_time
        return result

    async def _analyze_images_with_ai(
        self,
        product_name: str,
        product_url: str,
        downloaded_images: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze images using Gemini Vision API to generate comprehensive tags"""

        # Prepare images for Gemini
        images = []
        for img_data in downloaded_images:
            # Extract base64 data from data URL
            image_data_url = img_data['base64']
            if image_data_url.startswith("data:"):
                # Remove data:image/...;base64, prefix
                base64_data = image_data_url.split(",", 1)[1]
                import base64
                image_bytes = base64.b64decode(base64_data)

                # Create proper image object for Gemini
                import io
                from PIL import Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)

        # Prepare the prompt for Gemini
        prompt = f"""
PRODUCT TAG GENERATION REQUEST:

Product Name: {product_name}
Product URL: {product_url}

TASK: Analyze the following product images and generate comprehensive, accurate tags that describe:
1. Product category and type
2. Style and design characteristics
3. Materials and construction
4. Colors and patterns
5. Brand and model information
6. Use case and target audience
7. Key features and attributes

TAG GENERATION GUIDELINES:
- Generate up to 20 relevant tags (prioritize quality over quantity)
- Prioritize measurement/dimension tags over all other tags
- Use specific, descriptive terms
- Be accurate and avoid assumptions
- Use common industry terminology
- Include style descriptors (modern, vintage, minimalist, etc.)
- Include material information (leather, metal, wood, fabric, etc.)
- Include color information (black, brown, silver, etc.)
- Include brand/model information if visible
- Include functional attributes (adjustable, foldable, waterproof, etc.)

RESPONSE FORMAT:
Respond with valid JSON in this exact structure:
{{
  "all_tags": ["tag1", "tag2", "tag3", ...],
  "category_tags": ["category1", "category2", ...],
  "style_tags": ["style1", "style2", ...],
  "material_tags": ["material1", "material2", ...],
  "color_tags": ["color1", "color2", ...],
  "brand_tags": ["brand1", "model1", ...],
  "confidence_scores": {{
    "category": 0.95,
    "style": 0.88,
    "materials": 0.92,
    "colors": 0.90,
    "brand": 0.75
  }},
  "analysis_notes": "Brief explanation of key observations"
}}

IMPORTANT:
- Only return valid JSON, no additional text
- Be specific and accurate in your analysis
- Consider all images provided for a comprehensive view
- Use industry-standard terminology
- **REQUIRED: You MUST include confidence_scores for each category (category, style, materials, colors, brand)**
- Each confidence score should be between 0.0 and 1.0
- If you're very confident, use 0.8-1.0
- If moderately confident, use 0.5-0.7
- If uncertain, use 0.3-0.5
"""

        try:
            # Use Gemini Vision model (2.5-flash for better quality while maintaining speed)
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Generate content with Gemini
            if images:
                # Use vision model with images
                response = model.generate_content([prompt] + images)
            else:
                # Use text-only model
                response = model.generate_content(prompt)

            # Check if response has text content
            if hasattr(response, 'text') and response.text:
                content = response.text.strip()
            else:
                # Handle case where response doesn't have text
                return {
                    "all_tags": [],
                    "category_tags": [],
                    "style_tags": [],
                    "material_tags": [],
                    "color_tags": [],
                    "brand_tags": [],
                    "confidence_scores": {},
                    "analysis_notes": "Gemini API returned no text content"
                }

            # Clean up JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Parse JSON response
            parsed = json.loads(content)
            
            # Validate that confidence_scores exist, if not add defaults
            if 'confidence_scores' not in parsed or not parsed.get('confidence_scores'):
                logging.warning("AI response missing confidence_scores, adding default values")
                parsed['confidence_scores'] = {
                    'category': 0.7,
                    'style': 0.7,
                    'materials': 0.7,
                    'colors': 0.8,
                    'brand': 0.5
                }
            
            return parsed

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse AI response as JSON: {e}")
            logging.error(f"Raw response: {content}")
            return {
                "all_tags": [],
                "category_tags": [],
                "style_tags": [],
                "material_tags": [],
                "color_tags": [],
                "brand_tags": [],
                "confidence_scores": {},
                "analysis_notes": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            logging.error(f"Error in AI analysis: {e}")
            return {
                "all_tags": [],
                "category_tags": [],
                "style_tags": [],
                "material_tags": [],
                "color_tags": [],
                "brand_tags": [],
                "confidence_scores": {},
                "analysis_notes": f"Analysis error: {str(e)}"
            }

    async def generate_tags_batch(
        self, 
        products: List[Dict[str, Any]], 
        max_images_per_product: int = 5
    ) -> List[TagResult]:
        """
        Generate tags for multiple products in batch
        
        Args:
            products: List of product dictionaries with 'product_name', 'product_url', 'image_urls'
            max_images_per_product: Maximum images to analyze per product
            
        Returns:
            List of TagResult objects
        """
        results = []
        
        for i, product in enumerate(products, 1):
            product_name = product.get('product_name', 'Unknown Product')
            product_url = product.get('product_url', '')
            image_urls = product.get('image_urls', [])
            
            logging.info(f"[{i}/{len(products)}] Generating tags for: {product_name}")
            
            result = await self.generate_tags_for_product(
                product_name=product_name,
                product_url=product_url,
                image_urls=image_urls,
                max_images=max_images_per_product
            )
            
            results.append(result)
            
            # Log result summary
            if result.error_message:
                logging.warning(f"✗ Failed to generate tags for {product_name}: {result.error_message}")
            else:
                logging.info(f"✓ Generated {len(result.tags)} combined tags for {product_name}")
                logging.info(f"  Categories: {result.category_tags}")
                logging.info(f"  Styles: {result.style_tags}")
                logging.info(f"  Materials: {result.material_tags}")
                logging.info(f"  Colors: {result.color_tags}")
        
        return results

# Utility functions for tag processing
def save_tag_results(results: List[TagResult], filename: str = "tag_results.jsonl"):
    """Save tag generation results to JSONL file"""
    with open(filename, "w") as f:
        for result in results:
            result_dict = {
                "product_name": result.product_name,
                "product_url": result.product_url,
                "image_urls": result.image_urls,
                "tags": result.tags,  # Combined tags
                "generated_tags": result.generated_tags,  # Keep for compatibility
                "category_tags": result.category_tags,
                "style_tags": result.style_tags,
                "material_tags": result.material_tags,
                "color_tags": result.color_tags,
                "brand_tags": result.brand_tags,
                "confidence_scores": result.confidence_scores,
                "error_message": result.error_message,
                "processing_time": result.processing_time
            }
            f.write(json.dumps(result_dict) + "\n")

def load_tag_results(filename: str = "tag_results.jsonl") -> List[TagResult]:
    """Load tag generation results from JSONL file"""
    results = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    result = TagResult(
                        product_name=data.get("product_name", ""),
                        product_url=data.get("product_url", ""),
                        image_urls=data.get("image_urls", []),
                        tags=data.get("tags", data.get("generated_tags", [])),  # Support both formats
                        generated_tags=data.get("generated_tags", data.get("tags", [])),  # Compatibility
                        category_tags=data.get("category_tags", []),
                        style_tags=data.get("style_tags", []),
                        material_tags=data.get("material_tags", []),
                        color_tags=data.get("color_tags", []),
                        brand_tags=data.get("brand_tags", []),
                        confidence_scores=data.get("confidence_scores", {}),
                        error_message=data.get("error_message"),
                        processing_time=data.get("processing_time", 0.0)
                    )
                    results.append(result)
    return results

# Example usage
async def main():
    """Example usage of the tag generator with Gemini"""
    # Initialize the tag generator
    generator = AITagGenerator()

    # Example product data
    products = [
        {
            "product_name": "Assonet Display Cabinet Modern Black",
            "product_url": "https://www.newport.se/shop/mobler/forvaring/vitrinskap/assonet-vitrinskap-modern-black",
            "image_urls": [
                "https://cdn.newporthome.eu/thumbnail/8e/71/e2/1676282465/a85454f000774c37aab0c8490fb61210_1280x1600.jpg"
            ]
        }
    ]

    # Generate tags
    results = await generator.generate_tags_batch(products)

    # Save results
    save_tag_results(results)

    # Print results
    for result in results:
        print(f"\nProduct: {result.product_name}")
        print(f"Combined Tags: {result.tags}")
        print(f"Categories: {result.category_tags}")
        print(f"Styles: {result.style_tags}")

if __name__ == "__main__":
    asyncio.run(main())



