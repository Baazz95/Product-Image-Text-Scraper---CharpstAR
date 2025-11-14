"""
Text-based product extraction and classification using Google Gemini Pro

This module extracts product information from HTML using Gemini Pro:
- Product type detection
- Specifications extraction
- Price extraction
- Category/subcategory classification

Adapted from scrape_dimensions_gemini.py for integration with image scraping system.
"""

import json
import os
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Comment
import google.generativeai as genai

# Import utility functions from scrape_dimensions.py
from scrape_dimensions import remove_cookie_consent, unit_to_canonical

# Import category taxonomy
from category_mapper import VALID_CATEGORIES, CATEGORY_EXAMPLES, CATEGORIES_WITH_SUBCATEGORIES

logger = logging.getLogger(__name__)

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")  # Default to 2.5 Pro

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set. Text extraction will fail.")


@dataclass
class TextExtractionResult:
    """Result of text extraction from HTML"""
    product_type: Optional[str] = None  # Raw product identification (e.g., "massage chair", "dining table")
    product_type_confidence: float = 0.0  # Confidence for raw product type identification
    category: Optional[str] = None  # Final category classification (one of VALID_CATEGORIES)
    subcategory: Optional[str] = None  # Subcategory classification (one of the subcategories for the category)
    category_confidence: float = 0.0  # Confidence for category classification
    specifications: Dict[str, Any] = field(default_factory=dict)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    price: Optional[float] = None
    currency: Optional[str] = None
    price_confidence: float = 0.0
    language_detected: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


def extract_structured_data(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract structured data from HTML (JSON-LD, tables, definition lists)
    
    Returns:
        Dictionary with structured data sections as strings
    """
    structured = {}
    
    # 1. Extract JSON-LD (schema.org) data - highest priority
    json_ld_specs = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "{}")
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if isinstance(obj, dict):
                    # Extract product specifications from schema.org Product
                    if obj.get("@type") == "Product":
                        specs = {}
                        # Common schema.org properties
                        if "height" in obj:
                            specs["height"] = obj["height"]
                        if "width" in obj:
                            specs["width"] = obj["width"]
                        if "depth" in obj:
                            specs["depth"] = obj["depth"]
                        if "weight" in obj:
                            specs["weight"] = obj["weight"]
                        if "material" in obj:
                            specs["material"] = obj["material"]
                        if "color" in obj:
                            specs["color"] = obj["color"]
                        # Additional properties
                        if "additionalProperty" in obj:
                            for prop in obj["additionalProperty"]:
                                if isinstance(prop, dict) and "name" in prop and "value" in prop:
                                    specs[prop["name"]] = prop["value"]
                        if specs:
                            json_ld_specs.append(json.dumps(specs, ensure_ascii=False))
        except (json.JSONDecodeError, AttributeError, KeyError):
            continue
    
    if json_ld_specs:
        structured["json_ld"] = "\n".join(json_ld_specs)
    
    # 2. Extract specification tables
    table_specs = []
    for table in soup.find_all("table"):
        # Check if table looks like a specification table
        table_text = table.get_text(separator=" | ", strip=True).lower()
        spec_keywords = ["specifikation", "specification", "detalj", "detail", "mått", "dimension", "egenskap", "property"]
        if any(keyword in table_text for keyword in spec_keywords):
            # Extract table as structured text
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells and len(cells) >= 2:
                    rows.append(" | ".join(cells))
            if rows:
                table_specs.append("\n".join(rows))
    
    if table_specs:
        structured["tables"] = "\n\n".join(table_specs)
    
    # 3. Extract definition lists (dl/dt/dd pattern - common for specs)
    dl_specs = []
    for dl in soup.find_all("dl"):
        # Check if dl looks like specifications
        dl_text = dl.get_text(separator=" ", strip=True).lower()
        spec_keywords = ["specifikation", "specification", "detalj", "detail", "mått", "dimension"]
        if any(keyword in dl_text for keyword in spec_keywords) or len(dl.find_all("dt")) >= 3:
            # Extract dt/dd pairs
            pairs = []
            for dt in dl.find_all("dt"):
                term = dt.get_text(strip=True)
                dd = dt.find_next_sibling("dd")
                value = dd.get_text(strip=True) if dd else ""
                if term and value:
                    pairs.append(f"{term}: {value}")
            if pairs:
                dl_specs.append("\n".join(pairs))
    
    if dl_specs:
        structured["definition_lists"] = "\n\n".join(dl_specs)
    
    return structured


async def extract_specifications(html: str, url: str) -> Optional[Dict[str, Any]]:
    """
    Use Gemini Pro to extract product specifications using language-aware approach with structured data prioritization
    
    Returns:
        Dictionary with:
        - overall_dimensions: {height, width, depth, unit}
        - all_specifications: {key: value} - all other specifications
        - language_detected: detected language
        - specification_sections_found: list of section names found
        or None if extraction fails
    """
    try:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set")
            return None
        
        # Parse HTML to get clean text content
        soup = BeautifulSoup(html, "lxml")
        
        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Remove cookie consent banners first to prioritize product content
        soup = remove_cookie_consent(soup)
        
        # Extract structured data first (highest priority)
        structured_data = extract_structured_data(soup)
        
        # Extract free text from page (fallback for missing specs)
        # Get all text content, including from hidden elements (dropdowns, expandable sections)
        all_text_parts = []
        for element in soup.find_all(text=True):
            text = element.strip()
            if text and len(text) > 1:  # Skip single characters/spaces
                all_text_parts.append(text)
        
        # Also get text from divs that might contain specs
        for elem in soup.find_all("div", class_=lambda x: x and any(kw in str(x).lower() for kw in ["spec", "detail", "property", "info"])):
            elem_text = elem.get_text(separator=" ", strip=True)
            if elem_text and len(elem_text) > 10:
                all_text_parts.append(elem_text)
        
        # Combine free text
        free_text = " ".join(all_text_parts)
        
        # Limit free text to avoid excessive tokens (structured data is already limited)
        free_text = free_text[:10000]
        
        # Build structured data section for prompt
        structured_section = ""
        if structured_data:
            structured_section = "=== STRUCTURED DATA (HIGHEST PRIORITY - Use this first) ===\n"
            if structured_data.get("json_ld"):
                structured_section += f"\n[JSON-LD Schema.org Data]\n{structured_data['json_ld']}\n"
            if structured_data.get("tables"):
                structured_section += f"\n[Specification Tables]\n{structured_data['tables']}\n"
            if structured_data.get("definition_lists"):
                structured_section += f"\n[Definition Lists (dt/dd pairs)]\n{structured_data['definition_lists']}\n"
        else:
            structured_section = "=== STRUCTURED DATA ===\n(No structured data found)\n"
        
        # Create language-aware specification extraction prompt with structured data prioritization
        prompt = f"""You are a product specification extraction assistant. Your task is to comprehensively extract product specifications from a product page.

PRIORITY ORDER (check in this order):
1. STRUCTURED DATA (highest priority - most reliable, use this first)
2. FREE TEXT (use only if structured data is missing information)

{structured_section}

=== FREE TEXT (Fallback - use if structured data is incomplete) ===
{free_text[:8000] if free_text else "(No free text available)"}

PROCESS:
1. Identify the page language (e.g., Swedish, English, Norwegian, Danish, German, French, Spanish)
2. PRIORITIZE structured data first - extract from JSON-LD, tables, and definition lists
3. If structured data is missing information, supplement with free text
4. Find specification sections by looking for keywords in that language:
   - Swedish: "specifikationer", "detaljer", "mått", "tekniska detaljer", "egenskaper", "produktdetaljer"
   - English: "specifications", "details", "technical", "measurements", "features", "product details"
   - Norwegian: "spesifikasjoner", "detaljer", "mål", "tekniske detaljer", "egenskaper", "produktdetaljer"
   - Danish: "specifikationer", "detaljer", "mål", "tekniske detaljer", "egenskaber", "produktdetaljer"
   - German: "Spezifikationen", "Details", "Technische Daten", "Maße", "Eigenschaften", "Produktdetails"
   - French: "spécifications", "détails", "technique", "mesures", "caractéristiques", "détails du produit"
   - Spanish: "especificaciones", "detalles", "técnico", "medidas", "características", "detalles del producto"
   - (adapt to other languages as appropriate if the page is in a different language)
5. Extract ALL specifications from these sections
6. Categorize measurements:
   - Overall dimensions: height, width, depth/length (the three main dimensions of the entire product)
   - Component measurements: any other measurements (lens width, bridge width, weight, etc.)

IMPORTANT:
- Use structured data as the PRIMARY source - it's more reliable and accurate
- Only use free text to fill in missing information not found in structured data
- Extract overall dimensions ONLY if they represent the full product size (not component parts)
- If only component measurements exist (e.g., lens width for glasses), leave overall dimensions as null
- Include ALL other specifications in the "all_specifications" object
- Convert all measurements to consistent units (default to cm for dimensions)
- Translate specification keys to English for consistency
- Preserve measurement values with their units where provided

Return ONLY valid JSON:
{{
  "overall_dimensions": {{
    "height": <number or null>,
    "width": <number or null>,
    "depth": <number or null>,
    "unit": "<mm|cm|m|in|ft>"
  }},
  "all_specifications": {{
    "<english_key>": "<value>",
    ... (all other specifications: component dimensions, weight, material, color, etc.)
  }},
  "language_detected": "<language>",
  "specification_sections_found": ["<section names found>"]
}}"""

        # Initialize Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Call Gemini API with JSON response format
        # Note: response_mime_type might not be supported in all models, so we'll try both
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                }
            )
        except Exception as e:
            # Fallback if response_mime_type not supported
            if "response_mime_type" in str(e).lower() or "not supported" in str(e).lower():
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                    }
                )
                # Try to extract JSON from response
                ai_response = response.text.strip()
                # Remove markdown code blocks if present
                if ai_response.startswith("```"):
                    lines = ai_response.split("\n")
                    ai_response = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
                try:
                    spec_data = json.loads(ai_response)
                    # Normalize unit
                    overall_dims = spec_data.get("overall_dimensions", {})
                    if overall_dims.get("unit"):
                        overall_dims["unit"] = unit_to_canonical(overall_dims["unit"])
                    return spec_data
                except json.JSONDecodeError:
                    raise e
            else:
                raise
        
        ai_response = response.text.strip()
        
        # Parse JSON response
        try:
            spec_data = json.loads(ai_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse Gemini response as JSON: {e}")
            logger.warning(f"Response: {ai_response[:500]}")
            return None
        
        # Normalize unit in overall_dimensions
        overall_dims = spec_data.get("overall_dimensions")
        if overall_dims and isinstance(overall_dims, dict) and overall_dims.get("unit"):
            overall_dims["unit"] = unit_to_canonical(overall_dims["unit"])
        
        return spec_data
        
    except Exception as e:
        logger.error(f"Error during Gemini specification extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def detect_product_type_raw(html: str, url: str) -> Tuple[Optional[str], float]:
    """
    Use Gemini Pro to identify the raw product type (e.g., "sofa", "glasses", "massage chair")
    This is the first step - just identifying what the product is, before categorization.
    
    Returns:
        Tuple of (product_type, confidence) or (None, 0.0) if detection fails
    """
    try:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set")
            return None, 0.0
        
        soup = BeautifulSoup(html, "lxml")
        soup = remove_cookie_consent(soup)
        
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path_segments = [seg for seg in parsed_url.path.strip("/").split("/") if seg]
        
        title = soup.find("title")
        title_text = title.get_text() if title else ""
        text = soup.get_text(separator=" ", strip=True)[:5000]
        
        # Prompt for raw product type identification (no category constraints)
        prompt = f"""Identify what type of product this is from the product page.

PRIORITY ORDER (check in this order):
1. URL PATH SEGMENTS (most reliable): {path_segments}
2. Page title: {title_text}
3. Page content: {text[:2000]}

TASK: Identify the specific product type (e.g., "sofa", "glasses", "massage chair", "dining table", "outdoor lamp", etc.)
- Be specific but concise (1-3 words typically)
- Use common product names
- Check URL path segments FIRST - they often contain product type keywords in any language
- Consider product function and form

Return ONLY a valid JSON object:
{{
  "product_type": "<specific product type, e.g., 'sofa', 'massage chair', 'dining table'>",
  "confidence": <float between 0.0 and 1.0>
}}"""
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                }
            )
        except Exception:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                }
            )
        
        ai_response = response.text.strip()
        if ai_response.startswith("```"):
            lines = ai_response.split("\n")
            ai_response = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        
        result = json.loads(ai_response)
        product_type = result.get("product_type", "unknown")
        confidence = float(result.get("confidence", 0.0))
        
        return product_type, confidence
        
    except Exception as e:
        logger.warning(f"Gemini raw product type detection failed: {e}")
        return None, 0.0


async def detect_product_type(html: str, url: str, product_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
    """
    Use Gemini Pro to identify raw product type and classify into category/subcategory in one call.
    
    Returns:
        Tuple of (product_type, category, subcategory, confidence) or (None, None, None, 0.0) if fails
    """
    try:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set")
            return None, None, None, 0.0
        
        # Handle None or empty HTML (for 404 pages)
        html = html or ""
        soup = BeautifulSoup(html, "lxml")
        
        # Remove cookie consent banners first to prioritize product content
        soup = remove_cookie_consent(soup)
        
        # Extract URL components (domain, path segments, full URL)
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower() if parsed_url.netloc else ""
        path_segments = [seg for seg in parsed_url.path.strip("/").split("/") if seg]
        full_url = url
        
        # Get key text for classification
        title = soup.find("title")
        title_text = title.get_text() if title else ""
        
        # Get some page text (limit to avoid excessive tokens)
        text = soup.get_text(separator=" ", strip=True)[:5000]
        
        # Extract product name from URL if available (for 404 pages)
        product_name_from_url = None
        if path_segments:
            # Last segment often contains product name
            last_segment = path_segments[-1]
            # Clean up URL-encoded and hyphenated names
            product_name_from_url = last_segment.replace('-', ' ').replace('_', ' ').title()
        
        # Use provided product_name if available (from database), otherwise use URL-derived name
        if product_name:
            product_name_context = product_name
        elif product_name_from_url:
            product_name_context = product_name_from_url
        else:
            product_name_context = None
        
        # Build category taxonomy with subcategories for prompt
        taxonomy_text = "\n".join([
            f"{i+1}. {category}:\n   - {', '.join(subcategories)}"
            for i, (category, subcategories) in enumerate(CATEGORIES_WITH_SUBCATEGORIES.items())
        ])
        
        # Check if page content is minimal (likely 404 or error page)
        is_minimal_content = len(text.strip()) < 500 or not text.strip()
        
        # Check for malformed URLs (multiple URLs concatenated) or image hosting sites
        is_malformed_url = " + " in url or " +" in url or "+ " in url
        image_hosting_domains = ["imgur.com", "i.imgur.com", "imgbb.com", "ibb.co", "postimg.cc", 
                                 "postimg.org", "tinypic.com", "photobucket.com", "flickr.com"]
        parsed_check = urlparse(url.lower())
        domain_check = parsed_check.netloc.lower()
        is_image_hosting = any(hosting_domain in domain_check for hosting_domain in image_hosting_domains)
        
        # Build prompt - simplified and focused
        if is_malformed_url or is_image_hosting:
            # For malformed URLs or image hosting sites: Strong warning to use low confidence
            content_note = "\n⚠️ CRITICAL WARNING: This URL appears to be malformed (multiple URLs concatenated) or points to an image hosting site (imgur, imgbb, etc.), NOT a product page. You should return VERY LOW confidence (0.2 or less) or return null category/subcategory. Do NOT classify based on URL text alone. If uncertain, use 'Others'/'Uncategorized' with low confidence."
        elif is_minimal_content:
            # For 404/error pages: Simple note to prioritize domain/URL/product_name
            content_note = "\n⚠️ NOTE: Page content is minimal (possibly 404/error page). Prioritize domain name, URL path segments, and product name for classification."
        else:
            # For normal pages: Use page content primarily, domain/URL to strengthen
            content_note = ""
        
        prompt = f"""Analyze this product page and return:
1. The raw product type (what it is - e.g., "sofa", "bed", "hot tub", "dining table", "carpet", "rug", "sunglasses", "glasses")
2. The category from the approved taxonomy
3. The subcategory from the approved taxonomy

PRODUCT PAGE CONTEXT:
- DOMAIN: {domain}
- FULL URL: {full_url}
- URL PATH SEGMENTS: {path_segments}
- PRODUCT NAME: {product_name_context or product_name_from_url or 'N/A'}
- Page title: {title_text}
- Page content: {text[:2000]}{content_note}

APPROVED TAXONOMY (choose ONE category and ONE subcategory):
{taxonomy_text}

TASK:
1. Identify the raw product type (1-3 words, e.g., "sofa", "carpet", "sunglasses", "dining table")
2. Match it to the most appropriate category and subcategory from the taxonomy above
3. Use domain name as a contextual signal (e.g., synsam.se = eyewear, carpetshop.co.il = carpets, mobelmastarna.se = furniture)
4. Use page content as the primary source when available, domain/URL/product_name when content is minimal

Use the identified raw product type as the primary guide when matching to the category and subcategory.
Return ONLY valid JSON:
{{
  "product_type": "<raw product type, e.g., 'sofa', 'bed', 'hot tub', 'carpet', 'sunglasses'>",
  "category": "<one of the approved categories exactly as listed>",
  "subcategory": "<one of the approved subcategories for the chosen category exactly as listed>",
  "confidence": <float between 0.0 and 1.0>
}}

IMPORTANT: 
- "category" MUST be exactly one of: {', '.join(VALID_CATEGORIES)}
- "subcategory" MUST be one of the subcategories listed for the chosen category"""
        
        # Initialize Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                }
            )
        except Exception:
            # Fallback if response_mime_type not supported
            response = model.generate_content(
                prompt,
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
        product_type = result.get("product_type")
        category = result.get("category")
        subcategory = result.get("subcategory")
        confidence = float(result.get("confidence", 0.0))
        
        # Detect malformed URLs (multiple URLs concatenated) or image hosting sites
        # These should never have high confidence as they're not real product pages
        is_malformed_url = False
        is_image_hosting = False
        
        # Check for multiple URLs concatenated with "+"
        if " + " in url or " +" in url or "+ " in url:
            is_malformed_url = True
            logger.warning(f"Malformed URL detected (multiple URLs concatenated): {url}")
        
        # Check for image hosting sites
        image_hosting_domains = ["imgur.com", "i.imgur.com", "imgbb.com", "ibb.co", "postimg.cc", 
                                 "postimg.org", "tinypic.com", "photobucket.com", "flickr.com"]
        parsed_check = urlparse(url.lower())
        domain_check = parsed_check.netloc.lower()
        for hosting_domain in image_hosting_domains:
            if hosting_domain in domain_check:
                is_image_hosting = True
                logger.warning(f"Image hosting site detected: {hosting_domain} in URL {url}")
                break
        
        # Force confidence dampening for malformed URLs or image hosting sites
        if is_malformed_url or is_image_hosting:
            if confidence > 0.3:
                logger.warning(f"Confidence too high ({confidence:.2f}) for malformed URL/image hosting site. Dampening to 0.3 or lower.")
                confidence = min(confidence, 0.3)  # Cap at 0.3 for unreliable sources
            # If confidence is already low, we can still use it but log a warning
            if category and category != "Others":
                logger.warning(f"Classification '{category}' from unreliable source (malformed URL/image hosting). Consider using 'Others'/'Uncategorized' instead.")
        
        # Validate category is in approved list
        if category:
            from category_mapper import validate_category, validate_subcategory
            if not validate_category(category):
                logger.warning(f"Gemini returned invalid category: {category}, using 'Others'")
                category = "Others"
                subcategory = None  # Reset subcategory if category is invalid
                confidence = min(confidence, 0.5)  # Lower confidence for invalid category
            elif subcategory and not validate_subcategory(subcategory, category):
                logger.warning(f"Gemini returned invalid subcategory '{subcategory}' for category '{category}', setting to None")
                subcategory = None  # Invalid subcategory, set to None
        
        # Log results for debugging
        if product_type:
            logger.info(f"Product identification: '{product_type}' → Category: '{category}', Subcategory: '{subcategory}' (confidence: {confidence:.2f})")
        
        return product_type, category, subcategory, confidence
        
    except Exception as e:
        logger.warning(f"Gemini product type detection failed: {e}")
        return None, None, None, 0.0


async def extract_price(html: str, url: str) -> Optional[Dict[str, Any]]:
    """
    Use Gemini Pro to extract product price and currency from HTML with confidence scoring
    
    Returns:
        Dictionary with 'price' (number), 'currency' (string), and 'confidence' (float 0-1)
        Returns None if confidence is below threshold (0.7) or if extraction fails
    """
    try:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set")
            return None
        
        soup = BeautifulSoup(html, "lxml")
        
        # Remove cookie consent banners first to prioritize product content
        soup = remove_cookie_consent(soup)
        
        # Extract product name/title for proximity checking
        product_name = None
        
        # Try to get product name from h1 (most common)
        h1 = soup.find("h1")
        if h1:
            product_name = h1.get_text(strip=True)
        
        # Fallback to title tag if no h1
        if not product_name:
            title = soup.find("title")
            if title:
                product_name = title.get_text(strip=True)
                # Clean up title (remove site name, etc.)
                if " | " in product_name:
                    product_name = product_name.split(" | ")[0]
                elif " - " in product_name:
                    product_name = product_name.split(" - ")[0]
        
        # Fallback to schema.org product name
        if not product_name:
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                try:
                    data = json.loads(script.string or "{}")
                    items = data if isinstance(data, list) else [data]
                    for obj in items:
                        if isinstance(obj, dict) and obj.get("@type") == "Product":
                            product_name = obj.get("name")
                            if product_name:
                                break
                except (json.JSONDecodeError, AttributeError):
                    continue
        
        # Get text content (limit to avoid excessive tokens)
        text = soup.get_text(separator=" ", strip=True)[:8000]
        
        # Build prompt with product name for context
        prompt = f"""Extract the product price and currency from this product page.

PRODUCT NAME: {product_name if product_name else "Not clearly identified"}

Page content:
{text}

IMPORTANT:
- Extract ONLY the price for the product named above
- The product price is typically the FIRST price listed on the page
- If multiple prices appear, prioritize the one closest to the product name
- If the page shows multiple products (e.g., related products, "you may also like" sections), be very careful
- Only extract price if you are confident it belongs to the main product
- Consider proximity: prices near the product name are more reliable
- Ignore shipping costs, taxes, or prices from other products
- Ignore prices that appear after related products or recommendation sections

Return ONLY a valid JSON object:
{{
  "price": <number or null>,
  "currency": "<currency_code or null>",
  "confidence": <float between 0.0 and 1.0>
}}

Confidence scoring:
- 0.9-1.0: Price is clearly associated with the product name (same section, close proximity)
- 0.7-0.89: Price likely belongs to the product (moderate confidence)
- 0.5-0.69: Uncertain - price might belong to the product but could be from related items
- Below 0.5: Low confidence - price may be from related products or error pages

Currency should be a 3-letter code (SEK, USD, EUR, GBP, etc.). Return the price and confidence score even if confidence is low - all prices will be stored, with low confidence prices marked for review."""
        
        # Initialize Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                }
            )
        except Exception:
            # Fallback if response_mime_type not supported
            response = model.generate_content(
                prompt,
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
        
        # Get confidence score (default to 0 if not provided)
        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
        
        # Validate price is a number if present
        price = result.get("price")
        if price is not None:
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = None
        
        # Return result if we have price
        if price is not None:
            currency = result.get("currency") or ""
            return {
                "price": price,
                "currency": currency,
                "confidence": confidence
            }
        
        return None
        
    except Exception as e:
        logger.warning(f"Gemini price extraction failed: {e}")
        return None


async def extract_all_text_data(html: str, url: str, product_name: str) -> TextExtractionResult:
    """
    Extract all text-based product data from HTML
    
    Args:
        html: HTML content of the product page
        url: Product URL
        product_name: Product name for context
        
    Returns:
        TextExtractionResult with all extracted data
    """
    import time
    start_time = time.time()
    
    result = TextExtractionResult()
    
    try:
        if not GEMINI_API_KEY:
            result.error_message = "GEMINI_API_KEY not set"
            return result
        
        # Identify raw product type and classify into category/subcategory in one AI call
        # Pass product_name to help with classification (especially for 404 pages)
        product_type, category, subcategory, category_conf = await detect_product_type(html, url, product_name=product_name)
        result.product_type = product_type
        result.product_type_confidence = category_conf  # Use same confidence for product_type
        result.category = category
        result.subcategory = subcategory
        result.category_confidence = category_conf
        
        # Extract specifications
        spec_data = await extract_specifications(html, url)
        if spec_data:
            result.specifications = spec_data.get("all_specifications", {})
            result.dimensions = spec_data.get("overall_dimensions", {})
            result.language_detected = spec_data.get("language_detected")
        
        # Extract price
        price_data = await extract_price(html, url)
        if price_data:
            result.price = price_data.get("price")
            result.currency = price_data.get("currency")
            result.price_confidence = price_data.get("confidence", 0.0)
        
    except Exception as e:
        result.error_message = f"Error during text extraction: {str(e)}"
        logger.error(f"Error extracting text data: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    result.processing_time = time.time() - start_time
    return result

