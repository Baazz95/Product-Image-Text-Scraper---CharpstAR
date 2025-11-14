import json
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Comment
try:
	from readability import Document  # type: ignore
except Exception: 
	Document = None


# Common unit patterns and symbols
UNIT_PATTERN = r"(?P<unit>mm|millimeter(?:s)?|cm|centimeter(?:s)?|m|meter(?:s)?|in|inch(?:es)?|\"|′|'|ft|foot|feet)"
UNIT_PATTERN_NO_NAMED = r"(?:mm|millimeter(?:s)?|cm|centimeter(?:s)?|m|meter(?:s)?|in|inch(?:es)?|\"|′|'|ft|foot|feet)"
SEP_PATTERN = r"[x×\*]\s*"
NUM_PATTERN = r"(?:(?:\d+[\.,]\d+)|\d+)"


DIM_PATTERNS: List[re.Pattern] = [
    # 3D patterns like 600 × 610 × 580 mm
    re.compile(rf"\b({NUM_PATTERN})\s*{SEP_PATTERN}({NUM_PATTERN})\s*{SEP_PATTERN}({NUM_PATTERN})\s*{UNIT_PATTERN}\b", re.I),
    # 2D patterns like 60 x 40 cm
    re.compile(rf"\b({NUM_PATTERN})\s*{SEP_PATTERN}({NUM_PATTERN})\s*{UNIT_PATTERN}\b", re.I),
]


LABEL_MAP = {
    # Swedish general
    "bredd": "width",
    "djup": "depth",
    "höjd": "height",
    "hojd": "height",
    "diameter": "diameter",
    "mått": "dimensions",
    "matt": "dimensions",
    # Swedish glasses-specific
    "totalbredd": "width",
    "glasbredd": "width",
    "glashöjd": "height",
    "glashojd": "height",
    "skalmlängd": "width",  # temple length, often width-like
    "skalmlangd": "width",
    "näsbrygga": "width",  # bridge width
    "nasbrygga": "width",
    # English
    "width": "width",
    "depth": "depth",
    "height": "height",
    "diameter": "diameter",
    "dimensions": "dimensions",
    "total width": "width",
    "lens width": "width",
    "lens height": "height",
    "temple length": "width",
    "bridge width": "width",
}


@dataclass
class Dimensions:
    width: Optional[float] = None
    depth: Optional[float] = None
    height: Optional[float] = None
    diameter: Optional[float] = None
    circumference: Optional[float] = None
    unit: Optional[str] = None


def remove_cookie_consent(soup: BeautifulSoup) -> BeautifulSoup:
	"""
	Remove cookie consent banners and related elements from HTML soup.
	
	This helps ensure that cookie consent text doesn't dominate the first
	characters of extracted text, leaving more room for actual product content.
	
	Args:
		soup: BeautifulSoup object to clean
		
	Returns:
		BeautifulSoup object with cookie consent elements removed
	"""
	# Common cookie consent identifiers (case-insensitive)
	cookie_keywords = [
		"cookie", "consent", "gdpr", "privacy", "iab", "ccpa",
		"cookie-consent", "cookie-banner", "cookie-notice",
		"cookiepolicy", "cookie-policy", "privacy-policy",
		"accept-cookies", "cookie-settings", "cookiepreferences",
		"samtycke", "cookies",  # Swedish
		"kakor", "integritet",  # Swedish
	]
	
	# Remove elements by ID
	for keyword in cookie_keywords:
		# Match IDs containing cookie keywords
		for elem in soup.find_all(id=lambda x: x and keyword.lower() in str(x).lower()):
			elem.decompose()
	
	# Remove elements by class
	for keyword in cookie_keywords:
		# Match classes containing cookie keywords
		for elem in soup.find_all(class_=lambda x: x and keyword.lower() in str(x).lower()):
			elem.decompose()
	
	# Remove elements by aria-label
	for keyword in cookie_keywords:
		for elem in soup.find_all(attrs={"aria-label": lambda x: x and keyword.lower() in str(x).lower()}):
			elem.decompose()
	
	# Remove elements by common cookie consent text patterns
	cookie_text_patterns = [
		"cookie", "samtycke", "accept cookies", "accept all",
		"cookie policy", "privacy policy", "gdpr", "ccpa",
		"we use cookies", "this website uses cookies",
		"vi använder cookies", "denna webbplats använder cookies",
		"cookie settings", "cookie preferences",
		"nödvändig", "inställningar", "statistik", "marknadsföring",  # Swedish cookie categories
	]
	
	# Find elements with cookie-related text and remove them
	for pattern in cookie_text_patterns:
		for elem in soup.find_all(text=lambda x: x and pattern.lower() in x.lower()):
			# Skip Comment and NavigableString objects (they don't have parent attribute reliably)
			if isinstance(elem, (Comment, str)):
				continue
			# Get parent element and remove it
			try:
				parent = elem.parent
				if parent:
					parent.decompose()
			except AttributeError:
				# Some text nodes don't have parent, skip them
				continue
	
	# Remove common cookie consent container patterns
	# These are often divs with specific structures
	common_selectors = [
		'div[id*="cookie"]',
		'div[class*="cookie"]',
		'div[id*="consent"]',
		'div[class*="consent"]',
		'div[id*="gdpr"]',
		'div[class*="gdpr"]',
		'div[id*="privacy-banner"]',
		'div[class*="privacy-banner"]',
		'div[id*="cookie-banner"]',
		'div[class*="cookie-banner"]',
	]
	
	for selector in common_selectors:
		try:
			for elem in soup.select(selector):
				elem.decompose()
		except Exception:
			# Selector might not be valid, skip
			pass
	
	return soup


def fetch_html(url: str, use_js: bool = False) -> str:
    """
    Fetch HTML from URL, optionally using Playwright to render JavaScript.
    
    Args:
        url: URL to fetch
        use_js: If True, use Playwright to render JavaScript (slower but handles dynamic content)
    
    Returns:
        HTML content as string
    """
    if use_js:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait a bit for any dynamic content to load
                page.wait_for_timeout(2000)
                
                # Try to expand common dropdown/accordion patterns
                # Look for common Swedish terms: "specifikationer", "specifikation", "detaljer", etc.
                try:
                    # Try to click common specification dropdowns
                    spec_selectors = [
                        'button:has-text("Specifikationer")',
                        'button:has-text("Specifikation")',
                        'button:has-text("Detaljer")',
                        '[aria-label*="specifikation" i]',
                        '[aria-label*="specification" i]',
                        '.accordion-button',
                        '.collapse-button',
                    ]
                    for selector in spec_selectors:
                        try:
                            elements = page.query_selector_all(selector)
                            for elem in elements:
                                if elem.is_visible():
                                    elem.click()
                                    page.wait_for_timeout(500)  # Wait for content to expand
                        except:
                            continue
                except:
                    pass  # If clicking fails, just continue
                
                html = page.content()
                browser.close()
                return html
        except ImportError:
            print("Warning: Playwright not installed. Install with: pip install playwright && playwright install chromium", file=sys.stderr)
            # Fall back to requests
        except Exception as e:
            print(f"Warning: Playwright failed: {e}. Falling back to requests.", file=sys.stderr)
            # Fall back to requests
    
    # Standard requests fetch (no JavaScript)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def normalize_number(s: str) -> float:
    s = s.replace(" ", "").replace(",", ".")
    return float(s)


def unit_to_canonical(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.lower().strip()
    if not u:
        return None
    if u in ['"', 'in', 'inch', 'inches']:
        return 'in'
    if u.startswith('mm'):
        return 'mm'
    if u.startswith('cm'):
        return 'cm'
    if u in ['m', 'meter', 'meters']:
        return 'm'
    if u in ['ft', 'foot', 'feet', '′', '’']:
        return 'ft'
    return u


def calculate_diameter_if_circular(dims: Dimensions, tolerance: float = 0.01) -> None:
    """If width ≈ depth (within tolerance), calculate diameter for circular items"""
    if dims.width is not None and dims.depth is not None and dims.diameter is None:
        # Check if width and depth are approximately equal (indicating circular)
        ratio = abs(dims.width - dims.depth) / max(dims.width, dims.depth) if max(dims.width, dims.depth) > 0 else 1.0
        if ratio <= tolerance:
            # Width and depth are the same, so it's circular - diameter = width (or depth)
            dims.diameter = dims.width


def calculate_circumference_if_circular(dims: Dimensions) -> None:
    """Calculate circumference from diameter if available"""
    import math
    if dims.diameter is not None and dims.circumference is None:
        # Circumference = π × diameter, rounded to 3 decimal places
        dims.circumference = round(math.pi * dims.diameter, 3)


# Non-AI extraction strategies removed - project now uses AI-only approach
# This file now only contains utility functions used by AI scripts


