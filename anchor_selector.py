import os
import re
import json
import asyncio
import aiohttp
import aiofiles
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin, parse_qsl, urlencode, urlunparse

from bs4 import BeautifulSoup, Tag
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
import google.generativeai as genai
from playwright.async_api import async_playwright


# -----------------------------
# Data models
# -----------------------------

@dataclass
class DomainProfile:
    domain: str
    selector: Optional[str] = None
    notes: Optional[str] = None
    precision: float = 0.0

@dataclass
class DownloadResult:
    status: str
    ai_used: bool
    profile: DomainProfile
    selector_used: Optional[str] = None
    candidates_found: int = 0
    candidates_kept: int = 0
    saved_count: int = 0
    kept_links: List[str] = field(default_factory=list)
    html: Optional[str] = None  # HTML content for text extraction
    final_url: Optional[str] = None  # Final URL after redirects (if different from original)


# -----------------------------
# Helpers
# -----------------------------

# Strict image size and shape filters (strict profile; adaptive relaxation applied later)
MIN_LONGEST_EDGE = 1000
MIN_AREA = 500_000
MAX_ASPECT_RATIO = 6.0  # allow even more flexible aspect ratios
MIN_SHORT_EDGE_NON_SQUARE = 100  # further reduce minimum size requirement
MIN_SHORTEST_EDGE_ABSOLUTE = 100  # further reduce minimum size requirement

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))

def _absolutize(base_url: str, src: str) -> str:
    if not src:
        return src
    if src.startswith('//'):
        return 'https:' + src
    if src.startswith('/'):
        p = urlparse(base_url)
        return f"{p.scheme}://{p.netloc}{src}"
    if not src.startswith('http'):
        return urljoin(base_url, src)
    return src


def _normalize_for_match(u: str) -> str:
    try:
        p = urlparse(u)
        return f"{p.scheme}://{p.netloc}{p.path}"
    except Exception:
        return u

def _normalize_url_for_path_comparison(url: str) -> Tuple[str, str]:
    """
    Normalize URL to extract domain and path for comparison.
    Removes scheme, query params, fragments, and normalizes trailing slashes.
    
    Returns:
        Tuple of (normalized_domain, normalized_path)
    """
    try:
        p = urlparse(url)
        # Normalize domain (remove www. prefix for comparison)
        domain = p.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Normalize path (remove trailing slash, lowercase)
        path = p.path.rstrip('/').lower()
        if not path:
            path = '/'
        
        return domain, path
    except Exception:
        return '', ''

def urls_have_different_paths(original_url: str, final_url: str) -> bool:
    """
    Check if two URLs have significantly different paths.
    Returns True if paths differ (indicating redirect to different page).
    
    Examples:
        - https://www.cargobike.se/produkt/abc → https://cargobikeofsweden.com/sv/articles/34/xyz
          Returns True (different domain AND different path)
        - https://www.cargobike.se/produkt/abc → http://www.cargobike.se/produkt/abc
          Returns False (same path, only scheme changed)
        - https://www.cargobike.se/produkt/abc → https://cargobike.se/produkt/abc?utm_source=...
          Returns False (same path, only query params added)
    """
    if not final_url:
        return False
    
    orig_domain, orig_path = _normalize_url_for_path_comparison(original_url)
    final_domain, final_path = _normalize_url_for_path_comparison(final_url)
    
    # If domains differ significantly (not just www vs non-www), it's a redirect
    if orig_domain != final_domain:
        logger.info(f"Domain changed: {orig_domain} → {final_domain}, flagging as redirect")
        return True
    
    # If paths differ, it's a redirect
    if orig_path != final_path:
        logger.info(f"Path changed: {orig_path} → {final_path}, flagging as redirect")
        return True
    
    return False

def looks_like_image_url(u: str) -> bool:
    """Heuristic: accept real image URLs or CDN-style image endpoints without extensions.
    - Ends with common image extensions (possibly with query/fragment)
    - OR contains well-known product image paths like '/media/catalog/product'
    - OR query params include width/height/format typical of image CDNs
    """
    try:
        ul = u.lower()
    except Exception:
        ul = u
    if re.search(r"\.(jpg|jpeg|png|webp|gif)(?:[?#].*)?$", ul):
        return True
    if '/media/catalog/product' in ul:
        return True
    if ('width=' in ul and 'height=' in ul) or ('format=pjpg' in ul) or ('auto=webp' in ul):
        return True
    return False

def _upgrade_url_for_size(u: str, target_width: int = 1200) -> Optional[str]:
    """Return an upgraded URL with larger width/height query params when present.
    Preserves aspect ratio if both width and height are provided.
    """
    try:
        p = urlparse(u)
        q = dict(parse_qsl(p.query, keep_blank_values=True))
        # Normalize param keys
        keys = {k.lower(): k for k in q.keys()}
        w_key = next((keys[k] for k in keys if k in ("w", "width")), None)
        h_key = next((keys[k] for k in keys if k in ("h", "height")), None)
        if not w_key and not h_key:
            return None
        new_q = q.copy()
        if w_key and h_key:
            try:
                ow = max(1, int(q[w_key]))
                oh = max(1, int(q[h_key]))
                ratio = oh / ow
                new_q[w_key] = str(target_width)
                new_q[h_key] = str(max(1, int(target_width * ratio)))
            except Exception:
                new_q[w_key] = str(target_width)
                if h_key:
                    new_q[h_key] = q[h_key]
        elif w_key:
            new_q[w_key] = str(target_width)
        elif h_key:
            new_q[h_key] = str(target_width)
        new_query = urlencode(new_q, doseq=True)
        upgraded = urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))
        return upgraded
    except Exception:
        return None


def expand_cdn_variants(urls: List[str], target_widths: Optional[List[int]] = None) -> List[str]:
    if target_widths is None:
        target_widths = [1200, 1600]
    expanded: List[str] = []
    for u in urls:
        if not looks_like_image_url(u):
            continue
        for tw in target_widths:
            upgraded = _upgrade_url_for_size(u, target_width=tw)
            if upgraded and upgraded != u:
                expanded.append(upgraded)
    # Keep unique order
    return list(dict.fromkeys(expanded))


def _canonical_basename(u: str) -> str:
    try:
        path = urlparse(u).path
        name = path.rsplit('/', 1)[-1]
        base = re.sub(r"\.(jpg|jpeg|png|webp|gif)$", "", name, flags=re.I)
        base = re.sub(r"-\d+x\d+", "", base)
        base = base.replace('-scaled', '').replace('-c-default', '')
        return base.lower()
    except Exception:
        return u.lower()


def _looks_like_404(html: str) -> bool:
    """
    Heuristically detect 404/Not Found pages by title/headings/content.
    Supports multiple languages: English, Scandinavian (Swedish, Norwegian, Danish),
    and key European languages (French, Spanish, German, Italian).
    """
    try:
        soup = BeautifulSoup(html or '', 'lxml')
        title = ((soup.title.string or '') if soup.title else '').strip().lower()
        h1 = (soup.find('h1').get_text(strip=True) if soup.find('h1') else '').lower()
        h2 = (soup.find('h2').get_text(strip=True) if soup.find('h2') else '').lower()
        
        # Get body text for more comprehensive checking
        body_text = soup.get_text(' ', strip=True).lower() if soup.body else ''
        
        error_texts = []
        for sel in ['.error', '.error404', '.not-found', '#error', '#error404', 
                    '.page-not-found', '.notfound', '#notfound', '.404']:
            n = soup.select_one(sel)
            if n:
                error_texts.append(n.get_text(strip=True).lower())
        
        # Combine all text for checking
        head = ' '.join([title, h1, h2] + error_texts)
        combined_text = f"{head} {body_text[:1500]}"  # Check first 1500 chars of body
        
        # Check for 404 in text
        if ' 404' in f" {combined_text}" or '404 ' in f"{combined_text} ":
            return True
        
        # Multi-language error phrases
        phrases = (
            # English
            'not found',
            'page not found',
            'page does not exist',
            'page cannot be found',
            'this page does not exist',
            'the page you are looking for',
            'page unavailable',
            'page not available',
            'product not found',
            'item not found',
            'no longer available',
            'no longer in stock',
            'discontinued',
            'out of stock',
            'product unavailable',
            
            # Swedish
            'sidan du försökte nå hittades tyvärr inte',
            'hittades tyvärr inte',
            'sidan hittades inte',
            'hoppsan',
            'sidan är försvunnen',  # "the page is gone"
            'åh nej, den här sidan verkar inte finnas',  # User's specific example
            'den här sidan verkar inte finnas',
            'produkten hittades inte',
            'produkten finns inte',
            'sidan finns inte längre',
            'produkten är inte längre tillgänglig',
            'produkten är slut',
            'produkten är inte i lager',
            
            # Norwegian
            'siden ble ikke funnet',
            'siden finnes ikke',
            'fant ikke siden',
            'produktet ble ikke funnet',
            'produktet finnes ikke',
            'produktet er ikke lenger tilgjengelig',
            'produktet er utsolgt',
            
            # Danish
            'siden blev ikke fundet',
            'siden findes ikke',
            'kunne ikke finde siden',
            'produktet blev ikke fundet',
            'produktet findes ikke',
            'produktet er ikke længere tilgængelig',
            'produktet er udsolgt',
            
            # French
            'page non trouvée',
            'page introuvable',
            'cette page n\'existe pas',
            'produit non trouvé',
            'produit introuvable',
            'produit indisponible',
            'produit n\'est plus disponible',
            'produit épuisé',
            'produit retiré',
            
            # Spanish
            'página no encontrada',
            'página no disponible',
            'esta página no existe',
            'producto no encontrado',
            'producto no disponible',
            'producto ya no está disponible',
            'producto agotado',
            'producto descontinuado',
            
            # German
            'seite nicht gefunden',
            'seite existiert nicht',
            'diese seite existiert nicht',
            'produkt nicht gefunden',
            'produkt existiert nicht',
            'produkt nicht mehr verfügbar',
            'produkt ausverkauft',
            'produkt nicht verfügbar',
            
            # Italian
            'pagina non trovata',
            'pagina non disponibile',
            'questa pagina non esiste',
            'prodotto non trovato',
            'prodotto non disponibile',
            'prodotto non è più disponibile',
            'prodotto esaurito',
            'prodotto discontinuato',
        )
        
        return any(p in combined_text for p in phrases)
    except Exception:
        return False

def _stable_selector_for_node(node) -> str:
    if node.has_attr('id') and node['id'] and not re.search(r"\b[0-9a-f]{6,}\b", node['id'], re.I):
        return f"#{node['id']}"
    # Keep only safe CSS class tokens (letters, digits, underscore, hyphen)
    raw_classes = node.get('class', []) or []
    safe_classes = [c for c in raw_classes if re.fullmatch(r"[A-Za-z0-9_-]+", c or "") and len(c) >= 3 and not re.search(r"\b[0-9a-f]{6,}\b", c, re.I)]
    if safe_classes:
        unique = '.'.join(sorted(list(dict.fromkeys(safe_classes)))[:2])
        return f"{node.name}.{unique}" if unique else node.name
    return node.name


def _nearest_common_ancestor_selector(soup: BeautifulSoup, kept_urls: List[str]) -> Optional[str]:
    if not kept_urls:
        return None
    norm_kept = { _normalize_for_match(u): u for u in kept_urls }
    canon_kept = { _canonical_basename(u) for u in kept_urls }

    matched_nodes = []
    for img in soup.find_all('img'):
        src = img.get('src') or ''
        if _normalize_for_match(src) in norm_kept or _canonical_basename(src) in canon_kept:
            matched_nodes.append(img)
            continue
        srcset = img.get('srcset') or ''
        for part in [p.strip() for p in srcset.split(',') if p.strip()]:
            u = part.split()[0]
            if _normalize_for_match(u) in norm_kept or _canonical_basename(u) in canon_kept:
                matched_nodes.append(img)
                break
    for el in soup.find_all(True, attrs={'style': True}):
        style = el.get('style') or ''
        m = re.search(r"background-image:\s*url\((['\"]?)([^'\")]+)\1\)", style, re.I)
        if m and (_normalize_for_match(m.group(2)) in norm_kept or _canonical_basename(m.group(2)) in canon_kept):
            matched_nodes.append(el)

    if not matched_nodes:
        return None

    def ancestors(n):
        res = []
        cur = n
        while cur and getattr(cur, 'name', None) not in (None, '[document]'):
            res.append(cur)
            cur = cur.parent
        return res

    anc_lists = [ancestors(n) for n in matched_nodes]
    common = set(anc_lists[0])
    for lst in anc_lists[1:]:
        common &= set(lst)
    container = max(common, key=lambda n: sum(1 for _ in n.parents)) if common else matched_nodes[0].parent
    return _stable_selector_for_node(container)


async def _download_images(urls: List[str], output_dir: str) -> int:
    """Asynchronously downloads a list of images to a directory."""
    if not urls:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for i, url in enumerate(urls):
            try:
                # Basic extension detection
                path = urlparse(url).path
                ext = os.path.splitext(path)[1]
                if not ext or len(ext) > 5:
                    ext = '.jpg'
            except Exception:
                ext = '.jpg'
            
            # Deterministic, safe filename to avoid overwrites across runs
            import hashlib
            base = os.path.splitext(os.path.basename(path) or '')[0] or f"image_{i+1}"
            base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:60]
            short = hashlib.md5(url.encode('utf-8', 'ignore')).hexdigest()[:8]
            filename = os.path.join(output_dir, f"{base}_{short}{ext}")
            tasks.append(_download_one(session, url, filename))
        
        results = await asyncio.gather(*tasks)
        saved_count = sum(1 for r in results if r)
    return saved_count

async def _download_one(session: aiohttp.ClientSession, url: str, filename: str) -> bool:
    """Downloads a single image."""
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                async with aiofiles.open(filename, mode='wb') as f:
                    await f.write(await response.read())
                return True
    except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
        logger.warning(f"Download failed for {url}: {e}")
    return False


async def _deduplicate_images_by_ai_comparison(image_urls: List[str], api_key: str) -> List[str]:
    """
    Use AI to compare images and keep only the highest resolution version of each unique image.
    """
    if len(image_urls) <= 1:
        return image_urls
    
    logger.info(f"AI comparing {len(image_urls)} images to find duplicates...")
    
    # Download images for comparison
    images_data = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/jpeg,image/png,image/webp;q=0.9,image/*;q=0.1,*/*;q=0.1',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async def fetch_one(u: str):
                try:
                    # Set a per-request Referer to the image origin to satisfy hotlink checks
                    from urllib.parse import urlparse
                    pu = urlparse(u)
                    referer = f"{pu.scheme}://{pu.netloc}/"
                    async with session.get(u, timeout=15, headers={'Referer': referer}) as response:
                        if response.status == 200:
                            ct = (response.headers.get('Content-Type') or '').lower()
                            if not ct.startswith('image/'):
                                return None
                            # Skip formats Pillow can't open by default
                            if 'image/avif' in ct or 'image/svg' in ct:
                                return None
                            data = await response.read()
                            if len(data) < 20 * 1024 * 1024:
                                import io
                                from PIL import Image
                                image = Image.open(io.BytesIO(data))
                                return {'url': u, 'image': image, 'size': image.size, 'pixels': image.size[0] * image.size[1]}
                except Exception as e:
                    logger.warning(f"Failed to download image {u}: {e}")
                return None
            results = await asyncio.gather(*[fetch_one(u) for u in image_urls])
            for item in results:
                if item:
                    images_data.append(item)
    except Exception:
        pass
    
    if len(images_data) <= 1:
        return image_urls
    
    # Prepare AI prompt for image comparison
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Create comparison prompt
    comparison_prompt = f"""
IMAGE COMPARISON TASK

INPUT:
You are given {len(images_data)} images of the same product (possibly different angles, settings, or duplicate resolutions).

OBJECTIVE:
Identify and group exact duplicate images. Keep the highest-resolution version from each duplicate group. Retain all unique images that show different perspectives, settings, or details.

CRITERIA:
- EXACT DUPLICATES: Same image content, same angle, same setting → only resolution differs.
- KEEP SEPARATE: Different angle (front, side, back, close-up, detail).
- KEEP SEPARATE: Different setting (indoor, outdoor, lifestyle, studio).
- KEEP SEPARATE: Different product view (overview, texture, features).
- ONLY REMOVE: True duplicates (identical images that differ only in resolution).

RULES:
- Always keep maximum resolution image in duplicate sets.
- Be conservative: if unsure, treat as unique (do not remove).
- Do not infer new URLs. Only use the provided list: {image_urls}.

RESPONSE FORMAT:
Return JSON ONLY in this structure:

{{
  "groups": [
    {{
      "group_id": 1,
      "urls": ["url1", "url2"],
      "kept_url": "url1",
      "reason": "Exact duplicate, kept highest resolution"
    }}
  ],
  "final_urls": ["url1", "url3", "url5"]
}}

REQUIREMENTS:
- Use unique sequential group_ids.
- "final_urls" must list all kept images after removing duplicates.
"""

    
    # Prepare images for AI analysis
    images_for_ai = [data['image'] for data in images_data]
    
    try:
        response = model.generate_content([comparison_prompt] + images_for_ai)
        
        if hasattr(response, 'text') and response.text:
            content = response.text.strip()
            if content.startswith('```json'): 
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            final_urls = parsed.get('final_urls', [])
            
            # Validate that returned URLs are in our original list
            validated_urls = [url for url in final_urls if url in image_urls]
            
            # Safety check: don't remove more than 70% of images
            removal_ratio = (len(image_urls) - len(validated_urls)) / len(image_urls)
            if removal_ratio > 0.9:
                logger.warning(f"AI removed too many images ({removal_ratio:.1%}), keeping original selection")
                return image_urls
            
            logger.info(f"AI deduplication: {len(image_urls)} → {len(validated_urls)} images")
            return validated_urls
        else:
            logger.warning("AI returned no text, using original images")
            return image_urls
            
    except Exception as e:
        logger.warning(f"AI comparison failed: {e}, using original images")
        return image_urls


def _is_openai_supported_url(u: str) -> bool:
    """Heuristic for URLs the Vision API accepts as images."""
    try:
        path = urlparse(u).path.lower()
        if any(path.endswith(ext) for ext in [".png", ".jpeg", ".jpg", ".gif", ".webp"]):
            return True
        q = (urlparse(u).query or '').lower()
        return any(k in q for k in ("format=pjpg", "format=jpeg", "auto=webp", "format=png"))
    except Exception:
        return False

async def _fetch_html(url: str) -> str:
    async with AsyncWebCrawler(verbose=False) as crawler:
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, exclude_external_images=False)
        result = await crawler.arun(url=url, config=config)
        return getattr(result, 'html', '') or ''


def _collect_candidates(soup_or_tag: Union[BeautifulSoup, Tag], base_url: str, media_images: Optional[List[Dict]] = None) -> List[str]:
    """Modified to accept either a full soup object or a specific container tag."""
    urls: List[str] = []

    # Crawl4AI media images first (usually high-signal)
    if media_images:
        for item in media_images:
            src = item.get('src') or ''
            if src:
                urls.append(_absolutize(base_url, src))
    
    # img/src + srcset + lazy loading attributes
    for img in soup_or_tag.find_all('img'):
        # Regular src
        if img.get('src'):
            urls.append(_absolutize(base_url, img['src']))
        # Lazy loading attributes (common in carousels)
        for attr in ['data-src', 'data-original', 'data-zoom-image', 'data-large-image', 'data-lazy']:
            if img.get(attr):
                urls.append(_absolutize(base_url, img[attr]))
        # srcset
        if img.get('srcset'):
            for part in [p.strip() for p in img['srcset'].split(',') if p.strip()]:
                u = part.split()[0]
                urls.append(_absolutize(base_url, u))

    # picture/source
    for source in soup_or_tag.find_all('source'):
        if source.get('srcset'):
            for part in [p.strip() for p in source['srcset'].split(',') if p.strip()]:
                u = part.split()[0]
                urls.append(_absolutize(base_url, u))

    # noscript
    for ns in soup_or_tag.find_all('noscript'):
        try:
            inner = BeautifulSoup(ns.decode_contents(), 'lxml')
            for img in inner.find_all('img'):
                if img.get('src'):
                    urls.append(_absolutize(base_url, img['src']))
        except Exception:
            pass

    # meta og/twitter
    for meta in soup_or_tag.find_all('meta'):
        prop = (meta.get('property') or meta.get('name') or '').lower()
        if prop in ('og:image', 'twitter:image') and meta.get('content'):
            urls.append(_absolutize(base_url, meta['content']))

    # background-image
    for el in soup_or_tag.find_all(True, attrs={'style': True}):
        style = el.get('style') or ''
        m = re.search(r"background-image:\s*url\((['\"]?)([^'\")]+)\1\)", style, re.I)
        if m:
            urls.append(_absolutize(base_url, m.group(2)))

    # Deduplicate and drop obvious UI/noise aggressively
    uniq = []
    seen = set()
    ui_keywords = (
        # 'logo','icon','sprite','banner','promo','placeholder','avatar','favicon','badge','sticker',
        # 'thumb','thumbnail','mini','small','preview','social','share','breadcrumb','pagination','zoom-icon'
    )
    # Enhanced related product keywords
    related_keywords = (
        'recommend', 'related', 'upsell', 'cross', 'similar', 'also', 'might', 'like',
        'liknande', 'gillar', 'du kanske', 'rekommender', 'andra köpte', 'you-might', 'also-like',
        'suggested', 'recommendation', 'similar-products', 'related-items', 'cross-sell'
    )
    bad_paths = ('/icons/', '/ui/', '/logos/', '/sprites/', '/svg/', '/favicon')
    
    for u in urls:
        nu = _normalize_for_match(u)
        if nu in seen:
            continue
        lower_u = u.lower()
        
        # Skip UI elements
        if any(k in lower_u for k in ui_keywords):
            continue
        if any(bp in lower_u for bp in bad_paths):
            continue
            
        # Skip related product images
        if any(k in lower_u for k in related_keywords):
            continue
            
        # Drop explicit tiny sizes in filename like 150x150, 200x200
        m = re.search(r"(\d{2,4})x(\d{2,4})", lower_u)
        if m:
            try:
                a, b = int(m.group(1)), int(m.group(2))
                if a < 400 and b < 400:
                    continue
            except Exception:
                pass
        seen.add(nu)
        uniq.append(u)
    return uniq


async def _collect_and_probe_in_browser(page_url: str, timeout_ms: int = 90000) -> Tuple[Optional[str], List[str], Dict[str, Tuple[int, int]], List[str], Optional[str]]:
    """
    Simplified approach: Use crawl4AI for HTML + media, then browser for dimension probing.
    Returns (html, candidates, dims, gallery_candidates, final_url).
    final_url is the URL after redirects (None if no redirect or same as original).
    """
    all_candidates: List[str] = []
    gallery_candidates: List[str] = []
    dims: Dict[str, Tuple[int, int]] = {}
    html: Optional[str] = None

    # First, get HTML and media images from crawl4AI
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, exclude_external_images=False)
            result = await crawler.arun(url=page_url, config=config)
            
            # Check HTTP status code for 404 or other error codes
            status_code = getattr(result, 'status_code', None)
            if status_code and status_code >= 400:
                # Return empty results for HTTP error codes (404, 500, etc.)
                logger.warning(f"HTTP {status_code} error for URL: {page_url}")
                return None, [], {}, [], None
            
            html = getattr(result, 'html', '') or ''
            media_images = (getattr(result, 'media', {}) or {}).get('images', []) if hasattr(result, 'media') else []
            
            if html:
                soup = BeautifulSoup(html, 'lxml')
                static_candidates = _collect_candidates(soup, page_url, media_images=media_images)
                all_candidates.extend(static_candidates)
    except Exception as e:
        logger.warning(f"Error fetching URL {page_url}: {e}")
        pass

    # Then use browser to collect additional candidates and probe dimensions
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1366, 'height': 2200}
        )
        page = await context.new_page()
        
        # Block non-essential resources but allow stylesheets for accurate layout/lazy-loading
        async def block_requests(route):
            if route.request.resource_type in ["font", "media", "manifest", "other"]:
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", block_requests)
        
        final_url: Optional[str] = None
        try:
            await page.goto(page_url, wait_until='domcontentloaded', timeout=timeout_ms)
            # Capture final URL after redirects
            final_url = page.url
            await page.wait_for_timeout(1200)

            # Dismiss popups and cookie banners
            try:
                await page.keyboard.press('Escape')
                await page.evaluate(
                    r"""() => {
                        const clickAll = (sel) => { document.querySelectorAll(sel).forEach(el => { try { el.click(); } catch {} }); };
                        const sels = [
                          '#onetrust-accept-btn-handler', '#onetrust-reject-all-handler', '.ot-sdk-show-settings',
                          'button[aria-label*="accept" i]', 'button[aria-label*="close" i]', 'button[aria-label*="dismiss" i]',
                          'button.cookie-accept', 'button.cookies-accept', '.cookie-accept', '.cookies-accept',
                          '.modal-close', '.close', '.Popup__close', '.Dialog__close', '.newsletter__close'
                        ];
                        sels.forEach(clickAll);
                        const hide = [
                          '#onetrust-banner-sdk', '.ot-sdk-container', '.cookie-banner', '.cookie-consent', 
                          '.newsletter-modal', '.modal', '.overlay'
                        ];
                        hide.forEach(sel => { 
                          document.querySelectorAll(sel).forEach(el => { 
                            try { 
                              el.style.setProperty('display','none','important'); 
                              el.style.setProperty('visibility','hidden','important'); 
                              el.style.setProperty('pointer-events','none','important'); 
                            } catch {} 
                          }); 
                        });
                    }"""
                )
                await page.wait_for_timeout(800)
            except Exception:
                pass

            # Let lazy content load
            try:
                await page.wait_for_load_state('networkidle', timeout=min(15000, timeout_ms))
            except Exception:
                pass
            
            # Gentle scroll to trigger lazy loaders
            try:
                await page.evaluate("""
                  async () => {
                    await new Promise(r => { 
                      let y = 0; 
                      const h = document.body.scrollHeight; 
                      const i = setInterval(()=>{ 
                        y += Math.max(200, window.innerHeight/2); 
                        window.scrollTo(0,y); 
                        if(y >= h){ clearInterval(i); r(); } 
                      }, 150); 
                    });
                  }
                """)
            except Exception:
                pass
            
            # Force lazy images to load by triggering their load events
            try:
                await page.evaluate("""
                  () => {
                    // Trigger lazy loading for all images with data-src
                    document.querySelectorAll('img[data-src]').forEach(img => {
                      if (img.getAttribute('data-src')) {
                        img.src = img.getAttribute('data-src');
                      }
                    });
                    
                    // Also trigger for carousel images specifically
                    document.querySelectorAll('.carousel img, .product-gallery img, #carousel img, .productzoom img').forEach(img => {
                      const dataSrc = img.getAttribute('data-src') || img.getAttribute('data-original');
                      if (dataSrc && !img.src) {
                        img.src = dataSrc;
                      }
                    });
                  }
                """)
                await page.wait_for_timeout(2000)  # Wait for images to load
                
                # Try to interact with carousels to load all images (click next buttons, etc.)
                try:
                    await page.evaluate("""
                      () => {
                        // Find and click carousel next buttons to load more images
                        const nextButtons = document.querySelectorAll(
                          '.carousel-next, .carousel__next, .slick-next, .swiper-button-next, ' +
                          '[aria-label*="next" i], [aria-label*="Next" i], ' +
                          '.product-gallery__next, .gallery-next, .image-gallery-next'
                        );
                        nextButtons.forEach((btn, idx) => {
                          if (idx < 5) { // Limit to first 5 carousels
                            try { btn.click(); } catch {}
                          }
                        });
                      }
                    """)
                    await page.wait_for_timeout(1500)  # Wait for carousel to advance
                    
                    # Click a few more times to cycle through
                    for _ in range(3):
                        await page.evaluate("""
                          () => {
                            document.querySelectorAll(
                              '.carousel-next, .slick-next, .swiper-button-next, ' +
                              '[aria-label*="next" i], .product-gallery__next'
                            ).forEach(btn => {
                              try { btn.click(); } catch {}
                            });
                          }
                        """)
                        await page.wait_for_timeout(800)
                except Exception:
                    pass
            except Exception:
                pass

            # Collect browser-visible candidates with aggressive filtering for related products
            browser_res = await page.evaluate(r"""() => {
                const out = new Set();
                const gallery = new Set();
                const abs = (u) => { try { return new URL(u, location.href).href; } catch { return null; } };
                
                // Enhanced filtering to exclude related/recommended products
                const isRelatedProductContext = (el) => {
                    if (!el) return false;
                    // Check for related product indicators in multiple languages
                    const relatedKeywords = [
                        'recommend', 'related', 'upsell', 'cross', 'similar', 'also', 'might', 'like',
                        'liknande', 'gillar', 'du kanske', 'rekommender', 'andra köpte', 'you-might', 'also-like',
                        'suggested', 'recommendation', 'similar-products', 'related-items', 'cross-sell'
                    ];
                    
                    // Check if element is in a product gallery/carousel (these should be included)
                    const isProductGallery = (node) => {
                        if (!node) return false;
                        const id = (node.id || '').toLowerCase();
                        const className = (node.className || '').toString().toLowerCase();
                        const galleryKeywords = [
                            'product-gallery', 'product-images', 'carousel', 'gallery', 'main-image',
                            'bildkolumn', 'productzoom', 'thumbs', 'thumbnail', 'carousel-outer', 
                            'productzoom-wrapper', 'carousel-normal', 'carousel-zoom'
                        ];
                        return galleryKeywords.some(keyword => 
                            id.includes(keyword) || className.includes(keyword)
                        );
                    };
                    
                    // If it's in a product gallery, don't exclude it
                    let node = el;
                    for (let i = 0; i < 5 && node; i++) {
                        if (isProductGallery(node)) {
                            return false; // Don't exclude product gallery images
                        }
                        node = node.parentElement;
                    }
                    
                    // Check element and its parents for related product indicators
                    node = el;
                    for (let i = 0; i < 8 && node; i++) {
                        const id = (node.id || '').toLowerCase();
                        const className = (node.className || '').toString().toLowerCase();
                        const textContent = (node.textContent || '').toLowerCase();
                        
                        // Check for related product keywords
                        if (relatedKeywords.some(keyword => 
                            id.includes(keyword) || 
                            className.includes(keyword) || 
                            textContent.includes(keyword)
                        )) {
                            return true;
                        }
                        
                        // Check for heading text that indicates related products
                        if (node.tagName && ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(node.tagName)) {
                            const headingText = textContent.trim();
                            if (headingText.includes('du kanske också gillar') || 
                                headingText.includes('you might also like') ||
                                headingText.includes('related products') ||
                                headingText.includes('similar items') ||
                                headingText.includes('recommended')) {
                                return true;
                            }
                        }
                        
                        node = node.parentElement;
                    }
                    return false;
                };
                
                const isUiContext = (el) => !!(el && el.closest('header, footer, nav, aside, .breadcrumb, .breadcrumbs, .social, .share, .logo, .icons, .thumbnails, .miniatures, .pagination'));
                const addIfMain = (el, u) => { 
                    const v = abs(u); 
                    if (v && !isUiContext(el) && !isRelatedProductContext(el)) {
                        out.add(v); 
                    }
                };
                const addIfGallery = (el, u) => {
                    const v = abs(u);
                    if (v) {
                        if (!isUiContext(el) && !isRelatedProductContext(el)) {
                            out.add(v);
                        }
                        gallery.add(v);
                    }
                };
                
                // Collect from various sources - prioritize carousel/gallery images
                // First, collect carousel images (these are usually the main product images)
                // Include common gallery/carousel selectors across platforms (broadened)
                const carouselSelectors = [
                    '.carousel img', '.carousel-item img', '.carousel-inner img',
                    '.product-gallery img', '.product-gallery__item img', '.product-gallery__image img',
                    '.product-images img', '.product-images__item img',
                    '#carousel img', '#product-carousel img',
                    '.productzoom img', '.productzoom-wrapper img', '.productzoom-item img',
                    '#Bildkolumn img', '.bildkolumn img',  // Swedish sites
                    '.carousel-outer img', '.carousel-inner img', '.carousel-normal img', '.carousel-zoom img',
                    '.gallery img', '.gallery__image img', '.gallery-item img',
                    '.swiper-slide img', '.swiper-wrapper img', '.swiper-container img',
                    '.slick-slide img', '.slick-track img', '.slick-list img',
                    '.product__media img', '.product-media img', '.product-media-wrapper img',
                    '.media-gallery img', '.media-gallery__item img',
                    '.product-photos img', '.product-photos__item img',
                    '.product__gallery img', '.product__image img', '.product-image img',
                    '.main-image img', '.main-image-wrapper img',
                    '.product-main-image img', '.product-main-image-wrapper img',
                    '.product-images-wrapper img', '.product-images-container img',
                    '.image-gallery img', '.image-gallery-item img',
                    '[class*="carousel"] img', '[class*="gallery"] img', '[class*="product-image"] img',
                    '[id*="carousel"] img', '[id*="gallery"] img', '[id*="product-image"] img',
                    // Generic: containers with multiple images (likely carousels)
                    'div:has(> img:nth-of-type(2)) img',  // Container with 2+ images
                    'section:has(> img:nth-of-type(2)) img'
                ].join(', ');
                
                try {
                    document.querySelectorAll(carouselSelectors).forEach(el => {
                        if (el.getAttribute('src')) addIfGallery(el, el.getAttribute('src'));
                        if (el.currentSrc) addIfGallery(el, el.currentSrc);
                        // Check lazy loading attributes
                        ['data-src', 'data-original', 'data-zoom-image', 'data-large-image', 'data-lazy', 'data-lazy-src'].forEach(attr => {
                            const src = el.getAttribute(attr);
                            if (src) addIfGallery(el, src);
                        });
                    });
                } catch (e) {
                    // Fallback if :has() selector not supported
                    document.querySelectorAll(
                      '.carousel img, .product-gallery img, .product-images img, #carousel img, ' +
                      '.productzoom img, .gallery img, .swiper-slide img, .slick-slide img, ' +
                      '.product__media img, .product-media img, .product-photos img, ' +
                      '.product__gallery img, .product__image img, .main-image img, ' +
                      '.product-main-image img, .product-images-wrapper img, .image-gallery img'
                    ).forEach(el => {
                        if (el.getAttribute('src')) addIfGallery(el, el.getAttribute('src'));
                        if (el.currentSrc) addIfGallery(el, el.currentSrc);
                        ['data-src', 'data-original', 'data-zoom-image', 'data-large-image', 'data-lazy'].forEach(attr => {
                            const src = el.getAttribute(attr);
                            if (src) addIfGallery(el, src);
                        });
                    });
                }
                
                // Alternative approach: Find containers with multiple images (likely carousels)
                // Look for divs/sections that contain 3+ images - these are often carousels
                try {
                    const allContainers = document.querySelectorAll('div, section, article, main');
                    allContainers.forEach(container => {
                        const images = container.querySelectorAll('img');
                        if (images.length >= 3 && images.length <= 20) {  // 3-20 images suggests a carousel
                            // Check if container has carousel-like classes/ids
                            const containerId = (container.id || '').toLowerCase();
                            const containerClass = (container.className || '').toString().toLowerCase();
                            const isLikelyCarousel = containerId.includes('product') || 
                                                   containerId.includes('gallery') ||
                                                   containerId.includes('image') ||
                                                   containerClass.includes('product') ||
                                                   containerClass.includes('gallery') ||
                                                   containerClass.includes('carousel') ||
                                                   containerClass.includes('image');
                            
                            if (isLikelyCarousel) {
                                images.forEach(img => {
                                    if (img.getAttribute('src')) addIfGallery(img, img.getAttribute('src'));
                                    if (img.currentSrc) addIfGallery(img, img.currentSrc);
                                    ['data-src', 'data-original', 'data-zoom-image', 'data-large-image'].forEach(attr => {
                                        const src = img.getAttribute(attr);
                                        if (src) addIfGallery(img, src);
                                    });
                                });
                            }
                        }
                    });
                } catch (e) {
                    // Ignore errors in fallback detection
                }
                
                // Then collect all other images
                document.querySelectorAll('img[src]').forEach(el => addIfMain(el, el.getAttribute('src')));
                document.querySelectorAll('img').forEach(el => { if (el.currentSrc) addIfMain(el, el.currentSrc); });
                document.querySelectorAll('img[srcset], source[srcset]').forEach(el => {
                    const ss = el.getAttribute('srcset') || '';
                    ss.split(',').map(s => s.trim().split(' ')[0]).forEach(u => addIfMain(el, u));
                });
                document.querySelectorAll('[data-src], [data-original], [data-zoom-image], [data-large-image]').forEach(el => {
                    const src = el.getAttribute('data-src') || el.getAttribute('data-original') || el.getAttribute('data-zoom-image') || el.getAttribute('data-large-image');
                    if (src) addIfMain(el, src);
                });
                
                // Computed background images
                document.querySelectorAll('*').forEach(el => {
                    const bg = getComputedStyle(el).backgroundImage || '';
                    const m = bg.match(/url\((['\"]?)([^'\")]+)\1\)/i);
                    if (m && m[2]) addIfMain(el, m[2]);
                });
                
                // JSON-LD product images (only from main product, not related)
                document.querySelectorAll('script[type="application/ld+json"]').forEach(s => {
                    try {
                        const data = JSON.parse(s.textContent);
                        const images = [];
                        const push = (v) => { if (!v) return; if (Array.isArray(v)) v.forEach(push); else images.push(v); };
                        if (Array.isArray(data)) data.forEach(item => { push(item.image); if (item.offers && item.offers.image) push(item.offers.image); });
                        else { push(data.image); if (data.offers && data.offers.image) push(data.offers.image); }
                        images.forEach(u => { const v = abs(u); if (v) out.add(v); });
                    } catch {}
                });
                
                return { all: Array.from(out), gallery: Array.from(gallery) };
            }""")
            
            # Add browser candidates that aren't already in our list
            for b in (browser_res.get('all') if isinstance(browser_res, dict) else []):
                if b not in all_candidates:
                    all_candidates.append(b)
            # Capture gallery candidates
            gallery_candidates = list(dict.fromkeys((browser_res.get('gallery', []) if isinstance(browser_res, dict) else [])))
            
            # Debug logging for carousel detection
            if gallery_candidates:
                logger.info(f"Found {len(gallery_candidates)} gallery/carousel images for {page_url}")
                logger.debug(f"Gallery images: {gallery_candidates[:3]}...")
            else:
                logger.warning(f"No gallery/carousel images detected for {page_url} - may need better selectors")

            # Filter out obvious UI elements
            ui_keywords = (
                'logo','icon','sprite','banner','promo','placeholder','avatar','favicon','badge','sticker',
                'mini','small','preview','social','share','breadcrumb','pagination','zoom-icon','ui/'
            )
            bad_paths = ('/icons/', '/ui/', '/logos/', '/sprites/', '/svg/', '/favicon')
            
            filtered_candidates = []
            filtered_gallery = []
            for u in all_candidates:
                lower = u.lower()
                if any(k in lower for k in ui_keywords):
                    continue
                if any(bp in lower for bp in bad_paths):
                    continue
                # Drop explicit tiny sizes in filename
                m = re.search(r"([0-9]{2,4})x([0-9]{2,4})", lower)
                if m:
                    try:
                        a, b = int(m.group(1)), int(m.group(2))
                        if a < 400 and b < 400:
                            continue
                    except Exception:
                        pass
                filtered_candidates.append(u)
            # Apply similar filtering to gallery list
            for u in gallery_candidates:
                lower = u.lower()
                if any(k in lower for k in ui_keywords):
                    continue
                if any(bp in lower for bp in bad_paths):
                    continue
                m = re.search(r"([0-9]{2,4})x([0-9]{2,4})", lower)
                if m:
                    try:
                        a, b = int(m.group(1)), int(m.group(2))
                        if a < 400 and b < 400:
                            continue
                    except Exception:
                        pass
                filtered_gallery.append(u)
            
            all_candidates = list(dict.fromkeys(filtered_candidates))
            gallery_candidates = list(dict.fromkeys(filtered_gallery))

            # Expand CDN variants and probe dimensions
            expanded = expand_cdn_variants(all_candidates)
            final_probe_set = list(dict.fromkeys(all_candidates + expanded))
            
            if final_probe_set:
                results = await page.evaluate(
                    r"""async (urls, t) => {
                        const abs = (u) => { try { return new URL(u, location.href).href; } catch { return null; } };
                        const domMap = new Map();
                        
                        // First, try to get dimensions from DOM images (including lazy loaded ones)
                        document.querySelectorAll('img').forEach(el => {
                          try {
                            // Check both src and data-src for lazy loaded images
                            const src = el.getAttribute('src') || el.getAttribute('data-src') || el.getAttribute('data-original');
                            if (!src) return;
                            const u = abs(src);
                            if (!u) return;
                            const w = el.naturalWidth || 0;
                            const h = el.naturalHeight || 0;
                            if ((w || h) && (!domMap.has(u) || (w*h) > ((domMap.get(u).w||0) * (domMap.get(u).h||0)))) {
                              domMap.set(u, { w, h });
                            }
                          } catch {}
                        });
                        const loadOne = (u) => new Promise((resolve) => {
                          let done = false;
                          const img = new Image();
                          const to = setTimeout(() => {
                            if (!done) { done = true; resolve({ url: u, w: 0, h: 0, ok: false, source: 'probe_to' }); }
                          }, t);
                          img.onload = () => {
                            if (!done) { done = true; clearTimeout(to); resolve({ url: u, w: img.naturalWidth, h: img.naturalHeight, ok: true, source: 'probe_onload' }); }
                          };
                          img.onerror = () => {
                            if (!done) { done = true; clearTimeout(to); resolve({ url: u, w: 0, h: 0, ok: false, source: 'probe_onerror' }); }
                          };
                          img.crossOrigin = 'anonymous';
                          img.referrerPolicy = 'no-referrer-when-downgrade';
                          img.src = u;
                        });
                        const results = [];
                        const toLoad = [];
                        for (const u0 of urls) {
                          const u = abs(u0);
                          if (!u) continue;
                          const d = domMap.get(u);
                          if (d && (d.w > 0 && d.h > 0)) {
                            results.push({ url: u, w: d.w, h: d.h, ok: true, source: 'dom' });
                          } else {
                            toLoad.push(u);
                          }
                        }
                        const B = 25;
                        for (let i = 0; i < toLoad.length; i += B) {
                          const chunk = toLoad.slice(i, i + B);
                          const r = await Promise.all(chunk.map((u) => loadOne(u)));
                          results.push(...r);
                        }
                        return results;
                    }""",
                    [final_probe_set, timeout_ms]
                )
                for r in results:
                    if isinstance(r, dict) and r.get('url') and isinstance(r['url'], str):
                        dims[r['url']] = (int(r.get('w') or 0), int(r.get('h') or 0))

        except Exception as e:
            pass
        finally:
            # Capture final URL if not already captured
            if final_url is None:
                try:
                    final_url = page.url
                except Exception:
                    pass
            await context.close()
            await browser.close()
    
    # Normalize final_url - only return if different from original
    normalized_final_url = None
    if final_url and final_url != page_url:
        # Check if it's actually different (not just scheme/query params)
        if urls_have_different_paths(page_url, final_url):
            normalized_final_url = final_url
        # If paths are the same, don't store it (it's just a scheme/param change)
    
    return html, all_candidates, dims, gallery_candidates, normalized_final_url


def _parse_dims_from_url(url: str) -> Optional[Tuple[int, int]]:
    """Tries to extract w,h dimensions from common URL patterns."""
    # Pattern 1: 437Wx649H or 300x200
    match = re.search(r'(\d+)[wWdD]x(\d+)[hH]', url) or re.search(r'(\d+)x(\d+)', url)
    if match:
        try:
            w, h = int(match.group(1)), int(match.group(2))
            if w > 100 and h > 100: # Sanity check
                return w, h
        except (ValueError, IndexError):
            pass
    # Pattern 2: width=600 or w=600 (enhanced for SoffaDirekt)
    w_match = re.search(r'[?&](?:width|w)=(\d+)', url)
    h_match = re.search(r'[?&](?:height|h)=(\d+)', url)
    if w_match:
        try:
            w = int(w_match.group(1))
            if h_match:
                h = int(h_match.group(1))
                return w, h
            # If only width is specified, assume reasonable aspect ratio for furniture
            if w >= 750:  # SoffaDirekt uses w=750, w=1000
                return w, int(w * 0.75)  # Assume 4:3 aspect ratio for furniture
            elif w >= 400:
                return w, int(w * 0.8)   # Assume 5:4 aspect ratio for smaller images
        except (ValueError, IndexError):
            pass
    
    # Pattern 3: SoffaDirekt specific - look for zoom images (usually high res)
    if '/zoom/' in url and any(ext in url for ext in ['.jpg', '.jpeg', '.png']):
        # Zoom images are typically high resolution
        if 'w=1000' in url:
            return 1000, 750  # Common SoffaDirekt zoom size
        elif 'w=750' in url:
            return 750, 563   # Mobile size
        else:
            return 1200, 900  # Default high-res assumption for zoom images
    
    return None


def _select_largest_per_family(urls: List[str], dims: Dict[str, Tuple[int, int]]) -> List[str]:
    groups: Dict[str, Tuple[str, int]] = {}
    for u in urls:
        key = _canonical_basename(u)
        w, h = dims.get(u, (0, 0))
        score = max(w, h)
        if key not in groups or score > groups[key][1]:
            groups[key] = (u, score)
    return [groups[k][0] for k in groups]


async def _filter_urls_by_content_type(urls: List[str]) -> List[str]:
    """Keep only URLs that respond with an image/* content-type (HEAD)."""
    if not urls:
        return []
    kept: List[str] = []
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def try_head(u: str):
            try:
                r = await session.head(u, allow_redirects=True)
                return (u, r)
            except Exception:
                return (u, None)
        results = await asyncio.gather(*[try_head(u) for u in urls])
        for u, r in results:
            if r is None:
                continue
            try:
                ct = (r.headers.get('Content-Type') or '').lower()
                if ct.startswith('image/'):
                    kept.append(str(r.url))
            except Exception:
                continue
    return kept

async def _get_filtered_candidates(html: str, base_url: str, initial_candidates: List[str], initial_dims: Dict[str, Tuple[int, int]], container_selector: Optional[str] = None) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Simplified filtering logic based on standalone approach.
    """
    all_candidates = initial_candidates
    # If a container selector is provided, scope candidates to that container.
    if container_selector:
        try:
            soup = BeautifulSoup(html or '', 'lxml')
            containers = soup.select(container_selector)
            if containers:
                scoped: List[str] = []
                for c in containers:
                    scoped.extend(_collect_candidates(c, base_url))
                # Keep order and intersect with initial candidates if any were present
                scoped_unique = list(dict.fromkeys(scoped))
                if scoped_unique:
                    if all_candidates:
                        allowed = set(all_candidates)
                        all_candidates = [u for u in scoped_unique if u in allowed]
                    else:
                        all_candidates = scoped_unique
        except Exception:
            pass
    dims = initial_dims.copy()

    # Fill in missing dimensions from URL patterns
    for u in all_candidates:
        w, h = dims.get(u, (0, 0))
        if w == 0 or h == 0:
            parsed_dims = _parse_dims_from_url(u)
            if parsed_dims:
                dims[u] = parsed_dims

    def filter_minimal(min_longest: int) -> List[str]:
        """Filter candidates by size threshold."""
        selected: List[str] = []
        for u in all_candidates:
            w, h = dims.get(u, (0, 0))
            if w <= 0 or h <= 0:
                continue
            longest = max(w, h)
            shortest = min(w, h)
            ratio = (w / h) if h else 0
            ratio = ratio if ratio >= 1 else (1 / ratio if ratio > 0 else 0)
            
            if shortest < MIN_SHORTEST_EDGE_ABSOLUTE:
                continue
            if longest < min_longest:
                continue
            if ratio > MAX_ASPECT_RATIO:
                continue
            selected.append(u)
        return selected

    # Try progressive thresholds (relaxed for furniture/e-commerce)
    big_candidates = filter_minimal(150)
    if not big_candidates:
        big_candidates = filter_minimal(100)
    if not big_candidates:
        # Last resort: accept any measured with reasonable constraints
        big_candidates = []
        for u in all_candidates:
            w, h = dims.get(u, (0, 0))
            if w <= 0 or h <= 0:
                continue
            shortest = min(w, h)
            ratio = (w / h) if h else 0
            ratio = ratio if ratio >= 1 else (1 / ratio if ratio > 0 else 0)
            if shortest >= MIN_SHORTEST_EDGE_ABSOLUTE and ratio <= MAX_ASPECT_RATIO:
                big_candidates.append(u)

    if not big_candidates:
        # Final fallback: content-type check
        block_hosts = ("cookielaw.org", "bat.bing.com", "/img/award", "branschvinnare", "favicon", ".svg")
        plausible = [u for u in all_candidates if isinstance(u, str) and u.startswith('http') and not any(b in u.lower() for b in block_hosts) and looks_like_image_url(u)]
        try:
            kept_by_ct = await _filter_urls_by_content_type(plausible[:40])
            if kept_by_ct:
                big_candidates = _select_largest_per_family(kept_by_ct, dims)
        except Exception:
            pass

    if not big_candidates:
        return [], dims

    # Dedupe by family and keep top candidates by size
    dedup = _select_largest_per_family(big_candidates, dims)
    dedup_sorted = sorted(dedup, key=lambda u: (max(dims.get(u, (0, 0))), dims.get(u, (0, 0))[0] * dims.get(u, (0, 0))[1]), reverse=True)
    return dedup_sorted, dims


async def apply_learned_selector(
    url: str, product_name: str, profile: DomainProfile, output_dir: str, api_key: Optional[str] = None
) -> DownloadResult:
    """Applies a pre-existing selector to a URL to extract and download images."""
    # Prefer robust browser-based load and probing
    html, candidates, dims, gallery_candidates, final_url = await _collect_and_probe_in_browser(url)
    if not html:
        # Check if it's a 404/error (empty HTML could mean HTTP error was detected)
        profile.notes = "Failed to load page (possibly 404 or HTTP error)"
        return DownloadResult(status='NOT_FOUND', ai_used=False, profile=profile, html=None)
    if _looks_like_404(html):
        profile.notes = 'Page appears to be 404/Not Found (detected via content analysis)'
        return DownloadResult(status='NOT_FOUND', ai_used=False, profile=profile, html=html)

    # Scope filtering to the learned container if available
    final_links, _ = await _get_filtered_candidates(
        html,
        url,
        candidates,
        dims,
        container_selector=profile.selector
    )

    # Fallback condition: if selector yields no good images, re-learn.
    if not final_links:
        return await learn_domain_selector(url, product_name, output_dir, api_key, is_fallback=True)

    saved_count = await _download_images(final_links, output_dir)
    
    return DownloadResult(
        status='OK' if saved_count > 0 else 'NO_IMAGES_SAVED',
        ai_used=False,
        selector_used=profile.selector,
        candidates_found=len(final_links),
        candidates_kept=len(final_links),
        saved_count=saved_count,
        kept_links=final_links,
        profile=profile,
        html=html,
        final_url=final_url
    )


async def learn_domain_selector(url: str, product_name: str, output_dir: str, api_key: Optional[str] = None, is_fallback: bool = False) -> DownloadResult:
    """Simplified AI-based image extraction using Playwright + direct URL analysis."""
    # Validate URL format before processing
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            profile = DomainProfile(domain="invalid")
            profile.notes = f"Invalid URL format: {url}"
            return DownloadResult(status='INVALID_URL', ai_used=False, profile=profile, html=None)
        if parsed.scheme not in ['http', 'https']:
            profile = DomainProfile(domain="invalid")
            profile.notes = f"Unsupported URL scheme: {url}"
            return DownloadResult(status='INVALID_URL', ai_used=False, profile=profile, html=None)
    except Exception as e:
        profile = DomainProfile(domain="invalid")
        profile.notes = f"URL parsing error: {e}"
        return DownloadResult(status='INVALID_URL', ai_used=False, profile=profile, html=None)
    
    domain = urlparse(url).netloc.lower()
    profile = DomainProfile(domain=domain)

    # Get HTML and candidates using Playwright only
    html, candidates, dims, gallery_candidates, final_url = await _collect_and_probe_in_browser(url)
    if not html:
        # Check if it's a 404/error (empty HTML could mean HTTP error was detected)
        profile.notes = "Failed to load page (possibly 404 or HTTP error)"
        return DownloadResult(status='NOT_FOUND', ai_used=True, profile=profile, html=None, final_url=final_url)
    
    if not candidates:
        profile.notes = "Failed to find any image candidates."
        return DownloadResult(status='FETCH_FAILED', ai_used=True, profile=profile, html=html, final_url=final_url)

    soup = BeautifulSoup(html, 'lxml')
    if _looks_like_404(html):
        profile.notes = 'Page appears to be 404/Not Found (detected via content analysis)'
        return DownloadResult(status='NOT_FOUND', ai_used=False, profile=profile, html=html, final_url=final_url)
    
    page_title = (soup.title.string if soup.title else '') or ''
    h1 = (soup.find('h1').get_text(strip=True) if soup.find('h1') else '')

    # Enhanced size-based filtering to eliminate small and non-product images
    # Apply dimension parsing to all candidates first
    for u in candidates:
        w, h = dims.get(u, (0, 0))
        if w == 0 or h == 0:
            parsed_dims = _parse_dims_from_url(u)
            if parsed_dims:
                dims[u] = parsed_dims

    def enhanced_size_filter(candidates_list: List[str], dims_dict: Dict[str, Tuple[int, int]]) -> List[str]:
        """Filter candidates by enhanced size heuristics - exclude thumbnails and similar products, focus on main product images."""
        filtered = []
        for url in candidates_list:
            w, h = dims_dict.get(url, (0, 0))
            
            # Skip if no dimensions available
            if w <= 0 or h <= 0:
                continue
                
            # Calculate metrics
            area = w * h
            longest_edge = max(w, h)
            shortest_edge = min(w, h)
            aspect_ratio = longest_edge / shortest_edge if shortest_edge > 0 else 0
            
            # Enhanced filtering criteria to exclude thumbnails and similar products
            # 1. Minimum area - increased to exclude small thumbnails
            if area < 200_000:  # 200k pixels minimum
                continue
                
            # 2. Minimum longest edge - increased to exclude thumbnails
            if longest_edge < 400:  # At least 400px on longest side
                continue
                
            # 3. Minimum shortest edge - increased to exclude thumbnails
            if shortest_edge < 300:  # At least 300px on shortest side
                continue
                
            # 4. Reasonable aspect ratio
            if aspect_ratio > 6.0:  # Max 6:1 ratio
                continue
                
            # 5. Skip thumbnail-sized images (common thumbnail sizes)
            if longest_edge < 500 and area < 500_000:  # Exclude small images
                continue
                
            # 6. Skip images with thumbnail indicators in URL
            # if any(thumb_indicator in url.lower() for thumb_indicator in ['w=150', 'w=200', 'w=300', 'thumb', 'thumbnail', 'small', 'mini']):
            #     continue
            
            # 7. Skip similar/related product images (different product IDs)
            url_lower = url.lower()
            
            # Skip images that are clearly from different products
            if any(exclude_pattern in url_lower for exclude_pattern in [
                'similar', 'related', 'recommended', 'also-like', 'cross-sell',
                'upsell', 'bundle', 'set', 'collection', 'series'
            ]):
                continue
            
            # 8. Additional filtering will be handled by AI analysis
            # The AI will determine which images belong to the specific product
            
            filtered.append(url)
        return filtered

    # Apply enhanced filtering - NOW ENABLED WITH PROPER DIMENSIONS
    size_filtered = enhanced_size_filter(candidates, dims)

    # Trusted gallery mode: if we detect enough gallery images, use them directly
    trusted_gallery: List[str] = []
    if gallery_candidates:
        # Fill missing dims for gallery and apply same size filter
        for u in gallery_candidates:
            w, h = dims.get(u, (0, 0))
            if w == 0 or h == 0:
                pd = _parse_dims_from_url(u)
                if pd:
                    dims[u] = pd
        gallery_filtered = enhanced_size_filter(gallery_candidates, dims)
        # Dedupe by family; cap to 12
        gallery_dedup = _select_largest_per_family(gallery_filtered, dims)
        # Keep reasonable order and limit
        trusted_gallery = gallery_dedup[:12]
        if len(trusted_gallery) >= 2:
            kept_links = list(trusted_gallery)
            # Optional perceptual dedupe on gallery
            try:
                import io
                from PIL import Image
                import hashlib
                async with aiohttp.ClientSession(headers={ 'User-Agent': 'Mozilla/5.0' }) as session:
                    async def fetch_thumb(u: str):
                        try:
                            async with session.get(u, timeout=10) as r:
                                if r.status == 200:
                                    data = await r.read()
                                    im = Image.open(io.BytesIO(data)).convert('L').resize((32, 32))
                                    pixels = list(im.getdata())
                                    avg = sum(pixels) / len(pixels)
                                    bits = ''.join('1' if p > avg else '0' for p in pixels)
                                    return (u, hashlib.md5(bits.encode('utf-8')).hexdigest())
                        except Exception:
                            return (u, None)
                        return (u, None)
                    results = await asyncio.gather(*[fetch_thumb(u) for u in kept_links])
                    seen = set(); uniq = []
                    for u, h in results:
                        if not h or h not in seen:
                            if h:
                                seen.add(h)
                            uniq.append(u)
                    kept_links = uniq
            except Exception:
                pass
            profile.selector = None
            profile.precision = 1.0 if kept_links else 0.0
            profile.notes = f"Trusted gallery: {len(kept_links)} images (from {len(gallery_candidates)} detected)"
            saved_count = await _download_images(kept_links, output_dir)
            return DownloadResult(
                status='OK' if saved_count > 0 else 'NO_IMAGES_SAVED',
                ai_used=False,
                selector_used=None,
                candidates_found=len(gallery_candidates),
                candidates_kept=len(kept_links),
                saved_count=saved_count,
                kept_links=kept_links,
                profile=profile,
                html=html,
                final_url=final_url
            )

    # If exactly one plausible candidate, bypass AI and keep it
    if len(size_filtered) == 1:
        kept_links = list(size_filtered)
        # Dimension-based dedupe is a no-op with 1 image, but keep the call for consistency
        kept_links = _select_largest_per_family(kept_links, dims)
        profile.selector = None
        profile.precision = 1.0
        profile.notes = "Single candidate, AI bypass"
        saved_count = await _download_images(kept_links, output_dir)
        return DownloadResult(
            status='OK' if saved_count > 0 else 'NO_IMAGES_SAVED',
            ai_used=False,
            selector_used=None,
            candidates_found=len(size_filtered),
            candidates_kept=len(kept_links),
            saved_count=saved_count,
            kept_links=kept_links,
            profile=profile,
            html=html,
            final_url=final_url
        )
    
    
    # Debug info for troubleshooting
    if not size_filtered and candidates:
        profile.notes = f"Found {len(candidates)} candidates but none passed size filtering. Sample dims: {dict(list(dims.items())[:3])}"
    
    # If no candidates pass size filtering, use all candidates (dimension detection might be failing)
    if not size_filtered and candidates:
        size_filtered = candidates
        # Add debug info when size filtering fails
        profile.notes = f"Size filtering failed, using all {len(candidates)} candidates"
    
    if not size_filtered:
        profile.notes = "No candidates passed size filtering"
        return DownloadResult(status='NO_CANDIDATES', ai_used=True, profile=profile, html=html, final_url=final_url)

    # Limit to top candidates for AI processing (increased to capture more product views)
    TOP_N = 25
    ai_candidates = size_filtered[:TOP_N]
    

    # Prepare AI input with actual images using Vision API
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        profile.notes = "GEMINI_API_KEY missing"
        return DownloadResult(status='ERROR', ai_used=True, profile=profile, final_url=final_url)

    # Download and prepare images for Vision API
    async def download_image_for_analysis(image_url: str) -> Optional[bytes]:
        """Download image data for Vision API analysis."""
        try:
            # Convert relative URLs to absolute URLs
            if image_url.startswith('/'):
                from urllib.parse import urlparse
                parsed_base = urlparse(url)
                image_url = f"{parsed_base.scheme}://{parsed_base.netloc}{image_url}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/jpeg,image/png,image/webp;q=0.9,image/*;q=0.1,*/*;q=0.1',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': url,
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'cross-site'
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(image_url, timeout=15) as response:
                    if response.status == 200:
                        ct = (response.headers.get('Content-Type') or '').lower()
                        if not ct.startswith('image/'):
                            return None
                        if 'image/avif' in ct or 'image/svg' in ct:
                            return None
                        image_data = await response.read()
                        # Check if image is not too large (Vision API has limits)
                        if len(image_data) < 20 * 1024 * 1024:  # 20MB limit
                            return image_data
                    else:
                        # Debug: Log failed downloads for carousel images
                        is_carousel = any(keyword in image_url.lower() for keyword in ['64580-0000092694', 'artiklar', 'zoom', 'bilder'])
                        if is_carousel:
                            print(f"Failed to download carousel image {image_url}: HTTP {response.status}")
        except Exception as e:
            # Debug: Log exceptions for carousel images
            is_carousel = any(keyword in image_url.lower() for keyword in ['64580-0000092694', 'artiklar', 'zoom', 'bilder'])
            if is_carousel:
                print(f"Exception downloading carousel image {image_url}: {e}")
        return None

    # Download images for analysis (limit to top candidates to avoid costs)
    MAX_IMAGES_FOR_AI = 25  # Increased to capture more carousel images
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB limit per image
    ai_candidates_limited = ai_candidates[:MAX_IMAGES_FOR_AI]
    
    downloaded_images = []
    for i, image_url in enumerate(ai_candidates_limited):
        image_data = await download_image_for_analysis(image_url)
        if image_data and len(image_data) <= MAX_IMAGE_SIZE:
            # Convert to base64 for Vision API
            import base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            downloaded_images.append({
                "url": image_url,
                "base64": base64_image,
                "index": i + 1
            })
        elif image_data and len(image_data) > MAX_IMAGE_SIZE:
            # Skip images that are too large
            continue
        else:
            # Debug: Log failed downloads
            is_carousel = any(keyword in image_url.lower() for keyword in ['64580-0000092694', 'artiklar', 'zoom', 'bilder'])
            if is_carousel:
                print(f"Failed to download carousel image: {image_url}")
    
    if not downloaded_images:
        # If we have very few candidates, be permissive and keep them
        if len(ai_candidates) <= 2:
            kept_links = list(ai_candidates)
            kept_links = _select_largest_per_family(kept_links, dims)
            profile.selector = None
            profile.precision = 1.0 if kept_links else 0.0
            profile.notes = "AI bypass (≤2 candidates, no images downloadable for analysis)"
            saved_count = await _download_images(kept_links, output_dir)
            return DownloadResult(
                status='OK' if saved_count > 0 else 'NO_IMAGES_SAVED',
                ai_used=False,
                selector_used=None,
                candidates_found=len(size_filtered),
                candidates_kept=len(kept_links),
                saved_count=saved_count,
                kept_links=kept_links,
                profile=profile,
                html=html
            )
        profile.notes = "No images could be downloaded for AI analysis"
        return DownloadResult(status='NO_IMAGES_DOWNLOADED', ai_used=True, profile=profile, html=html)
    
    # Debug: Log what images are being analyzed
    carousel_analyzed = [img['url'] for img in downloaded_images if any(keyword in img['url'].lower() for keyword in ['64580-0000092694', 'artiklar', 'zoom', 'bilder'])]
    print(f"DEBUG: Analyzing {len(downloaded_images)} images, {len(carousel_analyzed)} are carousel images")
    if carousel_analyzed:
        print(f"DEBUG: Carousel images being analyzed: {carousel_analyzed[:3]}")
    else:
        print("DEBUG: NO CAROUSEL IMAGES BEING ANALYZED!")
    
    # Prepare content blocks with actual images
    content_blocks = [
            {"type": "text", "text": (
                f"PRODUCT ANALYSIS REQUEST:\n"
                f"Product Name: {product_name}\n"
                f"Page URL: {url}\n"
                f"Page Title: {page_title}\n"
                f"Main Heading: {h1}\n\n"
                "TASK: Analyze the following images and select ONLY IMAGES OF THE SPECIFIC PRODUCT: '{product_name}'.\n\n"
                "SELECTION CRITERIA:\n"
                "✅ KEEP: Images of the EXACT product '{product_name}' only\n"
                "✅ KEEP: Different views of THIS SPECIFIC product (front, back, side, close-ups, details)\n"
                "✅ KEEP: THIS product in different settings (outdoor, indoor, lifestyle shots)\n"
                "✅ KEEP: Features and details of THIS SPECIFIC product\n"
                "✅ KEEP: High-quality images from the main product gallery/carousel\n"
                "✅ KEEP: Carousel images, gallery images, zoom images of THIS product\n\n"
                "❌ EXCLUDE: Images of DIFFERENT products, even if similar\n"
                "❌ EXCLUDE: Related products, recommended items, 'you might also like' sections\n"
                "❌ EXCLUDE: Cross-sell products, similar items, other products\n"
                "❌ EXCLUDE: Product variants that are NOT '{product_name}'\n"
                "❌ EXCLUDE: Logos, icons, banners, social media images\n"
                "❌ EXCLUDE: UI elements, navigation images, decorative images\n"
                "❌ EXCLUDE: Small images, logo-sized images, tiny thumbnails\n"
                "❌ EXCLUDE: Images that appear to be icons, badges, or UI elements\n\n"
                "PRODUCT IDENTIFICATION:\n"
                "Look carefully at each image to determine if it shows the SPECIFIC product '{product_name}':\n"
                "- If the image shows a DIFFERENT product (even if similar), EXCLUDE it\n"
                "- If the image shows the SAME product in different views/angles, INCLUDE it\n"
                "- If you're unsure whether it's the same product, EXCLUDE it (be conservative)\n"
                "- Focus on the main product gallery/carousel images\n\n"
                "SIZE ANALYSIS:\n"
                "Pay special attention to image size and resolution:\n"
                "- REJECT images that appear to be small logos, icons, or badges\n"
                "- REJECT images that look like thumbnails or preview images\n"
                "- REJECT images that appear to be UI elements or navigation graphics\n"
                "- REJECT images that are clearly too small to be main product photos\n"
                "- SELECT images that appear to be full-size product photos\n\n"
                "IMPORTANT: Be SELECTIVE and CONSERVATIVE. Only select images that clearly show the SPECIFIC product '{product_name}'. It's better to miss some images than to include images of different products."
            )}
        ]
    
    # Add each image to the content blocks
    for img_data in downloaded_images:
        # Determine image format from URL
        image_url = img_data['url'].lower()
        if image_url.endswith('.png'):
            mime_type = "image/png"
        elif image_url.endswith('.gif'):
            mime_type = "image/gif"
        elif image_url.endswith('.webp'):
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"  # Default to jpeg
        
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{img_data['base64']}",
                "detail": "low"  # Use low detail to reduce costs
            }
        })
        content_blocks.append({
            "type": "text", 
            "text": f"Image {img_data['index']}: {img_data['url']}"
        })

    prompt_tail = (
        "RESPONSE FORMAT:\n"
        "Respond with JSON exactly as:\n"
        "{\n  \"kept_urls\": [\"url1\", \"url2\", \"url3\"],\n  \"notes\": \"brief explanation of selection\"\n}\n\n"
        "SELECTION RULES:\n"
        "1. Return the actual URLs of the selected images (not indices)\n"
        "2. Select ALL main product images including different views and angles\n"
        "3. Analyze the actual image content and size to make decisions\n"
        "4. Include product photos, gallery images, showcase images, and detail shots\n"
        "5. Include different views: front, back, side, top, close-ups, interior, exterior\n"
        "6. Include lifestyle shots showing the product in use or different settings\n"
        "7. Exclude: related products, recommendations, logos, icons, banners, thumbnails\n"
        "8. Exclude: small images, logo-sized images, tiny thumbnails, UI elements\n"
        "9. Focus on images showing the specific product: " + product_name + "\n"
        "10. If unsure, err on the side of including more images rather than fewer\n"
        "11. Only return URLs that are in the provided list above\n"
        "12. IMPORTANT: Select multiple different views of the same product if available\n"
        "13. TARGET: Select 3-8 different product images showing various angles, details, and settings\n"
        "14. Look for: main product shot, close-ups, different angles, lifestyle shots, detail views\n"
        "15. CRITICAL: Reject any image that appears to be a small logo, icon, or thumbnail\n"
        "16. CRITICAL: Only select images that appear to be full-size, high-quality product photos"
    )
    content_blocks.append({"type": "text", "text": prompt_tail})

    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Create the model (using flash for better free tier limits)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare the prompt for Gemini
        system_prompt = "You are an expert at identifying main product images from e-commerce pages. Analyze the actual images to select only the primary product photos. Return valid JSON only."
        
        # Combine system prompt with user content
        full_prompt = f"{system_prompt}\n\n"
        
        # Add the main text content
        for block in content_blocks:
            if block["type"] == "text":
                full_prompt += block["text"] + "\n\n"
        
        # Prepare images for Gemini
        images = []
        for block in content_blocks:
            if block["type"] == "image_url":
                # Extract base64 data from data URL
                image_data_url = block["image_url"]["url"]
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
        
        # Generate content with Gemini
        if images:
            # Use vision model with images
            response = model.generate_content([full_prompt] + images)
        else:
            # Use text-only model
            response = model.generate_content(full_prompt)
        
        # Check if response has text content
        if hasattr(response, 'text') and response.text:
            content = response.text.strip()
        else:
            # Handle case where response doesn't have text
            profile.notes = f"Gemini API returned no text content: {response}"
            return DownloadResult(status='GEMINI_NO_TEXT', ai_used=True, profile=profile, html=html)
        if content.startswith('```json'): 
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        parsed = json.loads(content)
    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg or "Bad Request" in error_msg:
            profile.notes = f"Gemini API error (400): {error_msg}"
            return DownloadResult(status='GEMINI_API_ERROR', ai_used=True, profile=profile, html=html)
        elif "413" in error_msg or "too large" in error_msg:
            profile.notes = f"Request too large: {error_msg}"
            return DownloadResult(status='REQUEST_TOO_LARGE', ai_used=True, profile=profile, html=html)
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            profile.notes = f"Gemini API quota exceeded: {error_msg}"
            return DownloadResult(status='GEMINI_QUOTA_EXCEEDED', ai_used=True, profile=profile, html=html)
        else:
            profile.notes = f"Gemini API call or JSON parse failed: {error_msg}"
        return DownloadResult(status='ERROR', ai_used=True, profile=profile, html=html)

    # Extract kept images by URL and deduplicate by base image
    kept_links: List[str] = []
    kept_urls = parsed.get('kept_urls', [])
    if isinstance(kept_urls, list):
        # Log AI raw kept URLs for debugging (truncate output)
        try:
            logger.info(f"AI kept_urls (raw): {kept_urls[:5]}{'...' if len(kept_urls) > 5 else ''}")
        except Exception:
            pass

        # Normalize mapping: compare by normalized URL and by canonical basename
        # Create mapping from normalized form to original URL from kept_urls
        norm_to_original = { _normalize_for_match(u): u for u in kept_urls if isinstance(u, str) }
        base_to_original = { _canonical_basename(u): u for u in kept_urls if isinstance(u, str) }
        norm_kept_set = set(norm_to_original.keys())
        base_kept_set = set(base_to_original.keys())
        
        for cand in ai_candidates:
            try:
                cand_norm = _normalize_for_match(cand)
                cand_base = _canonical_basename(cand)
                
                # Find matching original URL from kept_urls
                original_url = None
                if cand_norm in norm_kept_set:
                    original_url = norm_to_original[cand_norm]
                elif cand_base in base_kept_set:
                    original_url = base_to_original[cand_base]
                
                if original_url:
                    # Use the ORIGINAL URL from kept_urls, not the candidate (which might be corrupted)
                    kept_links.append(original_url)
            except Exception:
                # Fall back to strict membership if normalization fails
                if cand in kept_urls:
                    kept_links.append(cand)

    # If AI yielded nothing but candidates are very few, keep them
    if not kept_links and len(ai_candidates) <= 2:
        logger.info("AI returned no matches; ≤2 candidates present, keeping them by fallback.")
        kept_links = list(ai_candidates)
    
    # Now perform AI-based image comparison to remove duplicates (run when 2+ images)
    if len(kept_links) >= 2:
        kept_links = await _deduplicate_images_by_ai_comparison(kept_links, api_key)
    
    # Debug: Log what AI selected and what was analyzed
    carousel_analyzed = [img['url'] for img in downloaded_images if any(keyword in img['url'].lower() for keyword in ['64580-0000092694', 'artiklar', 'zoom', 'bilder'])]
    carousel_selected = [url for url in kept_links if any(keyword in url.lower() for keyword in ['64580-0000092694', 'artiklar', 'zoom', 'bilder'])]
    profile.notes = f"AI analyzed {len(downloaded_images)} images ({len(carousel_analyzed)} carousel), selected {len(kept_links)} ({len(carousel_selected)} carousel)"
    
    # Dimension-based deduplication: keep largest per canonical family
    kept_links = _select_largest_per_family(kept_links, dims)

    # Optional perceptual-hash dedupe (cheap visual fallback)
    try:
        import io
        from PIL import Image
        import hashlib
        import base64
        phash_map: Dict[str, Tuple[str, int]] = {}
        unique_links: List[str] = []
        # Download small thumbnails quickly for hashing
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async def fetch_thumb(u: str):
                try:
                    async with session.get(u, timeout=10) as r:
                        if r.status == 200:
                            data = await r.read()
                            im = Image.open(io.BytesIO(data)).convert('L').resize((32, 32))
                            # aHash
                            pixels = list(im.getdata())
                            avg = sum(pixels) / len(pixels)
                            bits = ''.join('1' if p > avg else '0' for p in pixels)
                            digest = hashlib.md5(bits.encode('utf-8')).hexdigest()
                            return (u, digest)
                except Exception:
                    return (u, None)
                return (u, None)
            results = await asyncio.gather(*[fetch_thumb(u) for u in kept_links])
            seen_hashes = set()
            for u, h in results:
                if not h:
                    unique_links.append(u)
                    continue
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                unique_links.append(u)
        kept_links = unique_links
    except Exception:
        pass

    # Update profile
    profile.selector = None  # No selector needed for this approach
    profile.precision = 1.0 if kept_links else 0.0
    profile.notes = f"AI-selected product images {'(fallback)' if is_fallback else ''}"

    # Download final images to local directory (for now, will be replaced with direct upload)
    saved_count = await _download_images(kept_links, output_dir)

    return DownloadResult(
        status='OK' if saved_count > 0 else 'NO_IMAGES_SAVED',
        ai_used=True,
        selector_used=None,
        candidates_found=len(size_filtered),
        candidates_kept=len(kept_links),
        saved_count=saved_count,
        kept_links=kept_links,
        profile=profile,
        html=html,
        final_url=final_url
    )


