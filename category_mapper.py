"""
Category validation and mapping

This module validates categories and subcategories against the approved taxonomy.
The taxonomy consists of 15 main categories, each with specific subcategories.
"""

from typing import Dict, Optional, Tuple, List

# Approved category taxonomy with subcategories
CATEGORIES_WITH_SUBCATEGORIES: Dict[str, List[str]] = {
    "Eyewear": [
        "Sunglasses", "Goggles", "Reading Glasses", "Safety Glasses",
        "Frame", "Lens", "Sports"
    ],
    "Vehicles": [
        "Cars", "Motorcycles", "Buses/Vans", "Trucks", "Scooters",
        "Bikes (Electric/Standard)",
        "Go-Karts", "ATVs", "Boats",
        "Industrial/Heavy", "Military", "Misc Vehicles"
    ],
    "Electronics": [
        "Computers (Desktop/Laptop)", "Monitors", "Peripherals", "Speakers",
        "Televisions", "Audio Equipment", "Phones", "Tablets",
        "Wearables", "Routers", "IoT Devices", "Misc Electronics"
    ],      
    "Home Appliances": [
        "Ovens", "Stoves", "Microwaves", "Fridges", "Freezers",
        "Washers", "Vacuum Cleaners", "Coffee Makers", "Blenders",
        "AC Units", "Fans", "Misc Home Appliances"
    ],
    "Outdoor & Garden": [
        "Sheds", "Greenhouses", "Gazebos/Pergolas", "Lawnmowers",
        "Grills/BBQ", "Tents", "Outdoor Seating",
        "Outdoor Lighting",
        "Firepits", "Fire Decor", "Misc Outdoor & Garden"   
    ],
    "Spa & Wellness": [
        "Hot Tubs", "Spas", "Pools", "Saunas",
        "Massage Chairs", "Pool Equipment", "Misc Spa & Wellness"           
    ],
    "Architectural Elements": [
        "Doors (Exterior/Interior)", "Windows (Balcony/Standard)", "Fences",
        "Walls", "Columns", "Fixtures", "Fireplaces",
        "Houses", "Buildings", "Misc Architectural Elements"
    ],
    "Home Decor": [
        "Lamps", "Ceiling Lights", "Candles", "Carpets/Rugs",
        "Ornaments", "Curtains", "Artificial Trees", "Pillow Covers", "Misc Home Decor"
    ],
    "Furniture": [
        "Chairs", "Sofas", "Cabinets", "Bookshelves",
        "Tables/Desks", "Beds", "Furniture Sets", "Fire Tables", "Misc Furniture"
    ],
    "Hardware & Components": [
        "Screws/Bolts", "Nails", "Hinges", "Brackets",
        "Knobs", "Handles", "Drawer Inserts",
        "Misc Hardware & Components"
    ],
    "Nursery & Baby Gear": [
        "Cribs/Cots", "Strollers", "Car Seats", "Toys",
        "Bibs", "Bouncers", "Baby Monitors", "Baby Gear", "Misc Nursery & Baby Gear"
    ],
    "Plants": [
        "Trees", "Flowers", "Bushes/Shrubs", "Grass",
        "Potted Plants", "Plant Pots/Planters",
        "Landscaping", "Conifers", "Misc Plants"
    ],
    "Fitness & Sport": [
        "Treadmills", "Exercise Bikes", "Weights", "Benches",
        "Trampolines", "Mats", "Sports Equipment",
        "Watercraft (Kayaks/Canoes)", "Surfboards",
        "Bikes (Sports/MTB)", "Misc Fitness & Sport"
    ],
    "Houseware": [
        "Plates", "Cups/Mugs", "Pans/Pots", "Cutlery",
        "Serving Tools", "Bathroom Accessories", "Storage Bins", "Misc Houseware"
    ],
    "Others": ["Suitcases", "Helmets", "Specialty Tools", "Uncategorized"]
}

# Flat list of valid categories (for backward compatibility)
VALID_CATEGORIES = list(CATEGORIES_WITH_SUBCATEGORIES.keys())

# Get valid subcategories for a given category
def get_subcategories_for_category(category: str) -> List[str]:
    """Get list of valid subcategories for a given category"""
    return CATEGORIES_WITH_SUBCATEGORIES.get(category, [])

# Category examples for better classification (optional - can be expanded)
CATEGORY_EXAMPLES: Dict[str, list] = {
    "Eyewear": [
        "glasses", "sunglasses", "reading glasses",
        "prescription glasses", "goggles"
    ],
    "Vehicles": [
        "car", "e-bike", "electric bike", "bike", "bicycle",
        "motorcycle", "scooter", "e-scooter", "boat", "moped"
    ],
    "Electronics": [
        "phone", "smartphone", "laptop", "tablet", "headphones",
        "speaker", "camera", "digital clock", "router",
        "computer", "desktop", "gaming computer"
    ],
    "Home Appliances": [
        "refrigerator", "fridge", "washing machine", "dishwasher",
        "oven", "microwave", "kettle", "coffee machine",
        "dough mixer", "tumble dryer", "stove"
    ],
    "Outdoor & Garden": [
        "garden furniture", "bbq", "barbecue", "outdoor lighting",
        "electric grill", "pergola", "carport", "garden shed",
        "gazebo", "greenhouse", "garden tools"
    ],
    "Spa & Wellness": [
        "massage chair", "sauna", "hot tub", "spa",
        "pool", "swimming pool", "jacuzzi"
    ],
    "Architectural Elements": [
        "door", "window", "fence", "gate", "railing",
        "wall panel", "wall cladding", "house", "garage",
        "annex", "outdoor office", "houses", "buildings"
    ],
    "Home Decor": [
        "mirror", "picture frame", "vase", "candle holder",
        "rug", "carpet", "curtain", "lamp", "wall clock", "decor clock"
    ],
    "Furniture": [
        "sofa", "armchair", "chair", "stool", "table",
        "bed", "cabinet", "wardrobe", "shelf", "shelving unit",
        "desk", "conference table", "dining table",
        "coffee table", "sofa bed"
    ],
    "Hardware & Components": [
        "screws", "bolts", "nails", "hinges", "handles",
        "brackets", "knob", "hinge", "bracket", "handle"
    ],
    "Nursery & Baby Gear": [
        "crib", "cot", "stroller", "push chair",
        "high chair", "baby monitor", "baby carrier",
        "bouncer", "potty"
    ],
    "Plants": [
        "indoor plants", "outdoor plants", "potted plant",
        "potted tree", "seeds", "plant pots", "planters"
    ],
    "Fitness & Sport": [
        "dumbbells", "weights", "treadmill", "exercise bike",
        "stationary bike", "gym equipment", "fitness equipment",
        "yoga mat", "sports equipment", "surfboard",
        "kayak", "canoe", "mountain bike", "dirt bike"
    ],
    "Houseware": [
        "kitchen utensils", "kitchen tools", "storage containers",
        "food containers", "cleaning supplies", "dinnerware",
        "plates", "cups", "mugs", "cutlery set"
    ],
    "Others": [
        # Catch-all for items that don't fit other categories
    ],
}



def map_product_type_to_category(product_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Map product type to category and subcategory (fuzzy matching fallback)
    
    Note: This function is kept for backward compatibility, but classification
    should happen directly in the Gemini prompt using CATEGORIES_WITH_SUBCATEGORIES.
    
    Args:
        product_type: Product type detected from text/image analysis
        
    Returns:
        Tuple of (category, subcategory) or (None, None) if no match found
    """
    if not product_type:
        return None, None
    
    # Normalize product type
    product_type_lower = product_type.lower().strip()
    
    # Try to match against category examples (fuzzy matching)
    for category, examples in CATEGORY_EXAMPLES.items():
        if any(example.lower() in product_type_lower or product_type_lower in example.lower() 
               for example in examples if examples):
            # Return category, but no subcategory (subcategory should come from Gemini)
            return category, None
    
    # If no match found, return None (classification should happen in Gemini prompt)
    return None, None


def validate_category(category: Optional[str]) -> bool:
    """
    Validate that category is in the approved taxonomy
    
    Args:
        category: Category to validate
        
    Returns:
        True if category is valid, False otherwise
    """
    if not category:
        return False
    
    category_normalized = category.strip()
    # Case-insensitive matching
    return any(cat.lower() == category_normalized.lower() for cat in VALID_CATEGORIES)


def validate_subcategory(subcategory: Optional[str], category: Optional[str]) -> bool:
    """
    Validate subcategory against the approved taxonomy
    
    Args:
        subcategory: Subcategory to validate
        category: Category to validate against
        
    Returns:
        True if subcategory is valid for the given category, False otherwise
    """
    if not category:
        return False
    
    if not subcategory:
        # Subcategory is optional - None/empty is valid
        return True
    
    # Get valid subcategories for this category
    valid_subcategories = get_subcategories_for_category(category)
    
    if not valid_subcategories:
        # Category has no subcategories defined, so any subcategory is invalid
        return False
    
    # Case-insensitive matching
    subcategory_normalized = subcategory.strip()
    return any(sub.lower() == subcategory_normalized.lower() for sub in valid_subcategories)

