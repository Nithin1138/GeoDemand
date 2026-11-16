"""
GeoDemand AI — Product Categories & Pricing Configuration

Defines realistic street-vendor product categories, item catalogs,
average selling prices (ASP), product mix distributions, Average Order Values (AOV),
Cost of Goods Sold (COGS), perishable spoilage rates, and hourly capacity limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ProductItem:
    """Individual product item in a vendor category's catalog."""
    name: str
    selling_price_inr: float
    cogs_inr: float
    mix_share: float  # Fraction of orders containing this item (0.0 - 1.0)
    preparation_time_min: float = 2.0

    @property
    def gross_margin_inr(self) -> float:
        return round(self.selling_price_inr - self.cogs_inr, 2)

    @property
    def cogs_rate(self) -> float:
        if self.selling_price_inr <= 0:
            return 0.0
        return round(self.cogs_inr / self.selling_price_inr, 4)


@dataclass(frozen=True)
class CategoryPricingConfig:
    """Complete commercial pricing & operational configuration for a vendor category."""
    category_id: str
    category_name: str
    default_aov_inr: float
    default_cogs_rate: float
    spoilage_rate: float
    default_inventory_capacity: int  # Max customers / units per hour
    items: List[ProductItem] = field(default_factory=list)

    @property
    def weighted_asp_inr(self) -> float:
        """Weighted Average Selling Price per item based on product mix."""
        if not self.items:
            return self.default_aov_inr
        total_share = sum(item.mix_share for item in self.items)
        if total_share <= 0:
            return self.default_aov_inr
        return round(sum(item.selling_price_inr * item.mix_share for item in self.items) / total_share, 2)

    @property
    def weighted_cogs_inr(self) -> float:
        """Weighted Cost of Goods Sold per item based on product mix."""
        if not self.items:
            return round(self.default_aov_inr * self.default_cogs_rate, 2)
        total_share = sum(item.mix_share for item in self.items)
        if total_share <= 0:
            return round(self.default_aov_inr * self.default_cogs_rate, 2)
        return round(sum(item.cogs_inr * item.mix_share for item in self.items) / total_share, 2)

    @property
    def weighted_cogs_rate(self) -> float:
        """Effective COGS percentage from item catalog."""
        asp = self.weighted_asp_inr
        if asp <= 0:
            return self.default_cogs_rate
        return round(self.weighted_cogs_inr / asp, 4)

    def to_dict(self) -> dict:
        return {
            "category_id": self.category_id,
            "category_name": self.category_name,
            "default_aov_inr": self.default_aov_inr,
            "default_cogs_rate": self.default_cogs_rate,
            "spoilage_rate": self.spoilage_rate,
            "default_inventory_capacity": self.default_inventory_capacity,
            "weighted_asp_inr": self.weighted_asp_inr,
            "weighted_cogs_inr": self.weighted_cogs_inr,
            "weighted_cogs_rate": self.weighted_cogs_rate,
            "item_count": len(self.items),
        }


# ---------------------------------------------------------------------------
# Standard Product Catalogs by Category
# ---------------------------------------------------------------------------

PRODUCT_CATEGORIES: Dict[str, CategoryPricingConfig] = {
    "food": CategoryPricingConfig(
        category_id="food",
        category_name="Street Food & Tiffins",
        default_aov_inr=100.0,
        default_cogs_rate=0.36,
        spoilage_rate=0.04,  # 4% daily perishable waste (batter, prepared gravies)
        default_inventory_capacity=60,  # Max 60 orders/hr based on tawa/stove throughput
        items=[
            ProductItem(name="Masala Dosa", selling_price_inr=60.0, cogs_inr=21.0, mix_share=0.35, preparation_time_min=3.0),
            ProductItem(name="Idli / Vada Combo", selling_price_inr=40.0, cogs_inr=12.0, mix_share=0.25, preparation_time_min=1.0),
            ProductItem(name="Meals / Rice Bowl", selling_price_inr=120.0, cogs_inr=48.0, mix_share=0.20, preparation_time_min=2.0),
            ProductItem(name="Veg / Egg Noodles", selling_price_inr=90.0, cogs_inr=31.5, mix_share=0.20, preparation_time_min=4.0),
        ],
    ),
    "fruits": CategoryPricingConfig(
        category_id="fruits",
        category_name="Fresh Fruits & Juices",
        default_aov_inr=80.0,
        default_cogs_rate=0.40,
        spoilage_rate=0.08,  # 8% perishable shrink (bruising, overripe cuts)
        default_inventory_capacity=45,  # Max 45 orders/hr
        items=[
            ProductItem(name="Fresh Fruit Salad Bowl", selling_price_inr=60.0, cogs_inr=24.0, mix_share=0.40, preparation_time_min=2.0),
            ProductItem(name="Cut Papaya / Watermelon", selling_price_inr=40.0, cogs_inr=15.0, mix_share=0.30, preparation_time_min=1.5),
            ProductItem(name="Tender Coconut", selling_price_inr=50.0, cogs_inr=22.5, mix_share=0.20, preparation_time_min=1.0),
            ProductItem(name="Premium Mixed Platter", selling_price_inr=120.0, cogs_inr=50.0, mix_share=0.10, preparation_time_min=3.0),
        ],
    ),
    "beverages": CategoryPricingConfig(
        category_id="beverages",
        category_name="Tea, Coffee & Cold Beverages",
        default_aov_inr=40.0,
        default_cogs_rate=0.30,
        spoilage_rate=0.02,  # 2% milk/lemon spoilage
        default_inventory_capacity=80,  # High speed kettle/urn dispensing
        items=[
            ProductItem(name="Irani Masala Chai", selling_price_inr=15.0, cogs_inr=4.5, mix_share=0.45, preparation_time_min=0.5),
            ProductItem(name="South Filter Coffee", selling_price_inr=25.0, cogs_inr=7.5, mix_share=0.30, preparation_time_min=1.0),
            ProductItem(name="Fresh Lime Soda", selling_price_inr=30.0, cogs_inr=9.0, mix_share=0.15, preparation_time_min=1.5),
            ProductItem(name="Badam Milk / Lassi", selling_price_inr=50.0, cogs_inr=16.0, mix_share=0.10, preparation_time_min=1.0),
        ],
    ),
    "repair": CategoryPricingConfig(
        category_id="repair",
        category_name="Mobile & Electronics Quick Repair",
        default_aov_inr=180.0,
        default_cogs_rate=0.24,
        spoilage_rate=0.00,  # 0% spoilage on non-perishable spare components
        default_inventory_capacity=8,  # Precision bench labor constraint (8 jobs/hr)
        items=[
            ProductItem(name="Tempered Glass / Screen Guard", selling_price_inr=100.0, cogs_inr=22.0, mix_share=0.40, preparation_time_min=5.0),
            ProductItem(name="Puncture / Quick Wire Fix", selling_price_inr=60.0, cogs_inr=12.0, mix_share=0.30, preparation_time_min=7.0),
            ProductItem(name="Charging Port / Cable Replacement", selling_price_inr=250.0, cogs_inr=65.0, mix_share=0.20, preparation_time_min=15.0),
            ProductItem(name="Diagnostic & Hardware Service", selling_price_inr=350.0, cogs_inr=80.0, mix_share=0.10, preparation_time_min=20.0),
        ],
    ),
    "salon": CategoryPricingConfig(
        category_id="salon",
        category_name="Mobile Grooming & Salon Van",
        default_aov_inr=170.0,
        default_cogs_rate=0.18,
        spoilage_rate=0.01,  # 1% consumables (creams, blades, aftershaves)
        default_inventory_capacity=7,  # Chair time constraint (~10 min/service)
        items=[
            ProductItem(name="Haircut & Styling", selling_price_inr=150.0, cogs_inr=25.0, mix_share=0.45, preparation_time_min=12.0),
            ProductItem(name="Beard Trim & Groom", selling_price_inr=80.0, cogs_inr=12.0, mix_share=0.30, preparation_time_min=8.0),
            ProductItem(name="Hair Color & Treatment", selling_price_inr=250.0, cogs_inr=55.0, mix_share=0.15, preparation_time_min=20.0),
            ProductItem(name="Head Massage & Tonic", selling_price_inr=120.0, cogs_inr=15.0, mix_share=0.10, preparation_time_min=10.0),
        ],
    ),
    "snacks": CategoryPricingConfig(
        category_id="snacks",
        category_name="Chaat, Samosas & Hot Snacks",
        default_aov_inr=60.0,
        default_cogs_rate=0.34,
        spoilage_rate=0.05,  # 5% fried snack end-of-day waste
        default_inventory_capacity=50,
        items=[
            ProductItem(name="Mirchi Bajji / Pakoda", selling_price_inr=40.0, cogs_inr=14.0, mix_share=0.40, preparation_time_min=3.0),
            ProductItem(name="Samosa Plate (2 pcs)", selling_price_inr=30.0, cogs_inr=10.5, mix_share=0.30, preparation_time_min=1.0),
            ProductItem(name="Masala Sweet Corn", selling_price_inr=50.0, cogs_inr=16.0, mix_share=0.20, preparation_time_min=2.0),
            ProductItem(name="Special Chaat / Pani Puri", selling_price_inr=50.0, cogs_inr=16.0, mix_share=0.10, preparation_time_min=2.0),
        ],
    ),
}


def get_category_pricing(category: Optional[str]) -> CategoryPricingConfig:
    """Retrieve pricing & commercial configuration for a vendor category with fallback."""
    cat_key = (category or "food").lower().strip()
    return PRODUCT_CATEGORIES.get(cat_key, PRODUCT_CATEGORIES["food"])
