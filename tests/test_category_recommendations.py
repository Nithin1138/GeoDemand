"""
Unit tests for Business Category Specialization & Differentiated Recommendations:
  - Domain-specific spatial affinity scoring
  - Category-calibrated customer throughput and capacity constraints
  - Category-aware multi-factor ranking
  - Customized SHAP-grounded natural language explanations
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "api"))

import pytest
from api.recommender import (
    compute_category_affinity,
    calculate_category_demand,
    rank_candidates,
    generate_explanation,
    CATEGORY_SPATIAL_WEIGHTS,
    CATEGORY_THROUGHPUT_SPECS,
)


def test_category_spatial_affinity_differentiation():
    """Verify that different business types have distinct affinities for different urban features."""
    # Healthcare cluster: 50 hospitals, 6 parks, low industrial
    medical_cluster = {
        "hospital_count": 50.0,
        "park_count": 6.0,
        "residential_ratio": 0.4,
        "office_count": 5.0,
        "mall_count": 2.0,
        "industrial_ratio": 0.0,
        "road_density": 0.4,
    }

    # Industrial/Transport corridor: high industrial, high road density, low hospitals
    industrial_corridor = {
        "hospital_count": 1.0,
        "park_count": 0.0,
        "residential_ratio": 0.05,
        "industrial_ratio": 0.35,
        "road_density": 0.90,
        "bus_stop_count": 4.0,
        "commercial_ratio": 0.15,
    }

    # Commercial lifestyle center: malls, offices, restaurants, posher residential
    lifestyle_hub = {
        "mall_count": 25.0,
        "residential_ratio": 0.35,
        "building_density": 2.5,
        "parking_count": 5.0,
        "office_count": 30.0,
        "restaurant_count": 40.0,
        "commercial_ratio": 0.45,
        "hospital_count": 3.0,
        "industrial_ratio": 0.0,
        "road_density": 0.50,
    }

    fruits_med_aff = compute_category_affinity("fruits", medical_cluster)
    repair_med_aff = compute_category_affinity("repair", medical_cluster)
    assert fruits_med_aff > repair_med_aff, "Fruits must have higher affinity for medical cluster than auto repair"

    repair_ind_aff = compute_category_affinity("repair", industrial_corridor)
    fruits_ind_aff = compute_category_affinity("fruits", industrial_corridor)
    assert repair_ind_aff > fruits_ind_aff, "Auto repair must have higher affinity for industrial corridor than fruits"

    salon_life_aff = compute_category_affinity("salon", lifestyle_hub)
    salon_ind_aff = compute_category_affinity("salon", industrial_corridor)
    assert salon_life_aff > salon_ind_aff, "Salon van must have higher affinity for lifestyle center than industrial"


def test_category_demand_capacity_limits():
    """Verify that expected customer throughput is realistically bounded by service capabilities."""
    raw_p50 = 45
    features = {
        "office_count": 15.0,
        "mall_count": 10.0,
        "hospital_count": 15.0,
        "road_density": 0.6,
        "commercial_ratio": 0.3,
        "residential_ratio": 0.3,
    }

    food_cust, _ = calculate_category_demand(raw_p50, "food", features)
    fruits_cust, _ = calculate_category_demand(raw_p50, "fruits", features)
    salon_cust, _ = calculate_category_demand(raw_p50, "salon", features)
    repair_cust, _ = calculate_category_demand(raw_p50, "repair", features)

    # Food vendors serve fast quick-service crowds (15 to 60 cust/hr)
    assert 15 <= food_cust <= 60
    # Fruits vendors serve 10 to 35 cust/hr
    assert 10 <= fruits_cust <= 35
    # Salon vans take 15-25 min per client, serving max 3 to 7 clients/hr
    assert 3 <= salon_cust <= 7
    # Auto repair takes 20-35 min per vehicle, serving 2 to 6 vehicles/hr
    assert 2 <= repair_cust <= 6

    # Verify relative ordering matches operational realities
    assert food_cust > fruits_cust > salon_cust
    assert salon_cust >= repair_cust


def test_category_aware_candidate_ranking():
    """Verify that category affinity influences the recommendation score when affinities are present."""
    cand_high_aff = {
        "h3_cell": "cell_custom",
        "expected_customers": 25,
        "relocation_adjusted_realized_profit_inr": 1500.0,
        "distance_km": 1.5,
        "competition_score": 0.2,
        "category_affinity": 250.0,  # strong category suitability
    }
    cand_low_aff = {
        "h3_cell": "cell_generic",
        "expected_customers": 25,
        "relocation_adjusted_realized_profit_inr": 1500.0,
        "distance_km": 1.5,
        "competition_score": 0.2,
        "category_affinity": 10.0,  # weak category suitability
    }

    ranked = rank_candidates([cand_high_aff, cand_low_aff], vendor_category="fruits")
    assert ranked[0]["h3_cell"] == "cell_custom"
    assert ranked[0]["recommendation_score"] > ranked[1]["recommendation_score"]


def test_category_tailored_explanation():
    """Verify that generate_explanation produces category-specific rationale."""
    cand_fruit = {
        "expected_customers": 32,
        "expected_profit_inr": 1850.0,
        "distance_km": 1.2,
        "_features": {"hospital_count": 25.0, "park_count": 4.0},
    }
    _, narrative_fruit = generate_explanation(cand_fruit, "fruits")
    assert "hospital" in narrative_fruit.lower() or "fruit" in narrative_fruit.lower()

    cand_salon = {
        "expected_customers": 6,
        "expected_profit_inr": 420.0,
        "distance_km": 0.8,
        "_features": {"mall_count": 20.0, "parking_count": 3.0},
    }
    _, narrative_salon = generate_explanation(cand_salon, "salon")
    assert "mall" in narrative_salon.lower() or "grooming" in narrative_salon.lower() or "retail" in narrative_salon.lower()
