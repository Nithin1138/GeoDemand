# Fleet Management, Competition Sharing & Multi-Vendor Dispatch Report

**Author / Maintainer:** Pujitha & Team  
**Date:** November 16, 2026  
**Status:** Implemented, Tested, & Production-Ready  
**Test Suite:** `tests/test_fleet_manager.py` (11/11 tests passed in 4.99s; total suite: 109/109 tests passed)

---

## 1. Executive Summary

When multiple mobile vendors operate independently in an urban area like Vijayawada, uncoordinated decisions lead to **market saturation, excessive clustering, and severe revenue cannibalization**—e.g., three chaat carts crowding the exact same street corner, splitting footfall and eroding everyone's margins. Conversely, organized fleets (such as franchise food cart networks, cooperative fruit sellers, or municipal hawker syndicates) require global coordination to maximize collective revenue while preserving individual vendor livelihoods.

To solve this, we implemented the **Fleet Manager (`src/fleet/fleet_manager.py`)**:
1. **Dynamic Competition Cannibalization**: Models non-linear foot traffic dilution as vendors co-locate.
2. **Cell Capacity Enforcement**: Enforces total physical and category-specific vendor limits per H3 resolution 8 cell.
3. **Atomic H3 Reservations**: Implements thread-safe soft (provisional with TTL expiry) and hard (confirmed) reservations, eliminating race conditions.
4. **Fleet-Wide Profit Optimization & Anti-Clustering Dispatch**: Coordinated multi-cart dispatch that spreads vendors across high-demand commercial hubs rather than clustering them into the same cell.

---

## 2. Competition Sharing & Non-Linear Demand Cannibalization

As identical category vendors cluster into an H3 cell, customer demand is shared with diminishing returns according to the roadmap specification:

$$\boxed{D_{\text{effective}}(H) = \frac{D_{\text{base}}(H)}{1 + \lambda \cdot N_{\text{same category}}(H)}}$$

where:
- $D_{\text{base}}(H)$ is the expected customer count at H3 cell $H$ without co-located competition.
- $N_{\text{same category}}(H)$ is the number of already active or reserved vendors of the same product category in cell $H$.
- $\lambda$ is the category-specific competition cannibalization coefficient.

### 2.1 Category Cannibalization Factors ($\lambda$)

| Category | $\lambda$ | Behavioral Rationale |
|---|---|---|
| **Food / Meals** | `0.35` | Moderate substitution; consumers choose between meal options. |
| **Snacks / Chaat** | `0.40` | High impulse substitution; adjacent snack stalls aggressively divide footfall. |
| **Fruits & Veg** | `0.30` | Produce buyers browse adjacent carts for freshness and price comparison. |
| **Beverages / Tea** | `0.25` | High volume, fast turnaround; shared footfall with slight brand loyalty. |
| **Repair & Services** | `0.50` | Highly localized finite demand (e.g. tire puncture / mechanic needs); heavy cannibalization. |
| **Salon & Grooming** | `0.45` | Time-intensive appointment/walk-in services. |

### 2.2 Demand Sharing Penalty
The demand lost to co-located competition is explicitly quantified as:
$$\Delta D_{\text{cannibalized}}(H) = D_{\text{base}}(H) - D_{\text{effective}}(H) = D_{\text{base}}(H) \cdot \left(1 - \frac{1}{1 + \lambda N_{\text{same category}}(H)}\right)$$

---

## 3. Cell Capacity & Reservation Engine

Each H3 resolution 8 cell ($\approx 0.74\text{ km}^2$, edge $\approx 461\text{m}$) has bounded physical curb space and municipal hawking capacity.

### 3.1 Capacity Limits
- **Total Cell Capacity**: $C_{\text{total}}(H)$ (default: `6` vendors).
- **Category-Specific Capacity**: $C_{\text{cat}}(H, c)$ (default: `2` food, `2` fruits, `2` beverages, `2` snacks, `1` repair, `1` salon).

A new reservation is permitted if and only if:
$$\text{ActiveTotal}(H) < C_{\text{total}}(H) \quad \text{AND} \quad \text{ActiveCategory}(H, c) < C_{\text{cat}}(H, c)$$

### 3.2 Soft vs Hard Reservations & TTL Auto-Expiry
- **Soft Reservation (`is_soft=True`)**: Holds an H3 spot provisionally during transit or dispatch calculation. Carries a configurable Time-To-Live (default: `15` minutes). If the vendor fails to confirm or check in before `expires_at`, the reservation automatically transitions to `EXPIRED`, immediately freeing up cell capacity and resetting the vendor's status from `RESERVED` to `IDLE`.
- **Hard Reservation (`is_soft=False`)**: Confirmed reservation spanning the full operational time window (`slot_start` to `slot_end`).
- **Cancellation**: Vendor or operator can explicitly cancel via `POST /v1/fleet/cancel-reservation`, freeing capacity in real time.
- **Conflict Prevention**: A vendor is prohibited from holding multiple concurrent active reservations in different cells.

### 3.3 Concurrency Control & Atomic Persistence
- Thread-safe state synchronization via `threading.RLock()` across all booking, cancellation, and sweep operations.
- SQLite persistence (`data/geodemand_fleet.db`) with relational tables (`fleet_vendors`, `fleet_reservations`) ensuring durable state across process restarts.
- **Stress-Tested Race Condition Immunity**: 30 simultaneous worker threads attempting to book the last remaining spot in a capacity-1 cell resulted in exactly 1 confirmed reservation and 29 atomic rejections with zero state corruption.

---

## 4. Multi-Vendor Fleet Dispatch Engine

The fleet dispatch engine allocates a fleet of available vendors $\{v_1, \dots, v_m\}$ across commercial candidate zones $\{H_1, \dots, H_k\}$.

### 4.1 Optimization Objective
Maximizes fleet-wide total net profit after road travel, fuel consumption, and fixed hourly costs:

$$\max \sum_{v=1}^{m} \sum_{j=1}^{k} y_{v, j} \cdot \Big[ \text{Revenue}\big(D_{\text{effective}}(H_j, v), \text{AOV}_v\big) - \text{COGS}_v - \text{FuelCost}(v, H_j) - \text{FixedOpex}_v \Big]$$

### 4.2 Clustering Mitigation (Dispersion Ratio)
Because $D_{\text{effective}}$ diminishes with each additional vendor of the same category, the optimizer naturally prefers routing vendors to distinct commercial zones. To further safeguard against clustering, a dispersion incentive rewards allocating vendors to unpopulated cells when demand levels are comparable.
- **Cluster Dispersion Ratio**:
  $$\text{Dispersion Ratio} = \frac{\text{Unique Assigned H3 Cells}}{\text{Total Dispatched Vendors}}$$
  A ratio of $1.0$ indicates zero clustering (every vendor operates in their own exclusive zone).

### 4.3 Support for Independent & Organized Fleets
- **Independent Operators (`fleet_id=None`)**: Single cart dispatch and reservations.
- **Organized Fleets (`fleet_id="corp_fleet_alpha"`)**: Multi-cart coordinated fleet deployment.

---

## 5. API Endpoints

### 5.1 Reserve H3 Cell
```http
POST /v1/fleet/reserve-cell
X-API-Key: <api_key>
Content-Type: application/json

{
  "vendor_id": "VND-101",
  "h3_cell": "88619aa6c7fffff",
  "vendor_category": "food",
  "duration_minutes": 90,
  "is_soft": true,
  "ttl_minutes": 15
}
```

### 5.2 Cancel Reservation
```http
POST /v1/fleet/cancel-reservation
X-API-Key: <api_key>
Content-Type: application/json

{
  "reservation_id": "RES-E7F28B1A02",
  "vendor_id": "VND-101"
}
```

### 5.3 Query Cell Occupancy & Demand Multipliers
```http
GET /v1/fleet/cell-occupancy?h3_cell=88619aa6c7fffff
```
Returns total capacity, active occupancy, remaining slots, per-category breakdown, and current competition multipliers ($1/(1+\lambda N)$).

### 5.4 Multi-Vendor Dispatch
```http
POST /v1/fleet/dispatch
X-API-Key: <api_key>
Content-Type: application/json

{
  "fleet_id": "fleet_benz_circle",
  "vendors": [
    {"vendor_id": "V1", "vendor_category": "food", "latitude": 16.5062, "longitude": 80.6480},
    {"vendor_id": "V2", "vendor_category": "food", "latitude": 16.5070, "longitude": 80.6490},
    {"vendor_id": "V3", "vendor_category": "fruits", "latitude": 16.5050, "longitude": 80.6470}
  ],
  "search_radius_km": 4.0,
  "max_travel_km": 5.0,
  "auto_reserve_soft": true
}
```

---

## 6. Test Suite & Verification Results

The fleet manager is validated by 11 targeted unit and integration tests in `tests/test_fleet_manager.py`:
1. `test_vendor_model_and_availability`: Vendor identity, category normalization, H3 index computation, and status flags.
2. `test_effective_demand_and_competition_penalties`: Mathematical verification of $D_{\text{effective}}$ across food, snacks, fruits, and repair categories.
3. `test_cell_capacity_and_category_limits`: Enforces physical caps and rejects category overflow.
4. `test_soft_reservation_and_ttl_expiry`: Verifies soft reservation auto-expiration freeing cell capacity.
5. `test_reservation_cancellation`: Verifies manual cancellation immediately frees cell capacity.
6. `test_prevent_conflicting_reservations`: Rejects double-booking for the same vendor.
7. `test_atomic_concurrency_race_condition`: 30 concurrent threads competing for 1 slot (exactly 1 confirmed, 29 rejected).
8. `test_multi_vendor_fleet_dispatch_clustering_mitigation`: 4 food vendors distributed across distinct zones (dispersion ratio $\ge 0.75$).
9. `test_fleet_dispatch_independent_and_organized`: Independent single carts vs multi-cart fleets.
10. `test_fleet_reserve_and_occupancy_endpoints`: FastAPI `POST /v1/fleet/reserve-cell` and `GET /v1/fleet/cell-occupancy`.
11. `test_fleet_dispatch_endpoint_contract`: FastAPI `POST /v1/fleet/dispatch` schema and execution.

**Execution Results:**
- Fleet Manager suite: **11 passed in 4.99s**
- Full application test suite: **109 passed in 11.32s**
