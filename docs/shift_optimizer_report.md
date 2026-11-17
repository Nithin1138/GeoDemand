# Multi-Stop Daily Shift Optimization & VRPTW Report

**Author / Maintainer:** Pujitha & Team  
**Date:** November 16, 2026  
**Status:** Implemented, Tested, & Production-Ready  
**Test Suite:** `tests/test_shift_optimization.py` (9/9 tests passed in 6.7s; total suite: 98/98 tests passed)

---

## 1. Executive Summary

Mobile street vendors and hawkers rarely operate at a single static location for an entire 8 to 12-hour working day. Foot traffic, meal times, office hours, and transit commute waves cause customer demand to peak at different commercial hubs across Vijayawada at distinct times of day. However, frequent relocations incur heavy physical setup/teardown friction, road transit delays, and fuel expenses.

To solve this, we formulated the vendor's day as a **Vehicle Routing Problem with Time Windows and Profit Orienteering (VRPTW-OP)**, powered by **Google OR-Tools CP-SAT**. The optimizer computes the mathematically optimal sequence of commercial zones, arrival times, dwell durations, and departure times to maximize total shift net profit while strictly respecting physical time budgets and operational constraints.

---

## 2. Mathematical Formulation

Let the vendor shift be defined by:
- **Shift Duration**: $T_{\text{shift}} \in [4, 14]$ hours (typically 8–12 hours), starting at $t_0$.
- **Candidate Pool**: Set of candidate H3 locations $\mathcal{C} = \{0, 1, \dots, N-1\}$, where index $0$ represents the vendor's current starting origin.
- **Relocation Limit**: Maximum relocation stops $K_{\max} \in [1, 5]$.
- **Overhead Durations**:
  - Setup time $S = 15\text{ min}$ (applied at all relocation stops $s > 0$).
  - Teardown time $E = 15\text{ min}$ (applied at every active stop prior to departure).
  - Parking/relocation overhead $O = 5\text{ min}$.
  - Minimum dwell time $D_{\min} \in [30, 240]\text{ min}$ (prevents futile cart-hopping).

### 2.1 Decision Variables
For each ordered stop step $s \in \{0, \dots, K_{\max}-1\}$:
1. **Visit Indicator**: $v_s \in \{0, 1\}$, where $v_s = 1$ if step $s$ is visited.
2. **Location Selection**: $x_{s, i} \in \{0, 1\}$ for $i \in \mathcal{C}$, indicating whether location $i$ is chosen for step $s$.
3. **Arrival Time**: $a_s \in [0, T_{\text{shift\_min}}]$ (minutes from $t_0$).
4. **Trading Dwell Duration**: $d_s \in [D_{\min}, T_{\text{shift\_min}}]$ (minutes of active customer serving).
5. **Departure Time**: $e_s \in [0, T_{\text{shift\_min}}]$.

### 2.2 Objective Function
The solver maximizes expected net trading profit minus transit fuel costs over the entire shift:

$$\max \sum_{s=0}^{K_{\max}-1} \sum_{i \in \mathcal{C}} x_{s, i} \cdot P_{\text{stop}}(i, d_s) - \sum_{s=0}^{K_{\max}-2} \sum_{i \in \mathcal{C}} \sum_{j \in \mathcal{C}} x_{s, i} \cdot x_{s+1, j} \cdot \text{FuelCost}(i, j)$$

where:
- $P_{\text{stop}}(i, d_s) = \text{Revenue}(i, d_s) - \text{COGS}(i, d_s) - \text{HourlyFixedOpex} \cdot \frac{d_s}{60}$
- $\text{FuelCost}(i, j) = \frac{\text{DistanceKm}(i, j)}{\text{FuelEfficiency}} \times \text{FuelPricePerLitre}$

To guarantee ultra-fast integer linear programming convergence in OR-Tools CP-SAT ($< 100\text{ms}$ solve time), profit values and costs are scaled to integer INR per minute.

### 2.3 Operational Constraints
1. **Contiguous Sequencing**:
   $$v_s \ge v_{s+1} \quad \forall s \in \{0, \dots, K_{\max}-2\}$$
2. **Exactly One Location per Active Stop**:
   $$\sum_{i \in \mathcal{C}} x_{s, i} = v_s \quad \forall s$$
3. **No-Repeat Constraint** (a vendor does not loop back to the same location):
   $$\sum_{s=0}^{K_{\max}-1} x_{s, i} \le 1 \quad \forall i \in \mathcal{C}$$
4. **Origin Starting Anchor**: Stop 0 starts at origin location $i=0$ ($x_{0, 0} = 1$).
5. **Timeline Propagation & Overhead Chain**:
   For each stop $s$:
   - Trading start: $t_{\text{start}, s} = a_s + (S \text{ if } s > 0 \text{ else } 0)$
   - Trading end: $t_{\text{end}, s} = t_{\text{start}, s} + d_s$
   - Departure: $e_s = t_{\text{end}, s} + E$
   For transition from stop $s$ to $s+1$:
   $$a_{s+1} \ge e_s + \text{TransitTime}(x_s, x_{s+1}) + O$$
6. **Commercial Time Window Compatibility**:
   If candidate $i$ has an active peak demand window $[W_{\text{start}}, W_{\text{end}}]$:
   $$t_{\text{start}, s} \ge W_{\text{start}} - M(1 - x_{s, i})$$
   $$t_{\text{end}, s} \le W_{\text{end}} + M(1 - x_{s, i})$$
7. **Strict Shift-End Constraint**:
   $$e_s \le T_{\text{shift\_min}} \quad \forall s$$

---

## 3. Dynamic Time Window Engine (`src/optimization/time_windows.py`)

Street vendor demand fluctuates according to meal cycles, office breaks, and transit rush hours. `time_windows.py` establishes category baselines and POI contextual modulations:

| Vendor Category | Morning Window | Afternoon Window | Evening Window | Night Window |
|---|---|---|---|---|
| **Food / Meals** | 07:00 – 10:30 (Breakfast, 1.25x) | 12:00 – 15:30 (Lunch, 1.50x) | 18:30 – 22:30 (Dinner, 1.40x) | — |
| **Fruits & Veg** | 06:30 – 11:30 (Morning Mkt, 1.40x)| — | 16:30 – 20:30 (Evening Mkt, 1.30x)| — |
| **Beverages / Tea**| 07:00 – 11:00 (Morning Rush, 1.35x)| 13:00 – 17:00 (Afternoon Tea, 1.40x)| 17:00 – 21:00 (Social Hour, 1.25x) | — |
| **Snacks & Chaat** | — | — | 16:00 – 21:30 (Snack Rush, 1.50x) | — |
| **Repair & Services**| 09:30 – 13:30 (Morning, 1.20x) | 14:00 – 18:30 (Afternoon, 1.20x) | — | — |

### POI Temporal Modulations:
- **Colleges & Universities**: Boosts afternoon snacks/beverages (+30% multiplier).
- **Tech Parks & Office Clusters**: Amplifies lunch (12:30–14:30) and tea (16:30–18:00) (+40%).
- **Transit Hubs (Railway / Bus Stations)**: Sustained all-day baseline with peak rush hours (07:30–10:00 & 18:00–21:30) (+35%).
- **Hospitals**: Consistent demand with morning and evening visitor hours (+25%).

---

## 4. Counterfactual Baseline Comparison

To ensure transparency and prevent unjustified cart movements, every generated itinerary is systematically benchmarked against a **Baseline Counterfactual**:
$$\text{Baseline: Staying at Origin Zone for 100\% of Shift Duration}$$

The itinerary derives:
- **Baseline Stay Profit**: Expected profit if the vendor never moves.
- **Optimized Shift Net Profit**: Total revenue minus COGS, fixed costs, and transit fuel expenses across all itinerary stops.
- **Profit Uplift (₹)**: $\text{Optimized Net Profit} - \text{Baseline Stay Profit}$.
- **Profit Uplift (%)**: $\frac{\text{Profit Uplift}}{\text{Baseline Stay Profit}} \times 100\%$.
- **Recommendation Verdict**:
  - `RECOMMENDED_SHIFT`: Net uplift $> ₹250/\text{hr}$ of shift duration.
  - `CONSIDER_SHIFT`: Net uplift between $₹100$ and $₹250/\text{hr}$.
  - `STAY_PUT_SHIFT`: Multi-stop itinerary fails to beat staying put after accounting for travel, fuel, and setup/teardown downtime.

---

## 5. API Endpoint Contract

```http
POST /v1/routes/shift-plan
X-API-Key: <api_key>
Content-Type: application/json
```

### Request Payload:
```json
{
  "latitude": 16.5062,
  "longitude": 80.6480,
  "vendor_category": "food",
  "shift_start_time": "2026-10-05T08:00:00Z",
  "shift_duration_hours": 8.0,
  "max_relocation_stops": 3,
  "min_dwell_time_minutes": 90,
  "setup_time_minutes": 15,
  "teardown_time_minutes": 15,
  "search_radius_km": 4.0
}
```

### Response Schema:
```json
{
  "status": "success",
  "shift_start_time": "2026-10-05T08:00:00Z",
  "shift_duration_hours": 8.0,
  "verdict": "RECOMMENDED_SHIFT",
  "recommendation_reason": "Optimized 2-stop shift generates ₹1,850.00 (+38.5%) net profit uplift over staying put, covering all transit and fuel costs.",
  "itinerary": [
    {
      "stop_number": 1,
      "h3_cell": "88619aa6c7fffff",
      "latitude": 16.5062,
      "longitude": 80.6480,
      "landmark_name": "Benz Circle Morning Market",
      "landmark_type": "market",
      "landmark_emoji": "🛍️",
      "is_current_cell": true,
      "arrival_time": "2026-10-05T08:00:00Z",
      "arrival_formatted": "08:00 AM",
      "setup_time_minutes": 0,
      "trading_start_time": "2026-10-05T08:00:00Z",
      "trading_start_formatted": "08:00 AM",
      "dwell_minutes": 210,
      "dwell_hours": 3.5,
      "trading_end_time": "2026-10-05T11:30:00Z",
      "trading_end_formatted": "11:30 AM",
      "teardown_time_minutes": 15,
      "departure_time": "2026-10-05T11:45:00Z",
      "departure_formatted": "11:45 AM",
      "travel_to_next_stop_minutes": 12,
      "travel_to_next_stop_km": 2.4,
      "fuel_cost_to_next_inr": 7.20,
      "expected_hourly_customers": 18,
      "customers_served_at_stop": 63,
      "gross_revenue_inr": 6300.00,
      "operating_costs_inr": 2268.00,
      "net_profit_inr": 3822.00
    },
    {
      "stop_number": 2,
      "h3_cell": "88619aa6c5fffff",
      "latitude": 16.5140,
      "longitude": 80.6550,
      "landmark_name": "MG Road Corporate Hub",
      "landmark_type": "office",
      "landmark_emoji": "🏢",
      "is_current_cell": false,
      "arrival_time": "2026-10-05T11:57:00Z",
      "arrival_formatted": "11:57 AM",
      "setup_time_minutes": 15,
      "trading_start_time": "2026-10-05T12:12:00Z",
      "trading_start_formatted": "12:12 PM",
      "dwell_minutes": 213,
      "dwell_hours": 3.55,
      "trading_end_time": "2026-10-05T15:45:00Z",
      "trading_end_formatted": "03:45 PM",
      "teardown_time_minutes": 15,
      "departure_time": "2026-10-05T16:00:00Z",
      "departure_formatted": "04:00 PM",
      "travel_to_next_stop_minutes": 0,
      "travel_to_next_stop_km": 0.0,
      "fuel_cost_to_next_inr": 0.0,
      "expected_hourly_customers": 24,
      "customers_served_at_stop": 85,
      "gross_revenue_inr": 8500.00,
      "operating_costs_inr": 3060.00,
      "net_profit_inr": 5158.00
    }
  ],
  "summary": {
    "total_stops": 2,
    "total_dwell_minutes": 423,
    "total_dwell_hours": 7.05,
    "total_transit_minutes": 12,
    "total_overhead_minutes": 45,
    "total_travel_distance_km": 2.4,
    "total_fuel_cost_inr": 7.20,
    "total_expected_customers": 148,
    "total_gross_revenue_inr": 14800.00,
    "total_net_profit_inr": 8972.80
  },
  "comparison_vs_stay": {
    "baseline_stay_profit_inr": 7122.80,
    "shift_plan_profit_inr": 8972.80,
    "profit_uplift_inr": 1850.00,
    "profit_uplift_pct": 25.97,
    "fuel_penalty_inr": 7.20,
    "relocation_count": 1
  },
  "solver_metadata": {
    "solver_status": "OPTIMAL",
    "solve_time_ms": 34.2,
    "algorithm": "Google OR-Tools CP-SAT (VRPTW-OP)",
    "timestamp": "2026-11-16T11:40:00Z"
  }
}
```

---

## 6. Test Suite & Validation Results

The shift optimizer is rigorously tested across 9 distinct scenarios in `tests/test_shift_optimization.py`:
1. `test_8_hour_shift_optimization`: Standard 8-hour shift produces feasible stops, non-overlapping timelines, and valid economics.
2. `test_12_hour_shift_optimization`: Long 12-hour shift accommodates multiple commercial windows (morning, lunch, evening rush).
3. `test_dynamic_time_window_resolution`: Resolves morning breakfast and lunch peaks with category multipliers.
4. `test_max_relocation_stops_limit`: Constrains itinerary to $\le K_{\max}$ stops.
5. `test_minimum_dwell_time_enforcement`: Ensures no stop trades for less than $D_{\min}$ (e.g. 90 minutes).
6. `test_travel_distance_and_fuel_accounting`: Validates distance matrices, OSRM transit durations, and fuel expense deduction.
7. `test_baseline_stay_profit_comparison`: Proves counterfactual profit computation and uplift percentages.
8. `test_infeasible_or_empty_candidates_fallback`: Gracefully falls back to a 1-stop safe origin stay when candidate pool is empty or constraints are infeasible.
9. `test_shift_plan_api_endpoint`: Validates end-to-end FastAPI endpoint contract, request handling, caching, and response schemas.

**Test Execution Summary:**
- Optimization suite: **9 passed, 0 failed in 6.73s**
- Full application suite: **98 passed, 0 failed in 8.95s**
