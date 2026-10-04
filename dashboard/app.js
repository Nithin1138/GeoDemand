/* GeoDemand AI Dashboard — Google Maps Experience & Hyperlocal Demand Heatmap */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────────────
let map;
let baseLayers = {};
let currentBaseLayer = 'streets';
let currentMarker, searchRadiusCircle;
let heatmapLayer, routeLayer, recommendationLayer;
let currentLat = 16.5062, currentLng = 80.6480;

// ── Map Initialization ─────────────────────────────────
function initMap() {
  map = L.map('map', {
    zoomControl: false,
    attributionControl: true,
  }).setView([currentLat, currentLng], 14);

  // Zoom control in bottom right (Google Maps style)
  L.control.zoom({ position: 'bottomright' }).addTo(map);

  // Base Layers
  baseLayers.streets = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
    subdomains: 'abc',
  });

  baseLayers.satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and GIS User Community',
    maxZoom: 19,
  });

  baseLayers.dark = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles © Esri, DeLorme, NAVTEQ',
    maxZoom: 16,
  });

  // Default to clean streets
  baseLayers.streets.addTo(map);

  heatmapLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
  recommendationLayer = L.layerGroup().addTo(map);

  // Click on map to set location
  map.on('click', (e) => {
    document.getElementById('latInput').value = e.latlng.lat.toFixed(6);
    document.getElementById('lngInput').value = e.latlng.lng.toFixed(6);
    currentLat = e.latlng.lat;
    currentLng = e.latlng.lng;
    
    // Clear old recommendations immediately so old routes don't hang around
    clearRecommendations();
    
    // Update marker and radius preview
    placeCurrentMarker(currentLat, currentLng);
    updateRadiusCircle(currentLat, currentLng);
  });

  placeCurrentMarker(currentLat, currentLng);
  updateRadiusCircle(currentLat, currentLng);

  // Update radius circle when radius input changes
  document.getElementById('searchRadius').addEventListener('input', () => {
    updateRadiusCircle(currentLat, currentLng);
  });
}

function setBaseLayer(type) {
  if (baseLayers[currentBaseLayer]) {
    map.removeLayer(baseLayers[currentBaseLayer]);
  }
  if (baseLayers[type]) {
    baseLayers[type].addTo(map);
    currentBaseLayer = type;
  }
  
  // Update button active state
  document.querySelectorAll('.layer-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(
    type === 'streets' ? 'btnLayerStreets' : type === 'satellite' ? 'btnLayerSatellite' : 'btnLayerDark'
  );
  if (activeBtn) activeBtn.classList.add('active');
}

function updateRadiusCircle(lat, lng) {
  const radiusKm = parseFloat(document.getElementById('searchRadius').value) || 2.0;
  if (searchRadiusCircle) {
    map.removeLayer(searchRadiusCircle);
  }
  searchRadiusCircle = L.circle([lat, lng], {
    radius: radiusKm * 1000,
    color: '#3b82f6',
    fillColor: '#3b82f6',
    fillOpacity: 0.04,
    weight: 2,
    dashArray: '6, 6',
  }).addTo(map);
}

function placeCurrentMarker(lat, lng, isWater = false) {
  if (currentMarker) map.removeLayer(currentMarker);
  
  const icon = L.divIcon({
    className: '',
    html: `
      <div class="origin-beacon-wrapper">
        <div class="origin-radar-pulse"></div>
        <div class="origin-core-dot"></div>
        <div class="origin-label-tag">📍 YOU ARE HERE</div>
      </div>
    `,
    iconSize: [120, 40],
    iconAnchor: [60, 20],
  });
  
  const waterNote = isWater ? '<br><span style="color:#fbbf24;font-size:11px;">⚠️ River / Bridge Anchor</span>' : '';
  currentMarker = L.marker([lat, lng], { icon, zIndexOffset: 2000 }).addTo(map);
  currentMarker.bindPopup(`
    <div style="min-width:160px;padding:4px;">
      <div style="font-weight:700;color:#f59e0b;margin-bottom:4px;">📍 PREDICTION ORIGIN</div>
      <div style="font-size:12px;color:#cbd5e1;">Lat: ${lat.toFixed(5)}<br>Lng: ${lng.toFixed(5)}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px;">Search Radius: ${document.getElementById('searchRadius').value} km</div>
      ${waterNote}
    </div>
  `);
}

// ── Geolocation ────────────────────────────────────────
function useCurrentLocation() {
  const btn = document.getElementById('useLocationBtn');
  btn.textContent = '⏳ Getting location...';
  btn.disabled = true;

  if (!navigator.geolocation) {
    showError('Geolocation is not supported by your browser. Please enter coordinates manually.');
    btn.innerHTML = '<span class="btn-icon">🎯</span> Use My Current Location';
    btn.disabled = false;
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      currentLat = pos.coords.latitude;
      currentLng = pos.coords.longitude;
      document.getElementById('latInput').value = currentLat.toFixed(6);
      document.getElementById('lngInput').value = currentLng.toFixed(6);
      clearRecommendations();
      placeCurrentMarker(currentLat, currentLng);
      updateRadiusCircle(currentLat, currentLng);
      map.setView([currentLat, currentLng], 15);
      btn.innerHTML = '<span class="btn-icon">✅</span> Location Set!';
      btn.disabled = false;
      setTimeout(() => {
        btn.innerHTML = '<span class="btn-icon">🎯</span> Use My Current Location';
      }, 2000);
    },
    (err) => {
      showError(`Location access denied: ${err.message}. Using manual input.`);
      btn.innerHTML = '<span class="btn-icon">🎯</span> Use My Current Location';
      btn.disabled = false;
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
}

// ── Real Street Road Routing (Google Maps Style) ───────
async function fetchRoadRoute(startLat, startLng, destLat, destLng) {
  try {
    const url = `https://router.project-osrm.org/route/v1/driving/${startLng},${startLat};${destLng},${destLat}?overview=full&geometries=geojson`;
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const data = await resp.json();
    if (data.code === 'Ok' && data.routes && data.routes.length > 0) {
      const route = data.routes[0];
      const coords = route.geometry.coordinates.map(pt => [pt[1], pt[0]]);
      return {
        coordinates: coords,
        distanceKm: (route.distance / 1000).toFixed(2),
        durationMin: Math.max(1, Math.round(route.duration / 60)),
      };
    }
  } catch (e) {
    // Graceful fallback to straight line if offline
  }
  return null;
}

// ── Recommendations ────────────────────────────────────
async function getRecommendations() {
  const lat = parseFloat(document.getElementById('latInput').value);
  const lng = parseFloat(document.getElementById('lngInput').value);
  const category = document.getElementById('vendorCategory').value;
  const aov = parseFloat(document.getElementById('aov').value);
  const radius = parseFloat(document.getElementById('searchRadius').value);

  if (isNaN(lat) || isNaN(lng)) {
    showError('Please enter valid latitude and longitude values.');
    return;
  }

  currentLat = lat;
  currentLng = lng;
  placeCurrentMarker(lat, lng);
  updateRadiusCircle(lat, lng);
  map.setView([lat, lng], 14);

  setLoading(true);
  clearError();
  clearRecommendations();

  // Get location context first
  try {
    const ctxResp = await fetch(`${API_BASE}/v1/location/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latitude: lat, longitude: lng }),
    });
    if (ctxResp.ok) {
      const ctx = await ctxResp.json();
      document.getElementById('h3CellDisplay').textContent = ctx.h3_cell;
      document.getElementById('locationInfo').classList.remove('hidden');
      if (ctx.is_water) {
        placeCurrentMarker(lat, lng, true);
      }
    }
  } catch (e) { /* non-critical */ }

  // Get recommendations
  try {
    const resp = await fetch(`${API_BASE}/v1/recommendations/live`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: lat, longitude: lng,
        vendor_category: category,
        average_order_value: aov,
        search_radius_km: radius,
        top_n: 5,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      const detail = err.detail;
      if (typeof detail === 'object' && detail.fix) {
        showError(`${detail.error}\n\nFix: ${detail.fix}`);
      } else {
        showError(`API error ${resp.status}: ${JSON.stringify(detail)}`);
      }
      return;
    }

    const data = await resp.json();
    await renderResults(data);

  } catch (e) {
    showError(`Network error: ${e.message}\n\nMake sure the API server is running:\nuvicorn src.api.app:app --reload`);
  } finally {
    setLoading(false);
  }
}

// ── Render Results & Heatmap ───────────────────────────
async function renderResults(data) {
  const startLat = data.current_location?.latitude || currentLat;
  const startLng = data.current_location?.longitude || currentLng;
  const startCell = data.current_location?.h3_cell || '—';
  const isStartWater = data.current_location?.is_water || false;
  const searchRadiusKm = data.search_radius_km || parseFloat(document.getElementById('searchRadius').value) || 2.0;

  // Re-pin start marker and radius circle accurately
  placeCurrentMarker(startLat, startLng, isStartWater);
  updateRadiusCircle(startLat, startLng);

  // Current location stats
  const cur = data.current_estimated_demand;
  const curProfit = data.current_expected_profit_inr;
  document.getElementById('currentCustomers').textContent = cur ?? '—';
  document.getElementById('currentProfit').textContent = curProfit != null ? `₹${Math.round(curProfit).toLocaleString()}` : '—';

  const weather = data.current_weather || {};
  document.getElementById('weatherIcon').textContent = weatherIcon(weather.condition);
  document.getElementById('weatherText').textContent = `${weather.condition || 'unknown'} · ${weather.temperature_c ?? '—'}°C`;
  const weatherBadge = document.getElementById('weatherBadge');
  weatherBadge.textContent = sourceLabel(weather.source_type);
  weatherBadge.className = `provenance-badge ${sourceBadgeClass(weather.source_type)}`;

  document.getElementById('currentSection').style.display = '';

  // Data freshness & quality summary
  renderFreshness(data.data_freshness, data.model_info, data.data_quality);
  document.getElementById('freshnessSection').style.display = '';

  // Eval count
  document.getElementById('evalCount').textContent = `${data.total_candidates_evaluated} land candidates in ${searchRadiusKm} km`;
  document.getElementById('recsPlaceholder').style.display = 'none';

  // Clear previous layers
  heatmapLayer.clearLayers();
  routeLayer.clearLayers();
  recommendationLayer.clearLayers();

  // ── 1. Render Hyperlocal H3 Demand Hexagon Heatmap ──
  const heatmap = data.demand_heatmap || [];
  heatmap.forEach(cell => {
    if (!cell.boundary || cell.boundary.length === 0) return;

    const intensity = cell.demand_intensity || 0;
    // Color gradient based on predicted demand intensity
    let hexFillColor, hexStrokeColor, hexOpacity;
    if (intensity >= 0.75) {
      hexFillColor = '#10b981'; // Emerald - High Demand
      hexStrokeColor = '#059669';
      hexOpacity = 0.40;
    } else if (intensity >= 0.50) {
      hexFillColor = '#06b6d4'; // Cyan - Moderate-High Demand
      hexStrokeColor = '#0891b2';
      hexOpacity = 0.30;
    } else if (intensity >= 0.25) {
      hexFillColor = '#3b82f6'; // Blue - Moderate Demand
      hexStrokeColor = '#2563eb';
      hexOpacity = 0.22;
    } else {
      hexFillColor = '#64748b'; // Slate - Low Demand
      hexStrokeColor = '#475569';
      hexOpacity = 0.12;
    }

    const hexPolygon = L.polygon(cell.boundary, {
      fillColor: hexFillColor,
      fillOpacity: hexOpacity,
      color: hexStrokeColor,
      weight: cell.rank ? 2.5 : 1,
      dashArray: cell.rank ? null : '3, 3',
    }).addTo(heatmapLayer);

    hexPolygon.bindTooltip(`
      <div style="font-family:Inter,sans-serif;font-size:12px;padding:2px;">
        <div style="font-weight:700;color:${hexFillColor};">${cell.rank ? `🏆 Rank #${cell.rank} Area` : '📍 Demand Zone'}</div>
        <div>👥 <b>${cell.expected_customers}</b> customers/hr</div>
        <div>💰 ₹${Math.round(cell.expected_profit_inr).toLocaleString()} profit/hr</div>
        <div style="font-size:10px;color:#94a3b8;margin-top:2px;">H3: ${cell.h3_cell.substring(0, 10)}…</div>
      </div>
    `, { sticky: true, className: 'route-line-tooltip' });
  });

  // ── Render Strategic Decision Banner ──
  const banner = document.getElementById('decisionBanner');
  if (banner && data.decision) {
    const dec = data.decision;
    const isStay = dec.verdict === 'STAY_PUT';
    const isConsider = dec.verdict === 'CONSIDER_MOVE';
    const cls = isStay ? 'verdict-stay' : isConsider ? 'verdict-consider' : 'verdict-move';
    const tagCls = isStay ? 'decision-tag-stay' : isConsider ? 'decision-tag-consider' : 'decision-tag-move';
    const icon = isStay ? '🔵' : isConsider ? '🟡' : '🟢';

    banner.className = `decision-banner ${cls}`;
    banner.innerHTML = `
      <div class="decision-banner-header">
        <span class="decision-tag ${tagCls}">${icon} ${dec.action}</span>
        <span style="font-size:11px;color:#94a3b8;font-weight:600;">Strategic Decision Guidance</span>
      </div>
      <div class="decision-headline">${dec.headline}</div>
      <div class="decision-rationale">${dec.rationale}</div>
    `;
    banner.style.display = 'flex';
  } else if (banner) {
    banner.style.display = 'none';
  }

  const list = document.getElementById('recsList');
  list.innerHTML = '';

  // ── Prepend Origin Summary Card ──
  const originCard = document.createElement('div');
  originCard.className = 'origin-summary-card';
  originCard.innerHTML = `
    <div class="origin-summary-header">
      <span class="origin-tag">📍 PREDICTED FROM (ORIGIN)</span>
      <span class="origin-coords">${startLat.toFixed(4)}, ${startLng.toFixed(4)}</span>
    </div>
    <div class="origin-subtext">
      Origin Cell: <span class="mono" style="color:#fbbf24;">${startCell.substring(0, 10)}…</span> · 
      Radius: <span style="color:#38bdf8;font-weight:600;">≤ ${searchRadiusKm} km</span> ·
      Baseline: <b>${cur ?? '—'}</b> cust/hr (₹${Math.round(curProfit ?? 0).toLocaleString()})
      ${isStartWater ? '<div class="water-warning-pill">🌊 Starting point on river/bridge — Destinations strictly in verified land zones</div>' : ''}
    </div>
  `;
  list.appendChild(originCard);

  const recs = data.recommendations || [];
  const mapPoints = [[startLat, startLng]];

  // ── 2. Render Google Maps Street Driving Routes & Top Pins ──
  for (let idx = 0; idx < recs.length; idx++) {
    const rec = recs[idx];
    mapPoints.push([rec.latitude, rec.longitude]);

    const isTop = idx === 0;
    const pinColor = isTop ? '#10b981' : idx === 1 ? '#3b82f6' : idx === 2 ? '#06b6d4' : '#64748b';

    // Fetch real road navigation geometry from OSRM
    const roadRoute = await fetchRoadRoute(startLat, startLng, rec.latitude, rec.longitude);
    const routeCoords = roadRoute ? roadRoute.coordinates : [[startLat, startLng], [rec.latitude, rec.longitude]];
    const travelTimeMin = roadRoute ? roadRoute.durationMin : rec.estimated_travel_time_min;
    const roadDistKm = roadRoute ? roadRoute.distanceKm : rec.distance_km;

    // Outer road casing (Google Maps white outline effect)
    L.polyline(routeCoords, {
      color: '#ffffff',
      weight: isTop ? 7 : 5,
      opacity: 0.9,
      lineCap: 'round',
      lineJoin: 'round',
    }).addTo(routeLayer);

    // Inner route line (Google Maps Blue / Green)
    const routeLine = L.polyline(routeCoords, {
      color: isTop ? '#10b981' : '#2563eb',
      weight: isTop ? 4.5 : 3.5,
      opacity: 1.0,
      lineCap: 'round',
      lineJoin: 'round',
    }).addTo(routeLayer);

    routeLine.bindTooltip(`🚗 <b>~${travelTimeMin} min drive</b> (${roadDistKm} km via road)`, {
      sticky: true,
      className: 'route-line-tooltip',
    });

    // Destination Pin (Google Maps Teardrop Style Marker)
    const rankIcon = L.divIcon({
      className: '',
      html: `
        <div style="
          width:28px;height:34px;
          background:${pinColor};
          border-radius:50% 50% 50% 0;
          transform:rotate(-45deg);
          border:2px solid #ffffff;
          box-shadow:0 3px 10px rgba(0,0,0,0.45);
          display:flex;align-items:center;justify-content:center;
          position:relative;
        ">
          <span style="
            transform:rotate(45deg);
            color:#ffffff;
            font-family:Inter,sans-serif;
            font-weight:800;
            font-size:12px;
          ">${rec.rank}</span>
        </div>
      `,
      iconSize: [28, 34],
      iconAnchor: [14, 34],
      popupAnchor: [0, -32],
    });

    L.marker([rec.latitude, rec.longitude], { icon: rankIcon, zIndexOffset: 500 - idx })
      .bindPopup(buildPopup(rec, travelTimeMin, roadDistKm))
      .addTo(recommendationLayer);

    // Build recommendation card
    const card = buildRecCard(rec, curProfit, travelTimeMin, roadDistKm);
    list.appendChild(card);

    // Hover effect on card highlights route line
    card.onmouseenter = () => {
      routeLine.setStyle({ weight: 6, color: '#f59e0b' });
    };
    card.onmouseleave = () => {
      routeLine.setStyle({ weight: isTop ? 4.5 : 3.5, color: isTop ? '#10b981' : '#2563eb' });
    };
  }

  // Fit map to show both the starting origin AND all recommendation destinations
  if (mapPoints.length > 0) {
    const bounds = L.latLngBounds(mapPoints);
    map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });
  }
}

function buildPopup(rec, travelTimeMin, roadDistKm) {
  return `
    <div style="min-width:180px;padding:4px;">
      <div style="font-weight:700;color:#10b981;margin-bottom:4px;">🏆 Recommendation #${rec.rank}</div>
      <div style="color:#94a3b8;font-size:11px;margin-bottom:8px;">H3: ${rec.h3_cell}</div>
      <div style="background:#0f1620;border-radius:6px;padding:6px 8px;margin-bottom:8px;font-size:11px;color:#38bdf8;">
        🚗 <b>${travelTimeMin} min drive</b> (${roadDistKm} km via road)
      </div>
      <table style="width:100%;font-size:12px;border-collapse:collapse;">
        <tr><td style="color:#94a3b8;padding:2px 0;">Expected Customers</td><td style="text-align:right;font-weight:600;">${rec.expected_customers}/hr</td></tr>
        <tr><td style="color:#94a3b8;padding:2px 0;">Expected Revenue</td><td style="text-align:right;font-weight:600;">₹${Math.round(rec.expected_revenue_inr).toLocaleString()}</td></tr>
        <tr><td style="color:#94a3b8;padding:2px 0;">Expected Profit</td><td style="text-align:right;font-weight:600;color:#10b981;">₹${Math.round(rec.expected_profit_inr).toLocaleString()}</td></tr>
        <tr><td style="color:#94a3b8;padding:2px 0;">Competition</td><td style="text-align:right;font-weight:600;">${rec.competition_level}</td></tr>
      </table>
      ${rec.top_drivers?.length > 0 ? `<div style="margin-top:8px;padding-top:8px;border-top:1px solid #1e293b;color:#94a3b8;font-size:11px;">💡 ${rec.top_drivers[0]}</div>` : ''}
    </div>
  `;
}

function buildRecCard(rec, curProfit, travelTimeMin, roadDistKm) {
  const card = document.createElement('div');
  card.className = `rec-card${rec.rank === 1 ? ' rank-1' : ''}`;

  const improvPct = rec.profit_improvement_pct;
  const improvInr = rec.profit_improvement_inr;
  const hasImprove = improvPct != null;
  const isPositive = improvInr > 0;

  const rankBadgeClass = rec.rank === 1 ? 'rank-badge-1' : rec.rank === 2 ? 'rank-badge-2' : rec.rank === 3 ? 'rank-badge-3' : 'rank-badge-n';
  const compClass = rec.competition_level === 'low' ? 'chip-comp-low' : rec.competition_level === 'high' ? 'chip-comp-high' : 'chip-comp-med';
  const compEmoji = rec.competition_level === 'low' ? '🟢' : rec.competition_level === 'high' ? '🔴' : '🟡';

  const interval = rec.prediction_interval || rec.confidence_interval || {};
  const cert = interval.certainty_level ? ` · ${interval.certainty_level} certainty` : '';
  const verdictCls = rec.decision_verdict === 'STAY_PUT' ? 'chip-verdict-stay' : rec.decision_verdict === 'CONSIDER_MOVE' ? 'chip-verdict-consider' : 'chip-verdict-move';
  const verdictEmoji = rec.decision_verdict === 'STAY_PUT' ? '🔵' : rec.decision_verdict === 'CONSIDER_MOVE' ? '🟡' : '🟢';

    const uncLevel = (interval.uncertainty_level || interval.certainty_level || 'MODERATE').toUpperCase();

    card.innerHTML = `
    <div class="rec-header">
      <div class="rec-rank ${rankBadgeClass}">#${rec.rank}</div>
      <div class="rec-cell-id">${rec.h3_cell.substring(0, 10)}…</div>
      <div class="rec-distance">🚗 <b>${travelTimeMin} min</b> (${roadDistKm} km)</div>
    </div>

    <div class="move-direction-banner">
      <span>🚀</span>
      <span><b>~${travelTimeMin} min drive (${roadDistKm} km via road)</b> from origin point</span>
    </div>

    <div class="rec-metrics">
      <div class="metric-box">
        <div class="metric-value blue">${rec.expected_customers}</div>
        <div class="metric-label">Expected Cust/hr</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">₹${Math.round(rec.expected_revenue_inr).toLocaleString()}</div>
        <div class="metric-label">Exp. Revenue</div>
      </div>
      <div class="metric-box">
        <div class="metric-value green">₹${Math.round(rec.expected_profit_inr).toLocaleString()}</div>
        <div class="metric-label">Exp. Profit</div>
      </div>
    </div>

    ${hasImprove ? `
    <div class="profit-improvement ${isPositive ? 'positive' : 'negative'}">
      ${isPositive ? '📈' : '📉'} 
      ${isPositive ? '+' : ''}₹${Math.round(improvInr).toLocaleString()} vs staying put
      (${isPositive ? '+' : ''}${improvPct?.toFixed(1)}%)
    </div>
    ` : ''}

    <div class="rec-signals">
      <div class="signal-chip ${verdictCls}">${verdictEmoji} ${rec.decision_action || 'EVALUATED'}</div>
      <div class="signal-chip chip-time">⏱️ ${travelTimeMin} min drive</div>
      <div class="signal-chip chip-land">🟢 Valid Land Zone</div>
      <div class="signal-chip ${compClass}">${compEmoji} ${rec.competition_level} comp</div>
      <div class="signal-chip chip-weather">${weatherIcon(rec.weather_condition)} ${rec.weather_condition}</div>
      <div class="signal-chip chip-traffic">🚦 ${rec.traffic_level || 'medium'} traffic</div>
    </div>

    <div class="rec-confidence">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
        <span>📊 P10–P90 Prediction Interval:</span>
        <span class="confidence-range">${interval.p10 ?? '?'}–${interval.p90 ?? '?'} cust/hr</span>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;font-size:11px;color:#94a3b8;">
        <span>Uncertainty: <b style="color:${uncLevel === 'LOW' ? '#34d399' : uncLevel === 'HIGH' ? '#f87171' : '#fbbf24'};">${uncLevel}</b> (Median P50: ${interval.p50 ?? '—'})</span>
        <span style="font-size:10px;">${interval.uncertainty_method === 'quantile_regression' ? 'Quantile P10/P90' : 'Approximate'}</span>
      </div>
    </div>

    ${rec.top_drivers?.length > 0 ? `
    <div class="rec-drivers">
      <div class="rec-drivers-title">💡 Why this location</div>
      ${rec.top_drivers.map(d => `<div class="driver-item">${d}</div>`).join('')}
    </div>
    ` : ''}
  `;

  card.onclick = () => {
    map.setView([rec.latitude, rec.longitude], 16);
  };

  return card;
}

// ── Freshness & Data Quality Table ──────────────────────────
function renderFreshness(freshness, modelInfo, dataQuality) {
  const table = document.getElementById('freshnessTable');
  table.innerHTML = '';

  const sources = [
    { key: 'weather', label: '🌦 Weather', data: freshness?.weather, quality: dataQuality?.weather },
    { key: 'traffic', label: '🚦 Traffic', data: freshness?.traffic, quality: dataQuality?.traffic },
    { key: 'static_poi', label: '🗺 POI Data', data: freshness?.static_poi, quality: dataQuality?.poi },
    { key: 'competition', label: '⚡ Competition', data: freshness?.competition, quality: dataQuality?.competition },
    { key: 'events', label: '🎉 Events', data: freshness?.events, quality: dataQuality?.events },
    { key: 'calendar', label: '📅 Holidays', data: freshness?.calendar, quality: 'real' },
    { key: 'fuel', label: '⛽ Fuel Price', data: { source_type: 'manual' }, quality: 'manual' },
  ];

  sources.forEach(({ label, data, quality }) => {
    const row = document.createElement('div');
    row.className = 'freshness-row';
    const age = data?.cache_age_seconds ? `${Math.round(data.cache_age_seconds)}s ago` : '';
    const fromCache = data?.from_cache ? ` (cached)` : '';
    const sType = data?.source_type || quality || 'derived';
    
    row.innerHTML = `
      <span class="freshness-label">${label}</span>
      <div style="display:flex;align-items:center;gap:6px;">
        ${age ? `<span style="font-size:10px;color:#64748b;">${age}${fromCache}</span>` : ''}
        <span class="provenance-badge ${sourceBadgeClass(sType)}">${sourceLabel(sType)}</span>
      </div>
    `;
    table.appendChild(row);
  });

  // Model & Derived Business Rows
  const derivations = [
    { label: '🤖 Demand Model', badge: 'ML ESTIMATE', cls: 'badge-derived' },
    { label: '💰 Revenue & Profit', badge: 'DERIVED', cls: 'badge-derived' },
  ];
  derivations.forEach(({ label, badge, cls }) => {
    const row = document.createElement('div');
    row.className = 'freshness-row';
    row.innerHTML = `
      <span class="freshness-label">${label}</span>
      <span class="provenance-badge ${cls}">${badge}</span>
    `;
    table.appendChild(row);
  });
}

// ── Helpers ────────────────────────────────────────────
function weatherIcon(condition) {
  if (!condition) return '🌤';
  if (condition.includes('rain')) return '🌧';
  if (condition.includes('cloud')) return '☁️';
  return '☀️';
}

function sourceLabel(sourceType) {
  const map = {
    real_live: 'LIVE',
    real_periodic: 'REAL',
    real_offline: 'REAL',
    hybrid: 'REAL',
    synthetic_fallback: 'PROXY',
    simulated: 'SIM',
    derived: 'DERIVED',
  };
  return map[sourceType] || sourceType?.toUpperCase() || 'UNKNOWN';
}

function sourceBadgeClass(sourceType) {
  const map = {
    real_live: 'badge-live',
    real_periodic: 'badge-real',
    real_offline: 'badge-real',
    hybrid: 'badge-real',
    synthetic_fallback: 'badge-synthetic',
    simulated: 'badge-simulated',
    derived: 'badge-derived',
  };
  return map[sourceType] || 'badge-cached';
}

function setLoading(loading) {
  const btn = document.getElementById('getRecsBtn');
  const bar = document.getElementById('loadingBar');
  btn.disabled = loading;
  btn.innerHTML = loading
    ? '<span class="btn-icon">⏳</span> Analyzing locations...'
    : '<span class="btn-icon">🚀</span> Get Recommendations';
  bar.classList.toggle('hidden', !loading);
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg;
  el.classList.remove('hidden');
}

function clearError() {
  document.getElementById('errorMsg').classList.add('hidden');
}

function clearRecommendations() {
  document.getElementById('recsList').innerHTML = '';
  document.getElementById('recsPlaceholder').style.display = '';
  document.getElementById('evalCount').textContent = '';
  if (heatmapLayer) heatmapLayer.clearLayers();
  if (routeLayer) routeLayer.clearLayers();
  if (recommendationLayer) recommendationLayer.clearLayers();
}

// ── System Health ──────────────────────────────────────
async function checkSystemHealth() {
  try {
    const resp = await fetch(`${API_BASE}/v1/system/data-health`);
    const data = await resp.json();
    const dot = document.querySelector('.status-dot');
    const txt = document.getElementById('statusText');

    if (data.model?.status === 'loaded') {
      dot.className = 'status-dot status-ok';
      txt.textContent = `Ready · MAE ${data.model?.metrics?.test_mae ?? '?'} cust/hr`;
    } else if (data.model?.status === 'not_trained') {
      dot.className = 'status-dot status-error';
      txt.textContent = 'Model not trained';
    } else {
      dot.className = 'status-dot status-error';
      txt.textContent = 'Model not loaded';
    }
  } catch {
    const dot = document.querySelector('.status-dot');
    dot.className = 'status-dot status-error';
    document.getElementById('statusText').textContent = 'API offline';
  }
}

// ── Init ───────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initMap();
  checkSystemHealth();
  setInterval(checkSystemHealth, 30000);
});
