/* GeoDemand AI · live data, clear estimates, and small, purposeful interactions. */
'use strict';

const API_BASE = window.location.origin;
const $ = (id) => document.getElementById(id);
const state = {
  map: null,
  baseLayers: {},
  baseLayer: 'streets',
  origin: { lat: 16.5062, lng: 80.6480 },
  marker: null,
  radius: null,
  heatmap: null,
  pins: null,
  route: null,
  results: null,
  selectedCell: null,
  markers: new Map(),
  generation: 0,
  request: null,
  routeRequest: null,
  routeGeneration: 0,
  routeCache: new Map(),
  heatmapVisible: true,
  locating: false,
  loading: false,
};

function icon(name, className = '') {
  return `<svg class="icon ${className}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}
function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
function money(value) {
  return value != null && Number.isFinite(Number(value))
    ? `₹${Math.round(Number(value)).toLocaleString('en-IN')}` : '—';
}
function number(value, digits = 0) {
  return value != null && Number.isFinite(Number(value))
    ? Number(value).toLocaleString('en-IN', { maximumFractionDigits: digits }) : '—';
}
function sourceInfo(type) {
  if (!type) return ['UNKNOWN', 'badge-cached'];
  const s = String(type).toLowerCase();
  if (s === 'real_live' || s === 'live') return ['LIVE', 'badge-live'];
  if (s.includes('periodic') || s.includes('openstreetmap') || s.includes('overpass')) return ['PERIODIC', 'badge-real'];
  if (s.includes('offline') || s.includes('holiday')) return ['OFFICIAL', 'badge-real'];
  if (s.startsWith('real')) return ['REAL', 'badge-real'];
  if (s.includes('proxy') || s.includes('synthetic')) return ['PROXY', 'badge-synthetic'];
  if (s.includes('simulated')) return ['SIMULATED', 'badge-simulated'];
  if (s.includes('fallback')) return ['FALLBACK', 'badge-synthetic'];
  if (s.includes('derived')) return ['DERIVED', 'badge-derived'];
  if (s.includes('estimate') || s.includes('ml')) return ['ML ESTIMATE', 'badge-derived'];
  if (s.includes('manual')) return ['MANUAL', 'badge-cached'];
  return ['REAL', 'badge-real'];
}

let mapboxToken = null;

function buildBaseLayers(token) {
  if (token) {
    return {
      streets: L.tileLayer(`https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/{z}/{x}/{y}?access_token=${token}`, {
        attribution: '© <a href="https://www.mapbox.com/">Mapbox</a> © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        tileSize: 512, zoomOffset: -1, maxZoom: 22,
      }),
      satellite: L.tileLayer(`https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/{z}/{x}/{y}?access_token=${token}`, {
        attribution: '© <a href="https://www.mapbox.com/">Mapbox</a> © Maxar',
        tileSize: 512, zoomOffset: -1, maxZoom: 22,
      }),
      dark: L.tileLayer(`https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{z}/{x}/{y}?access_token=${token}`, {
        attribution: '© <a href="https://www.mapbox.com/">Mapbox</a> © OpenStreetMap',
        tileSize: 512, zoomOffset: -1, maxZoom: 22,
      }),
    };
  }
  return {
    streets: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }),
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Imagery © Esri and contributors', maxZoom: 19,
    }),
    dark: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles © Esri, DeLorme, NAVTEQ', maxZoom: 16,
    }),
  };
}

/* Maps are optional: a failed CDN must not break the rest of the dashboard. */
async function initMap() {
  if (!window.L) {
    $('mapUnavailable').classList.remove('hidden');
    ['expandMapBtn', 'recenterBtn', 'heatmapToggle', 'btnLayerStreets', 'btnLayerSatellite', 'btnLayerDark'].forEach((id) => { $(id).disabled = true; });
    return;
  }

  // Pre-fetch system config for Mapbox token
  try {
    const res = await fetch(`${API_BASE}/v1/system/config`);
    if (res.ok) {
      const cfg = await res.json();
      if (cfg && cfg.mapbox_token) mapboxToken = cfg.mapbox_token;
    }
  } catch (err) {
    console.info('Config fetch fallback to standard tiles', err);
  }

  state.map = L.map('map', { zoomControl: false }).setView([state.origin.lat, state.origin.lng], 13);
  L.control.zoom({ position: 'bottomright' }).addTo(state.map);

  state.baseLayers = buildBaseLayers(mapboxToken);

  Object.entries(state.baseLayers).forEach(([type, layer]) => {
    let failedTiles = 0;
    layer.on('loading', () => { failedTiles = 0; });
    layer.on('tileerror', () => {
      failedTiles += 1;
      if (type === state.baseLayer && failedTiles >= 4) {
        if (!layer._fallbackActive) {
          layer._fallbackActive = true;
          const fallback = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors', maxZoom: 19,
          });
          state.map.removeLayer(layer);
          fallback.addTo(state.map);
          state.baseLayers[type] = fallback;
        }
      }
    });
    layer.on('tileload', () => { if (type === state.baseLayer) $('mapUnavailable').classList.add('hidden'); });
  });

  state.baseLayers.streets.addTo(state.map);
  state.heatmap = L.layerGroup().addTo(state.map);
  state.pins = L.layerGroup().addTo(state.map);
  state.route = L.layerGroup().addTo(state.map);

  state.map.on('click', (event) => {
    const lat = event.latlng.lat;
    const lng = ((event.latlng.lng + 180) % 360 + 360) % 360 - 180;
    if (lat < -90 || lat > 90) return;
    $('latInput').value = lat.toFixed(6);
    $('lngInput').value = lng.toFixed(6);
    invalidateResults();
    updateOrigin(lat, lng);
    announce('Starting point updated. Run a new analysis for this location.');
  });

  updateOrigin(state.origin.lat, state.origin.lng);

  if (window.ResizeObserver) {
    new ResizeObserver(() => state.map.invalidateSize()).observe($('map'));
  }
}

function updateOrigin(lat, lng, center = false) {
  state.origin = { lat, lng };
  $('mapCoordinates').textContent = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
  $('mapLocationName').textContent = Math.abs(lat - 16.5062) < .01 && Math.abs(lng - 80.6480) < .01 ? 'Vijayawada' : 'Starting point';
  if (!state.map) return;
  if (state.marker) state.map.removeLayer(state.marker);
  state.marker = L.marker([lat, lng], {
    icon: L.divIcon({ className: '', html: '<div class="origin-marker"><span></span></div>', iconSize: [24, 24], iconAnchor: [12, 12] }),
    zIndexOffset: 1000,
  }).addTo(state.map).bindPopup(`<div class="popup-title">Your starting point</div><div>${lat.toFixed(5)}, ${lng.toFixed(5)}</div><div class="popup-note">Click the map or edit your coordinates to move this point.</div>`);
  updateRadius();
  if (center) state.map.setView([lat, lng], 14);
}

function updateRadius() {
  const radius = Number($('searchRadius').value);
  $('radiusValue').textContent = `${radius} km`;
  $('mapRadius').textContent = `${radius} km radius`;
  $('zonesMeta').textContent = `Find opportunities within ${radius} km`;
  $('searchRadius').style.setProperty('--range-progress', `${(radius - .5) / 7.5 * 100}%`);
  if (!state.map) return;
  if (state.radius) state.map.removeLayer(state.radius);
  state.radius = L.circle([state.origin.lat, state.origin.lng], {
    radius: radius * 1000, color: '#6b9580', fillColor: '#81af89', fillOpacity: .035, weight: 1.5, dashArray: '5 6', interactive: false,
  }).addTo(state.map);
}

function setBaseLayer(type) {
  if (!state.map || !state.baseLayers[type]) return;
  state.map.removeLayer(state.baseLayers[state.baseLayer]);
  state.baseLayer = type;
  $('mapUnavailable').classList.add('hidden');
  state.baseLayers[type].addTo(state.map);
  $('map').dataset.layer = type;
  document.querySelectorAll('[data-layer]').forEach((button) => {
    const active = button.dataset.layer === type;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}
function toggleHeatmap() {
  state.heatmapVisible = !state.heatmapVisible;
  $('heatmapToggle').setAttribute('aria-pressed', String(state.heatmapVisible));
  if (state.map && state.heatmap) {
    if (state.heatmapVisible) state.heatmap.addTo(state.map);
    else state.map.removeLayer(state.heatmap);
  }
}
function toggleExpandedMap(forceClose = false) {
  const expanded = !forceClose && !$('mapCard').classList.contains('expanded');
  $('mapCard').classList.toggle('expanded', expanded);
  $('expandMapBtn').setAttribute('aria-pressed', String(expanded));
  $('expandMapBtn').setAttribute('aria-label', expanded ? 'Restore map size' : 'Expand map');
  if (state.map) requestAnimationFrame(() => state.map.invalidateSize());
}

/* Updating any input invalidates the entire forecast, not just its pins. */
function invalidateResults() {
  const wasAnalyzed = Boolean(state.results || state.loading);
  state.generation += 1;
  state.request?.abort();
  state.request = null;
  state.routeGeneration += 1;
  state.routeRequest?.abort();
  state.routeRequest = null;
  state.routeCache.clear();
  state.results = null;
  state.selectedCell = null;
  state.markers.clear();
  state.heatmap?.clearLayers();
  state.pins?.clearLayers();
  state.route?.clearLayers();
  $('recsList').replaceChildren();
  $('decisionBanner').classList.add('hidden');
  ['currentProfit', 'currentCustomers', 'bestProfit', 'zonesCount'].forEach((id) => { $(id).textContent = '—'; });
  $('profitUplift').className = 'summary-foot';
  $('profitUplift').innerHTML = '<span class="mini-dot"></span>Discover your earning potential';
  $('currentProfitMeta').textContent = 'Your starting location · estimate';
  $('customersMeta').textContent = 'At your starting point · ML forecast';
  $('zonesMeta').textContent = `Find opportunities within ${$('searchRadius').value} km`;
  $('recsCount').textContent = '0';
  $('evalCount').textContent = 'Good opportunities start with a little insight.';
  $('sortRecs').disabled = true;
  $('exportBtn').disabled = true;
  $('h3CellDisplay').textContent = 'Not analyzed yet';
  $('lastUpdated').textContent = 'Not run yet';
  $('weatherTemperature').textContent = '—°';
  $('weatherText').textContent = 'Weather appears after analysis';
  $('weatherIcon').innerHTML = icon('sun');
  $('weatherBadge').textContent = 'NOT CHECKED';
  $('weatherBadge').className = 'provenance-badge badge-cached';
  $('waterWarning').classList.add('hidden');
  $('freshnessTable').innerHTML = `<div class="sources-empty">${icon('layers')}<p>Run an analysis to see the sources used for your forecast.</p></div>`;
  clearError();
  setLoading(false);
  setEmptyState('Your next opportunity is out there.', 'Set your starting point and business details. We’ll compare nearby locations so you can make your next move with confidence.');
  if (wasAnalyzed) $('formNote').innerHTML = `${icon('info')}Details changed. Run a fresh analysis.`;
}

async function useCurrentLocation() {
  if (state.locating) return;
  const button = $('useLocationBtn');
  if (!navigator.geolocation) {
    showError('Your browser doesn’t support device location. Enter coordinates or choose a point on the map.');
    return;
  }
  clearError();
  state.locating = true;
  button.disabled = true;
  button.innerHTML = `${icon('target')}Finding your location…`;
  navigator.geolocation.getCurrentPosition((position) => {
    $('latInput').value = position.coords.latitude.toFixed(6);
    $('lngInput').value = position.coords.longitude.toFixed(6);
    invalidateResults();
    updateOrigin(position.coords.latitude, position.coords.longitude, true);
    finish();
    announce('Device location set. You can now find your best locations.');
  }, (error) => {
    const reason = error.code === 1 ? 'Location permission was denied.' : error.code === 3 ? 'Finding your location took too long.' : 'Your device location is unavailable.';
    showError(`${reason} You can still enter coordinates or choose a point on the map.`);
    finish();
  }, { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 });
  function finish() {
    state.locating = false;
    button.disabled = false;
    button.innerHTML = `${icon('target')}Use my current location`;
  }
}

async function getRecommendations(event) {
  event?.preventDefault();
  if (state.locating) {
    showError('Please wait for your device location, then run the analysis.');
    return;
  }
  if (!$('analysisForm').reportValidity()) return;
  const lat = Number($('latInput').value);
  const lng = Number($('lngInput').value);
  const aov = Number($('aov').value);
  const radius = Number($('searchRadius').value);
  if (!Number.isFinite(lat) || !Number.isFinite(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180 || !Number.isFinite(aov) || aov < 20 || aov > 500 || !Number.isFinite(radius) || radius < .5 || radius > 8) {
    showError('Check your coordinates, order value (₹20–₹500), and search radius (0.5–8 km).');
    return;
  }
  invalidateResults();
  const generation = state.generation;
  const controller = new AbortController();
  state.request = controller;
  const timeout = setTimeout(() => controller.abort('timeout'), 120000);
  updateOrigin(lat, lng, true);
  setLoading(true);
  setEmptyState('Finding the right spots for you…', 'Comparing demand, profit, and local conditions. This can take a moment.', true);
  try {
    const response = await fetch(`${API_BASE}/v1/recommendations/live`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ latitude: lat, longitude: lng, vendor_category: $('vendorCategory').value, average_order_value: aov, search_radius_km: radius, top_n: 5 }),
    });
    const data = await response.json();
    if (generation !== state.generation) return;
    if (!response.ok) {
      const detail = data.detail;
      const message = typeof detail === 'string' ? detail : detail?.error || `The analysis service returned an error (${response.status}).`;
      throw new Error(`${message}${detail?.fix ? `\n${detail.fix}` : '\nPlease check your details and try again.'}`);
    }
    state.results = data;
    renderResults(data);
    $('formNote').innerHTML = `${icon('check')}Analysis ready. Explore your opportunities.`;
    announce(`Analysis complete. ${data.recommendations?.length || 0} locations found.`);
  } catch (error) {
    if (generation !== state.generation) return;
    const message = controller.signal.aborted
      ? 'The analysis took too long. Please try again, or choose a smaller search radius.'
      : error instanceof TypeError || error instanceof SyntaxError
        ? 'We couldn’t connect to the analysis service. Check your connection and make sure the API server is running, then try again.'
        : error.message;
    showError(message);
    setEmptyState('Let’s try that again.', 'Your forecast couldn’t be completed. Check the message above, adjust your details if needed, and run a new analysis.');
  } finally {
    clearTimeout(timeout);
    if (generation === state.generation) {
      state.request = null;
      setLoading(false);
    }
  }
}

function renderResults(data) {
  const recs = data.recommendations || [];
  const current = data.current_location || {};
  const lat = current.latitude ?? state.origin.lat;
  const lng = current.longitude ?? state.origin.lng;
  updateOrigin(lat, lng);
  $('currentProfit').textContent = money(data.current_expected_profit_inr);
  $('currentCustomers').textContent = number(data.current_estimated_demand);
  const best = recs.reduce((top, rec) => !top || rec.expected_profit_inr > top.expected_profit_inr ? rec : top, null);
  $('bestProfit').textContent = money(best?.expected_profit_inr);
  if (best && data.current_expected_profit_inr != null) {
    const uplift = best.expected_profit_inr - data.current_expected_profit_inr;
    $('profitUplift').textContent = `${uplift >= 0 ? '+' : '−'}${money(Math.abs(uplift))}/hr vs. your starting point`;
    $('profitUplift').className = `summary-foot ${uplift >= 0 ? 'positive' : 'negative'}`;
  } else $('profitUplift').textContent = 'No nearby profit estimate available';
  $('zonesCount').textContent = number(data.total_candidates_evaluated);
  $('zonesMeta').textContent = `Land zones within ${number(data.search_radius_km ?? $('searchRadius').value, 1)} km`;
  $('recsCount').textContent = String(recs.length);
  $('evalCount').textContent = `${number(data.total_candidates_evaluated)} land zones compared · Hourly forecasts, tailored to your business`;
  $('sortRecs').disabled = recs.length === 0;
  $('exportBtn').disabled = recs.length === 0;
  $('h3CellDisplay').textContent = current.h3_cell || 'Unavailable';
  $('waterWarning').classList.toggle('hidden', !current.is_water);
  const generatedAt = new Date(data.generated_at);
  $('lastUpdated').textContent = Number.isNaN(generatedAt.getTime()) ? 'Just now' : generatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  renderWeather(data.current_weather || {});
  renderFreshness(data.data_freshness || {}, data.data_quality || {});
  renderDecision(data.decision);
  if (recs.length) $('recsPlaceholder').classList.add('hidden');
  else setEmptyState('Your starting point may be the best spot.', 'No nearby destinations were returned for these settings. Review the guidance above, or try a different radius or starting point.');
  renderCards();
  renderMapData(data);
  if (recs.length) selectRecommendation(recs[0].h3_cell, false);
}

function renderWeather(weather) {
  const condition = weather.condition || 'Conditions unavailable';
  $('weatherText').textContent = condition.replace(/_/g, ' ');
  $('weatherTemperature').textContent = weather.temperature_c == null ? '—°' : `${number(weather.temperature_c, 1)}°C`;
  const lowered = condition.toLowerCase();
  $('weatherIcon').innerHTML = lowered.includes('rain') ? '<span aria-label="Rain">☂</span>' : lowered.includes('cloud') ? '<span aria-label="Cloudy">☁</span>' : icon('sun');
  const [label, cls] = sourceInfo(weather.source_type);
  $('weatherBadge').textContent = label;
  $('weatherBadge').className = `provenance-badge ${cls}`;
}
function renderFreshness(freshness, quality) {
  const f = freshness || {};
  const q = quality || {};
  const sources = [
    ['Weather', f.weather, q.weather],
    ['Traffic', f.traffic, q.traffic],
    ['Places & POIs', f.static_poi, q.poi],
    ['Competition', f.competition, q.competition],
    ['Events', f.events, q.events],
    ['Calendar', f.calendar, 'offline'],
    ['Fuel price', null, q.fuel || 'manual'],
    ['Demand model', null, 'ml_estimate'],
    ['Revenue & profit', null, 'derived'],
  ];
  $('freshnessTable').innerHTML = sources.map(([label, data, type]) => {
    const rawType = data?.source_type || type;
    const [badge, cls] = sourceInfo(rawType);
    const age = data?.cache_age_seconds;
    const provider = data?.source_name === 'india_holiday_calendar'
      ? 'Official AP & India Holiday Calendar'
      : data?.source_name === 'ticketmaster_api'
      ? 'Ticketmaster Discovery API'
      : data?.source_name
      ? data.source_name.replace(/_/g, ' ')
      : (label === 'Fuel price' ? 'Manual local pump' : label === 'Calendar' ? 'Official AP public holidays' : '');
    return `
      <div class="freshness-row">
        <span class="freshness-label">${label}</span>
        <span class="provenance-badge ${cls}">${badge}</span>
        ${provider ? `<span class="source-provider-note">${escapeHTML(provider)}</span>` : ''}
        ${data?.from_cache ? `<span class="source-age">Cached${age != null ? ` · ${number(age)}s old` : ''}</span>` : ''}
      </div>`;
  }).join('');
}
function renderDecision(decision) {
  const banner = $('decisionBanner');
  if (!decision) { banner.classList.add('hidden'); return; }
  const kind = decision.verdict === 'STAY_PUT' ? 'stay' : decision.verdict === 'CONSIDER_MOVE' ? 'consider' : 'move';
  banner.className = `decision-banner verdict-${kind}`;
  banner.innerHTML = `<span class="decision-symbol">${icon(kind === 'stay' ? 'pin' : kind === 'consider' ? 'info' : 'trend')}</span><div><div class="decision-headline">${escapeHTML(decision.headline)}</div><p class="decision-rationale">${escapeHTML(decision.rationale || decision.reason)}</p></div><span class="decision-tag">${escapeHTML(decision.action || 'DECISION GUIDANCE')}</span>`;
}
function sortedRecommendations() {
  const recs = [...(state.results?.recommendations || [])];
  const key = $('sortRecs').value;
  if (key === 'profit') recs.sort((a, b) => b.expected_profit_inr - a.expected_profit_inr);
  else if (key === 'distance') recs.sort((a, b) => a.distance_km - b.distance_km);
  else if (key === 'demand') recs.sort((a, b) => b.expected_customers - a.expected_customers);
  else recs.sort((a, b) => a.rank - b.rank);
  return recs;
}
function renderCards() {
  const list = $('recsList');
  list.replaceChildren();
  sortedRecommendations().forEach((rec) => list.appendChild(buildRecCard(rec)));
}
function buildRecCard(rec) {
  const card = document.createElement('article');
  card.className = `rec-card${rec.rank === 1 ? ' rank-1' : ''}${state.selectedCell === rec.h3_cell ? ' selected' : ''}`;
  card.dataset.cell = rec.h3_cell;
  const delta = rec.profit_improvement_inr;
  const interval = rec.prediction_interval || rec.confidence_interval || {};
  const uncertainty = interval.uncertainty_level || interval.certainty_level || 'Not available';
  const hasInterval = interval.p10 != null && interval.p90 != null;
  const medianPosition = hasInterval && interval.p90 > interval.p10 && interval.p50 != null ? Math.max(0, Math.min(100, (interval.p50 - interval.p10) / (interval.p90 - interval.p10) * 100)) : 50;
  const comp = ['low', 'medium', 'high'].includes(rec.competition_level) ? rec.competition_level : 'unknown';
  const time = rec.estimated_travel_time_min == null ? '—' : `${number(rec.estimated_travel_time_min)} min`;
  card.innerHTML = `
    <div class="rec-header"><span class="rec-rank">${number(rec.rank)}</span><div><h3 class="rec-title">${rec.rank === 1 ? 'Top recommended area' : `Recommended area ${number(rec.rank)}`}</h3><p class="rec-coordinates">${number(rec.latitude, 4)}, ${number(rec.longitude, 4)}</p></div>${rec.rank === 1 ? '<span class="rec-tag">TOP PICK</span>' : ''}</div>
    <div class="rec-profit">${money(rec.expected_profit_inr)}<span>/ hour · est. profit</span></div>
    <div class="profit-improvement ${delta == null ? '' : delta >= 0 ? 'positive' : 'negative'}">${delta == null ? 'Estimated profit at this location' : `${delta >= 0 ? '+' : '−'}${money(Math.abs(delta))}/hr compared with staying here`}</div>
    ${rec.explanation ? `<p class="rec-explanation-text">${escapeHTML(rec.explanation)}</p>` : ''}
    <div class="rec-metrics"><div><div class="metric-value">${number(rec.expected_customers)}</div><div class="metric-label">Customers / hr</div></div><div><div class="metric-value">${time}</div><div class="metric-label">Est. travel time</div></div><div><div class="metric-value">${number(rec.distance_km, 2)} km</div><div class="metric-label">Distance</div></div></div>
    <div class="rec-signals"><span class="signal-chip chip-comp-${comp}">${escapeHTML(comp)} competition</span>${rec.weather_condition ? `<span class="signal-chip">${escapeHTML(rec.weather_condition.replace(/_/g, ' '))}</span>` : ''}${rec.decision_action ? `<span class="signal-chip">${escapeHTML(rec.decision_action)}</span>` : ''}</div>
    <button class="btn rec-select" type="button" aria-label="View recommended area ${number(rec.rank)} on map" aria-pressed="${state.selectedCell === rec.h3_cell}">${icon('map')}View on map${icon('arrow', 'button-arrow')}</button>
    <details class="rec-details"><summary>Why this location & forecast details</summary><div class="confidence-row"><span>Estimated revenue / hr</span><strong>${money(rec.expected_revenue_inr)}</strong></div><div class="confidence-row"><span>P10–P90 customer range</span><strong>${hasInterval ? `${number(interval.p10)}–${number(interval.p90)} / hr` : 'Unavailable'}</strong></div>${hasInterval ? `<div class="interval-bar" style="--median-position:${medianPosition}%" aria-hidden="true"></div>` : ''}<div class="confidence-row"><span>Uncertainty</span><strong>${escapeHTML(uncertainty)}</strong></div><div class="confidence-row"><span>Median (P50)</span><strong>${number(interval.p50)} customers / hr</strong></div><div class="confidence-row"><span>Method</span><strong>${interval.uncertainty_method === 'quantile_regression' ? 'Quantile regression' : 'Approximate / unspecified'}</strong></div>${rec.top_drivers?.length ? `<div class="rec-drivers"><div class="rec-drivers-title">What’s driving the opportunity</div><ul>${rec.top_drivers.map((driver) => `<li>${escapeHTML(driver)}</li>`).join('')}</ul></div>` : ''}<div class="confidence-row"><span>H3 cell</span><strong>${escapeHTML(rec.h3_cell)}</strong></div></details>`;
  card.querySelector('.rec-select').addEventListener('click', () => selectRecommendation(rec.h3_cell, true));
  return card;
}

function renderMapData(data) {
  if (!state.map) return;
  state.heatmap.clearLayers();
  state.pins.clearLayers();
  state.markers.clear();
  (data.demand_heatmap || []).forEach((cell) => {
    if (!cell.boundary?.length) return;
    const intensity = cell.demand_intensity ?? 0;
    const color = intensity >= .75 ? '#287b53' : intensity >= .5 ? '#63a579' : intensity >= .25 ? '#9abb8b' : '#bdcdb1';
    L.polygon(cell.boundary, { color, fillColor: color, fillOpacity: .18 + intensity * .17, weight: cell.rank ? 1.7 : .7 }).addTo(state.heatmap)
      .bindTooltip(`<strong>${cell.rank ? `Recommended area ${number(cell.rank)}` : 'Demand zone'}</strong><br/>${number(cell.expected_customers)} customers / hr · ${money(cell.expected_profit_inr)} profit / hr<br/><small>Model estimates</small>`, { sticky: true });
  });
  const points = [[state.origin.lat, state.origin.lng]];
  (data.recommendations || []).forEach((rec) => {
    points.push([rec.latitude, rec.longitude]);
    const marker = L.marker([rec.latitude, rec.longitude], { icon: L.divIcon({ className: '', html: `<div class="rank-marker${rec.rank === 1 ? ' top' : ''}">${number(rec.rank)}</div>`, iconSize: [30, 30], iconAnchor: [15, 30] }) }).addTo(state.pins);
    marker.bindPopup(`<div class="popup-title">Recommended area ${number(rec.rank)}</div><div class="popup-row"><span>Est. hourly profit</span><strong>${money(rec.expected_profit_inr)}</strong></div><div class="popup-row"><span>Customers / hr</span><strong>${number(rec.expected_customers)}</strong></div><div class="popup-row"><span>Est. travel time</span><strong>${number(rec.estimated_travel_time_min)} min</strong></div><div class="popup-note">Forecast, not guaranteed earnings.</div>`);
    marker.on('click', () => selectRecommendation(rec.h3_cell, false));
    state.markers.set(rec.h3_cell, marker);
  });
  if (points.length > 1) state.map.fitBounds(points, { paddingTopLeft: [45, 70], paddingBottomRight: [45, 40], maxZoom: 15 });
}

async function selectRecommendation(cell, focusMap) {
  const rec = state.results?.recommendations?.find((item) => item.h3_cell === cell);
  if (!rec) return;
  state.selectedCell = cell;
  document.querySelectorAll('.rec-card').forEach((card) => {
    const selected = card.dataset.cell === cell;
    card.classList.toggle('selected', selected);
    card.querySelector('.rec-select').setAttribute('aria-pressed', String(selected));
  });
  if (!state.map) { announce('Map unavailable. Location coordinates are shown on the recommendation card.'); return; }
  if (focusMap) {
    if (!$('mapCard').classList.contains('expanded')) $('explore').scrollIntoView({ behavior: 'smooth', block: 'start' });
    state.map.fitBounds([[state.origin.lat, state.origin.lng], [rec.latitude, rec.longitude]], { paddingTopLeft: [50, 75], paddingBottomRight: [50, 50], maxZoom: 16 });
    state.markers.get(cell)?.openPopup();
  }
  state.routeRequest?.abort();
  const routeGeneration = ++state.routeGeneration;
  const generation = state.generation;
  state.route.clearLayers();
  // Keep a visibly dashed preview while routing; never invent a street route.
  const preview = L.polyline([[state.origin.lat, state.origin.lng], [rec.latitude, rec.longitude]], { color: '#6b9580', weight: 2, dashArray: '5 7' }).addTo(state.route);
  preview.bindTooltip('Straight-line preview · loading road route', { sticky: true });
  const controller = new AbortController();
  state.routeRequest = controller;
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    let route = state.routeCache.get(cell);
    if (!route) {
      const params = new URLSearchParams({ origin_lat: state.origin.lat, origin_lng: state.origin.lng, dest_lat: rec.latitude, dest_lng: rec.longitude });
      const response = await fetch(`${API_BASE}/v1/routing/route?${params}`, { signal: controller.signal });
      if (!response.ok) throw new Error('Route unavailable');
      route = await response.json();
    }
    if (generation !== state.generation || routeGeneration !== state.routeGeneration) return;
    // The backend may return interpolated grid coordinates. These are not roads.
    if (route.code === 'Fallback' || route.source === 'grid_road_interpolation' || !route.coordinates?.length) throw new Error('Estimated geometry only');
    state.routeCache.set(cell, route);
    state.route.clearLayers();
    L.polyline(route.coordinates, { color: '#ffffff', weight: 7, opacity: .9, interactive: false }).addTo(state.route);
    L.polyline(route.coordinates, { color: '#28745b', weight: 4, opacity: .9 }).addTo(state.route).bindTooltip(`Road route · ${number(route.duration_min)} min · ${number(route.distance_km, 2)} km`, { sticky: true });
  } catch {
    if (generation === state.generation && routeGeneration === state.routeGeneration) preview.setTooltipContent('Road route unavailable · straight-line preview, not navigation');
  } finally {
    clearTimeout(timeout);
    if (routeGeneration === state.routeGeneration) state.routeRequest = null;
  }
}

function exportRecommendations() {
  if (!state.results?.recommendations?.length) return;
  // Neutralize formula injection when exported API text is opened in spreadsheets.
  const csv = (value) => {
    let text = String(value ?? '');
    if (typeof value === 'string' && /^[\s]*[=+\-@]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  };
  const rows = [['Rank', 'Latitude', 'Longitude', 'H3 cell', 'Est. customers/hr', 'Est. revenue INR/hr', 'Est. profit INR/hr', 'Distance km', 'Est. travel min', 'Decision', 'Generated at']];
  sortedRecommendations().forEach((rec) => rows.push([rec.rank, rec.latitude, rec.longitude, rec.h3_cell, rec.expected_customers, rec.expected_revenue_inr, rec.expected_profit_inr, rec.distance_km, rec.estimated_travel_time_min, rec.decision_action, state.results.generated_at]));
  const blob = new Blob(['\uFEFF', rows.map((row) => row.map(csv).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `geodemand-locations-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  announce('Recommendation forecasts exported as CSV.');
}

function setEmptyState(title, text, loading = false) {
  $('recsPlaceholder').classList.remove('hidden');
  $('recsPlaceholder').querySelector('h3').textContent = title;
  $('recsPlaceholder').querySelector('.empty-copy p').textContent = text;
  $('recsPlaceholder').querySelector('.empty-steps').classList.toggle('hidden', loading);
  $('setupBtn').classList.toggle('hidden', loading);
}
function setLoading(loading) {
  state.loading = loading;
  $('getRecsBtn').disabled = loading;
  $('getRecsBtn').innerHTML = loading ? `${icon('clock')}Finding your best locations…` : `${icon('spark')}Find my best locations${icon('arrow', 'button-arrow')}`;
  $('loadingBar').classList.toggle('hidden', !loading);
  $('recsList').setAttribute('aria-busy', String(loading));
  $('formNote').innerHTML = loading
    ? `${icon('info')}Comparing local signals. Hang tight.`
    : state.results ? `${icon('check')}Analysis ready. Explore your opportunities.`
      : `${icon('check')}Built on local signals. Powered by AI.`;
}
function showError(message) { $('errorMsg').textContent = message; $('errorMsg').classList.remove('hidden'); }
function clearError() { $('errorMsg').textContent = ''; $('errorMsg').classList.add('hidden'); }
function announce(message) { $('announcement').textContent = message; }

async function checkSystemHealth() {
  try {
    const response = await fetch(`${API_BASE}/v1/system/data-health`, { signal: AbortSignal.timeout(8000) });
    if (!response.ok) throw new Error('Offline');
    const data = await response.json();
    const ready = data.model?.status === 'loaded';
    document.querySelector('.status-dot').className = `status-dot ${ready ? 'status-ok' : 'status-error'}`;
    $('statusText').textContent = ready ? 'System ready' : data.model?.status === 'not_trained' ? 'Model not trained' : 'Model unavailable';
    $('systemStatus').title = ready ? 'Demand model loaded. Weather and other signals are checked during analysis.' : 'The API is online, but the demand model is not ready.';
  } catch {
    document.querySelector('.status-dot').className = 'status-dot status-error';
    $('statusText').textContent = 'API offline';
    $('systemStatus').title = 'Start the API with uvicorn src.api.app:app --reload';
  }
}
function toggleMenu(open) {
  $('sidebar').classList.toggle('open', open);
  $('sidebarScrim').classList.toggle('hidden', !open);
  $('menuBtn').setAttribute('aria-expanded', String(open));
  $('menuBtn').setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
}
function focusSetup() {
  $('analysisForm').scrollIntoView({ behavior: 'smooth', block: 'center' });
  $('vendorCategory').focus({ preventScroll: true });
}

window.addEventListener('DOMContentLoaded', () => {
  $('todayDate').textContent = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  initMap();
  updateRadius();
  checkSystemHealth();
  setInterval(() => { if (!document.hidden) checkSystemHealth(); }, 60000);
  $('analysisForm').addEventListener('submit', getRecommendations);
  $('useLocationBtn').addEventListener('click', useCurrentLocation);
  $('sortRecs').addEventListener('change', renderCards);
  $('exportBtn').addEventListener('click', exportRecommendations);
  $('searchRadius').addEventListener('input', () => { invalidateResults(); updateRadius(); });
  const categoryDefaultAov = { food: 80, fruits: 60, salon: 150, repair: 180 };
  $('vendorCategory').addEventListener('change', (e) => {
    const cat = e.target.value;
    if (categoryDefaultAov[cat]) {
      $('aov').value = categoryDefaultAov[cat];
    }
    invalidateResults();
  });
  ['latInput', 'lngInput', 'aov', 'vendorCategory'].forEach((id) => {
    $(id).addEventListener('input', () => {
      invalidateResults();
      if (id === 'latInput' || id === 'lngInput') {
        const lat = Number($('latInput').value), lng = Number($('lngInput').value);
        if ($('latInput').value !== '' && $('lngInput').value !== '' && Number.isFinite(lat) && Number.isFinite(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) updateOrigin(lat, lng, true);
      }
    });
  });
  document.querySelectorAll('[data-layer]').forEach((button) => button.addEventListener('click', () => setBaseLayer(button.dataset.layer)));
  $('heatmapToggle').addEventListener('click', toggleHeatmap);
  $('expandMapBtn').addEventListener('click', () => toggleExpandedMap());
  $('recenterBtn').addEventListener('click', () => state.map?.setView([state.origin.lat, state.origin.lng], 14));
  $('setupBtn').addEventListener('click', focusSetup);
  document.querySelectorAll('[data-help]').forEach((button) => button.addEventListener('click', () => $('helpDialog').showModal()));
  $('closeHelpBtn').addEventListener('click', () => $('helpDialog').close());
  $('startGuideBtn').addEventListener('click', () => { $('helpDialog').close(); focusSetup(); });
  $('helpDialog').addEventListener('click', (event) => { if (event.target === $('helpDialog') && (event.clientX < $('helpDialog').getBoundingClientRect().left || event.clientX > $('helpDialog').getBoundingClientRect().right || event.clientY < $('helpDialog').getBoundingClientRect().top || event.clientY > $('helpDialog').getBoundingClientRect().bottom)) $('helpDialog').close(); });
  $('menuBtn').addEventListener('click', () => toggleMenu(!$('sidebar').classList.contains('open')));
  $('sidebarScrim').addEventListener('click', () => toggleMenu(false));
  document.querySelectorAll('.nav-link').forEach((link) => link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((item) => { item.classList.remove('active'); item.removeAttribute('aria-current'); });
    link.classList.add('active');
    link.setAttribute('aria-current', 'page');
    toggleMenu(false);
    toggleExpandedMap(true);
  }));
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { toggleMenu(false); toggleExpandedMap(true); } });
});
