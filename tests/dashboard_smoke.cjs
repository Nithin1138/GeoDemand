/*
 * Browser smoke check for the real dashboard with deterministic API test fixtures.
 * Start the app first, then run with an existing Playwright installation:
 *   NODE_PATH=/path/to/node_modules DASHBOARD_URL=http://127.0.0.1:8000 node tests/dashboard_smoke.cjs
 * No fixtures are loaded by the production dashboard. No project dependencies are installed.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

const baseURL = process.env.DASHBOARD_URL || 'http://127.0.0.1:8000';
const fixtures = {
  generated_at: '2026-10-07T09:30:00Z',
  search_radius_km: 3,
  current_location: { latitude: 16.5062, longitude: 80.648, h3_cell: 'test-origin-cell', is_water: false },
  current_expected_profit_inr: 320,
  current_estimated_demand: 12,
  total_candidates_evaluated: 42,
  current_weather: { condition: 'clear', temperature_c: 28.5, source_type: 'real_live' },
  data_freshness: { weather: { source_type: 'real_live' }, traffic: { source_type: 'synthetic_fallback', from_cache: true, cache_age_seconds: 60 }, calendar: { source_type: 'real_offline' } },
  data_quality: { competition: 'real_periodic', poi: 'real_periodic', events: 'simulated', fuel: 'manual' },
  decision: { verdict: 'RECOMMENDED_MOVE', action: 'RECOMMENDED MOVE', headline: 'A nearby area offers more potential', rationale: 'Compare the profit uplift with the cost of travel.' },
  recommendations: [
    { rank: 1, h3_cell: 'test-cell-1', latitude: 16.517, longitude: 80.652, expected_customers: 24, expected_profit_inr: 680, expected_revenue_inr: 2400, profit_improvement_inr: 360, distance_km: 1.5, estimated_travel_time_min: 6, competition_level: 'low', weather_condition: 'clear', decision_action: 'RECOMMENDED MOVE', prediction_interval: { p10: 18, p50: 24, p90: 32, uncertainty_level: 'LOW', uncertainty_method: 'quantile_regression' }, top_drivers: ['Strong nearby foot traffic', '<script>window.injected = true</script>'] },
    { rank: 2, h3_cell: 'test-cell-2', latitude: 16.514, longitude: 80.661, expected_customers: 30, expected_profit_inr: 800, expected_revenue_inr: 3000, profit_improvement_inr: 480, distance_km: 2, estimated_travel_time_min: 8, competition_level: 'medium', weather_condition: 'clear', decision_action: 'CONSIDER MOVE', prediction_interval: { p10: 20, p50: 30, p90: 44, uncertainty_level: 'MODERATE' }, top_drivers: ['Nearby shopping areas'] },
    { rank: 3, h3_cell: 'test-cell-3', latitude: 16.509, longitude: 80.648, expected_customers: 15, expected_profit_inr: 400, expected_revenue_inr: 1500, profit_improvement_inr: 80, distance_km: .4, estimated_travel_time_min: 2, competition_level: 'high', weather_condition: 'cloudy', decision_action: 'CONSIDER MOVE', top_drivers: [] },
  ],
  demand_heatmap: [{ boundary: [[16.511, 80.650], [16.515, 80.650], [16.516, 80.654], [16.512, 80.656]], demand_intensity: .8, rank: 1, expected_customers: 24, expected_profit_inr: 680 }],
};

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  let checks = 0;
  const errors = [];
  const check = (condition, message) => { assert.ok(condition, message); checks += 1; };
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  let mode = 'success';
  let calls = 0;
  let resolveDelayed;
  let delayedSeen = false;
  let requestBody;
  await page.route('**/v1/system/data-health', (route) => route.fulfill({ json: { model: { status: 'loaded' } } }));
  await page.route('**/v1/recommendations/live', async (route) => {
    calls += 1;
    requestBody = route.request().postDataJSON();
    if (mode === 'failure') return route.fulfill({ status: 503, json: { detail: { error: 'Demand model is unavailable.', fix: 'Train the model and try again.' } } });
    if (mode === 'delayed') {
      delayedSeen = true;
      await new Promise((resolve) => { resolveDelayed = resolve; });
      await route.fulfill({ json: fixtures }).catch(() => {});
      return;
    }
    if (mode === 'empty') return route.fulfill({ json: { ...fixtures, recommendations: [], decision: { verdict: 'STAY_PUT', action: 'STAY HERE', headline: 'Stay at your current location', rationale: 'No nearby area justifies a move.' } } });
    return route.fulfill({ json: fixtures });
  });
  await page.route('**/v1/routing/route?**', (route) => route.fulfill({ json: { code: 'Fallback', source: 'grid_road_interpolation', coordinates: [[16.5062, 80.648], [16.517, 80.652]], duration_min: 5, distance_km: 1.5 } }));
  try {
    await page.goto(baseURL, { waitUntil: 'networkidle' });
    check(await page.locator('#statusText').textContent() === 'System ready', 'System health is shown');
    check(await page.locator('#currentProfit').textContent() === '—', 'No fabricated starting forecast');
    check(await page.locator('#recsPlaceholder').isVisible(), 'Useful initial empty state');
    check(await page.locator('#exportBtn').isDisabled(), 'Export disabled before results');
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Desktop has no horizontal overflow');
    await page.getByRole('button', { name: 'How it works', exact: true }).click();
    check(await page.locator('#helpDialog').isVisible(), 'Guide opens');
    await page.keyboard.press('Escape');
    check(!await page.locator('#helpDialog').isVisible(), 'Guide closes with Escape');
    await page.locator('#setupBtn').click();
    check(await page.locator('#vendorCategory').evaluate((element) => element === document.activeElement), 'Setup shortcut focuses the form');
    await page.locator('#latInput').fill('100');
    await page.locator('#getRecsBtn').click();
    check(calls === 0, 'Invalid coordinates do not call the API');
    await page.locator('#latInput').fill('16.5062');
    await page.locator('#getRecsBtn').click();
    await page.waitForSelector('.rec-card');
    check(requestBody.average_order_value === 100 && requestBody.search_radius_km === 3, 'The submitted business settings reach the API');
    check(await page.locator('#currentProfit').textContent() === '₹320', 'Current estimate renders');
    check(await page.locator('#bestProfit').textContent() === '₹800', 'Best nearby profit uses the maximum, not only the top-ranked area');
    check(await page.locator('#zonesCount').textContent() === '42', 'Evaluated zones render');
    check(await page.locator('.rec-card').count() === 3, 'All recommendation cards render');
    check(await page.locator('#decisionBanner').isVisible(), 'Decision guidance renders');
    check(await page.locator('#weatherBadge').textContent() === 'LIVE', 'Live weather is labeled');
    check((await page.locator('#freshnessTable').textContent()).includes('PROXY'), 'Proxy sources are not labeled live');
    check(!await page.locator('#exportBtn').isDisabled(), 'Export enabled after results');
    await page.locator('.rec-card').first().locator('summary').click();
    check(await page.locator('.rec-card').first().locator('.rec-details').getAttribute('open') !== null, 'Forecast details expand');
    check((await page.locator('.rec-card').first().textContent()).includes('18–32'), 'Prediction interval is preserved');
    check(await page.evaluate(() => !window.injected), 'Untrusted API text does not execute');
    await page.locator('#sortRecs').selectOption('profit');
    check(await page.locator('.rec-card').first().getAttribute('data-cell') === 'test-cell-2', 'Profit sorting works');
    await page.locator('#sortRecs').selectOption('distance');
    check(await page.locator('.rec-card').first().getAttribute('data-cell') === 'test-cell-3', 'Distance sorting works');
    await page.locator('.rec-card').first().locator('.rec-select').click();
    check(await page.locator('.rec-card').first().locator('.rec-select').getAttribute('aria-pressed') === 'true', 'Location selection updates its accessible state');
    if (await page.evaluate(() => !!window.L)) {
      await page.waitForFunction(() => state.routeRequest === null);
      check(await page.evaluate(() => state.route.getLayers()[0].getTooltip().getContent().includes('not navigation')), 'Interpolated backend geometry is not presented as a real road');
      await page.locator('#btnLayerSatellite').click();
      check(await page.locator('#btnLayerSatellite').getAttribute('aria-pressed') === 'true', 'Map style switch works');
      await page.locator('#btnLayerStreets').click();
      await page.locator('#heatmapToggle').click();
      check(await page.locator('#heatmapToggle').getAttribute('aria-pressed') === 'false', 'Demand zones can be hidden');
      await page.locator('#heatmapToggle').click();
      await page.locator('#expandMapBtn').click();
      check(await page.locator('#mapCard').evaluate((element) => element.classList.contains('expanded')), 'Map expands');
      await page.keyboard.press('Escape');
      check(!await page.locator('#mapCard').evaluate((element) => element.classList.contains('expanded')), 'Expanded map restores with Escape');
    }
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#exportBtn').click();
    const download = await downloadPromise;
    check(download.suggestedFilename().endsWith('.csv'), 'CSV download works');
    const csv = await fs.readFile(await download.path(), 'utf8');
    check(csv.includes('Est. profit INR/hr') && csv.includes('test-cell-3'), 'CSV contains labeled forecasts and data');
    await page.locator('#searchRadius').fill('4');
    check(await page.locator('#mapRadius').textContent() === '4 km radius', 'Radius preview updates');
    check(await page.locator('#currentProfit').textContent() === '—' && await page.locator('.rec-card').count() === 0, 'Input changes clear old forecasts and cards');
    check(!await page.locator('#decisionBanner').isVisible(), 'Input changes clear old decision guidance');
    check(await page.locator('#weatherBadge').textContent() === 'NOT CHECKED', 'Input changes clear old weather and provenance');
    mode = 'delayed';
    await page.locator('#getRecsBtn').click();
    await page.waitForFunction(() => document.querySelector('#recsList').getAttribute('aria-busy') === 'true');
    while (!delayedSeen) await new Promise((resolve) => setTimeout(resolve, 10));
    check(await page.locator('#getRecsBtn').isDisabled(), 'Loading prevents accidental repeated submission');
    await page.locator('#aov').fill('150');
    resolveDelayed();
    await page.waitForTimeout(100);
    check(await page.locator('.rec-card').count() === 0 && await page.locator('#currentProfit').textContent() === '—', 'A stale in-flight response cannot overwrite edited settings');
    mode = 'failure';
    await page.locator('#getRecsBtn').click();
    await page.waitForSelector('#errorMsg:not(.hidden)');
    check((await page.locator('#errorMsg').textContent()).includes('Demand model is unavailable'), 'Backend errors are actionable');
    check(!await page.locator('#getRecsBtn').isDisabled(), 'Failure returns the form to a usable state');
    mode = 'empty';
    await page.locator('#getRecsBtn').click();
    await page.waitForFunction(() => document.querySelector('#currentProfit').textContent === '₹320');
    check(await page.locator('.rec-card').count() === 0 && await page.locator('#recsPlaceholder').isVisible(), 'Zero results have a useful empty state');
    check(await page.locator('#decisionBanner').isVisible(), 'Stay guidance still appears with zero destinations');
    mode = 'success';
    await page.locator('#getRecsBtn').click();
    await page.waitForSelector('.rec-card');
    if (process.env.SCREENSHOT_DIR) {
      await fs.mkdir(process.env.SCREENSHOT_DIR, { recursive: true });
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: path.join(process.env.SCREENSHOT_DIR, 'dashboard-desktop.png'), fullPage: true });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => scrollTo(0, 0));
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile has no horizontal overflow');
    await page.locator('#menuBtn').click();
    check(await page.locator('#menuBtn').getAttribute('aria-expanded') === 'true', 'Mobile navigation opens');
    await page.locator('.nav-link[href="#recommendations"]').click();
    check(await page.locator('#menuBtn').getAttribute('aria-expanded') === 'false', 'Choosing a mobile section closes navigation');
    check(await page.locator('#getRecsBtn').isVisible(), 'Mobile form remains available');
    if (process.env.SCREENSHOT_DIR) {
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: path.join(process.env.SCREENSHOT_DIR, 'dashboard-mobile.png'), fullPage: true });
    }
    // A separate page exercises CDN failure rather than mocking Leaflet itself.
    const offlineMapPage = await context.newPage();
    offlineMapPage.on('pageerror', (error) => errors.push(error.message));
    await offlineMapPage.route('**/unpkg.com/**', (route) => route.abort());
    await offlineMapPage.route('**/v1/system/data-health', (route) => route.fulfill({ status: 503, json: {} }));
    await offlineMapPage.route('**/v1/recommendations/live', (route) => route.fulfill({ json: fixtures }));
    await offlineMapPage.goto(baseURL, { waitUntil: 'networkidle' });
    check(await offlineMapPage.locator('#mapUnavailable').isVisible(), 'Map CDN failure is explained');
    check(await offlineMapPage.locator('#statusText').textContent() === 'API offline', 'Non-success health responses are not shown as ready');
    await offlineMapPage.locator('#getRecsBtn').click();
    await offlineMapPage.waitForSelector('.rec-card');
    check(await offlineMapPage.locator('.rec-card').count() === 3, 'Analysis remains usable without the map');
    check(errors.length === 0, `No browser JavaScript errors: ${errors.join('; ')}`);
    console.log(`PASS: ${checks} browser checks (API fixtures, real page/assets, desktop + mobile + unavailable-map states).`);
  } finally {
    resolveDelayed?.();
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
