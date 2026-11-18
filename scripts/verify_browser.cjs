/* Optional browser QA: npm install && node scripts/verify_browser.cjs */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');

(async () => {
  const out = path.resolve(__dirname, '../results/final');
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PATHSNAP_BROWSER_PATH || chromium.executablePath(),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
    const errors = [];
    let streetTileResponses = 0;
    page.on('pageerror', error => errors.push(String(error)));
    page.on('response', response => { if (response.url().startsWith('https://tile.openstreetmap.org/') && response.ok()) streetTileResponses++; });
    await page.goto(process.env.PATHSNAP_UI_URL || 'http://127.0.0.1:8001');
    await page.waitForFunction(() => !document.querySelector('#run').disabled);
    const uiUrl = page.url();
    const apiUrl = await page.inputValue('#api');
    await page.screenshot({ path: path.join(out, 'ui_empty.png'), fullPage: true });

    async function reconstruct(trip) {
      await page.selectOption('#trip', trip);
      const response = page.waitForResponse(r => r.url().endsWith('/predict') && r.request().method() === 'POST');
      await page.click('#run');
      const data = await (await response).json();
      await page.waitForFunction(
        () => document.querySelector('#status').textContent === 'Reconstruction complete',
        null, { timeout: 180000 },
      );
      assert.equal(data.trip_id, trip);
      return data;
    }
    async function verifyMapFit() {
      const fit = await page.evaluate(() => {
        const svg = document.querySelector('#map');
        const rect = svg.getBoundingClientRect();
        const box = svg.viewBox.baseVal;
        const points = document.querySelector('#layer-gps').getBBox();
        const route = document.querySelector('#layer-decoded_segments').getBBox();
        const left = Math.min(points.x, route.x), right = Math.max(points.x + points.width, route.x + route.width);
        const top = Math.min(points.y, route.y), bottom = Math.max(points.y + points.height, route.y + route.height);
        return {
          aspectError: Math.abs(box.width / box.height - rect.width / rect.height),
          centerErrorX: Math.abs((left + right) / 2 - box.x - box.width / 2) / box.width,
          centerErrorY: Math.abs((top + bottom) / 2 - box.y - box.height / 2) / box.height,
        };
      });
      assert(fit.aspectError < .001, 'Map must fit its actual viewport aspect ratio');
      assert(fit.centerErrorX < .01 && fit.centerErrorY < .01, 'GPS and decoded route must be centered');
      return fit;
    }
    async function verifyStreetMap() {
      await page.waitForFunction(() => document.querySelector('#street-view').getAttribute('aria-pressed') === 'true' &&
        !streetTiles.isLoading() && [...document.querySelectorAll('.leaflet-tile')].some(image => image.naturalWidth > 0), null, { timeout: 30000 });
      assert(await page.locator('.leaflet-control-attribution').isVisible(), 'Map attribution must be visible');
      assert((await page.locator('.leaflet-control-attribution').textContent()).includes('OpenStreetMap'));
      const fit = await page.evaluate(() => {
        const center = streetMap.latLngToContainerPoint(streetBounds.getCenter()), size = streetMap.getSize();
        return { boundsVisible: streetMap.getBounds().contains(streetBounds), centerErrorX: Math.abs(center.x / size.x - .5), centerErrorY: Math.abs(center.y / size.y - .5) };
      });
      assert(fit.boundsVisible && fit.centerErrorX < .01 && fit.centerErrorY < .01, 'Street map must contain and center the route');
      return fit;
    }
    const data = await reconstruct('00000000');
    const desktopStreetFit = await verifyStreetMap();
    await page.screenshot({ path: path.join(out, 'ui_desktop.png'), fullPage: true });
    const streetZoom = await page.evaluate(() => streetMap.getZoom());
    await page.click('#zoom-in');
    assert((await page.evaluate(() => streetMap.getZoom())) > streetZoom);
    await page.click('#reset');
    assert.equal(await page.evaluate(() => streetMap.getZoom()), streetZoom);
    await page.click('#network-view');
    const desktopFit = await verifyMapFit();
    await page.screenshot({ path: path.join(out, 'ui_network.png'), fullPage: true });
    const initialBox = await page.locator('#map').getAttribute('viewBox');
    await page.click('#zoom-in');
    assert.notEqual(await page.locator('#map').getAttribute('viewBox'), initialBox);
    await page.click('#reset');
    assert.equal(await page.locator('#map').getAttribute('viewBox'), initialBox);
    await page.locator('#map').focus();
    await page.keyboard.press('ArrowRight');
    assert.notEqual(await page.locator('#map').getAttribute('viewBox'), initialBox);
    await page.keyboard.press('0');
    assert.equal(await page.locator('#map').getAttribute('viewBox'), initialBox);
    await page.locator('label.layer').filter({ has: page.locator('[data-layer="gps"]') }).click();
    assert.equal(await page.locator('#layer-gps').evaluate(node => node.style.display), 'none');
    await page.locator('label.layer').filter({ has: page.locator('[data-layer="gps"]') }).click();
    assert.notEqual(await page.locator('#layer-gps').evaluate(node => node.style.display), 'none');
    const downloadPromise = page.waitForEvent('download');
    await page.click('#download');
    const download = await downloadPromise;
    assert.equal(download.suggestedFilename(), 'pathsnap-00000000.json');
    const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal(exported.predictions.length, data.observation_count);
    assert(!JSON.stringify(exported).includes('reference_arc_id'));
    await page.click('#street-view');
    await verifyStreetMap();
    await page.locator('#evaluation > summary').click();
    await page.screenshot({ path: path.join(out, 'api_response.png'), fullPage: true });
    await page.locator('#evaluation > summary').click();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(150);
    const mobileStreetFit = await verifyStreetMap();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile page must not overflow horizontally');
    await page.screenshot({ path: path.join(out, 'ui_mobile.png'), fullPage: true });
    await page.click('#network-view');
    const mobileFit = await verifyMapFit();
    await page.click('#street-view');
    await page.setViewportSize({ width: 1440, height: 1000 });
    await reconstruct('00000091');
    const narrowStreetFit = await verifyStreetMap();
    await page.screenshot({ path: path.join(out, 'ui_route_focus.png'), fullPage: true });
    await page.click('#network-view');
    const narrowRouteFit = await verifyMapFit();
    await page.click('#street-view');
    await verifyStreetMap();

    // Simulate unavailable internet tiles; the local network and route must remain usable.
    await page.route('https://tile.openstreetmap.org/**', route => route.abort());
    await page.evaluate(() => { streetTiles.redraw(); });
    await page.waitForFunction(() => document.querySelector('#network-view').getAttribute('aria-pressed') === 'true');
    assert(await page.locator('#map-message').isVisible());
    await verifyMapFit();
    await page.unroute('https://tile.openstreetmap.org/**');
    await page.click('#street-view');
    await verifyStreetMap();

    // Error and missing-reference cases are explicit browser fixtures, not measured inference results.
    await page.route('**/trips', route => route.fulfill({ status: 503, body: 'Unavailable' }));
    await page.locator('#connection-settings > summary').click();
    await page.click('#connect');
    await page.waitForFunction(() => document.querySelector('#status').classList.contains('error'));
    assert(await page.locator('#run').isDisabled());
    assert(await page.locator('#download').isDisabled());
    assert.equal(await page.locator('#hmm').textContent(), '—');
    await page.unroute('**/trips');
    await page.click('#connect');
    await page.waitForFunction(() => !document.querySelector('#run').disabled);
    await page.route('**/predict', route => route.fulfill({ json: { ...data, metrics: null } }));
    await reconstruct('00000000');
    assert.equal(await page.locator('#classifier').textContent(), 'N/A');
    assert.equal(await page.locator('#hmm').textContent(), 'N/A');
    assert((await page.locator('#detail').textContent()).includes('without accuracy scores'));
    await page.unroute('**/predict');
    // Privacy behavior is checked on public development geometry before reserved evaluation.
    await page.route('**/trips', route => route.fulfill({ json: { mode: 'reserved-evaluator', trips: ['00000000'], trip_count: 1 } }));
    await page.route('**/predict', route => route.fulfill({ json: { ...data, mode: 'reserved-evaluator' } }));
    await page.locator('#connection-settings > summary').click();
    await page.click('#connect');
    await page.waitForFunction(() => !document.querySelector('#run').disabled);
    const beforePrivate = streetTileResponses;
    await reconstruct('00000000');
    assert(await page.locator('#street-view').isDisabled());
    assert.equal(await page.locator('#network-view').getAttribute('aria-pressed'), 'true');
    await page.waitForTimeout(500);
    assert.equal(streetTileResponses, beforePrivate, 'Reserved mode must not request external tiles');
    await page.goto('file://' + path.join(out, 'final_folium_map.html'));
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(out, 'folium_map.png'), fullPage: true });
    fs.writeFileSync(path.join(out, 'browser_verification.json'), JSON.stringify({
      javascript_errors: errors, desktop: [1440, 1000], mobile: [390, 844],
      trip: '00000000', additional_live_trip: '00000091', ui_url: uiUrl, api_url: apiUrl,
      ui_request: 'real POST /predict',
      map_fit: { desktop: desktopFit, mobile: mobileFit, narrow_route: narrowRouteFit },
      street_map: { provider: 'OpenStreetMap', successful_tile_responses: streetTileResponses, desktop: desktopStreetFit, mobile: mobileStreetFit, narrow_route: narrowStreetFit, attribution: 'visible', offline_tile_failure_fixture: 'passed' },
      reserved_privacy_development_fixture: 'external tiles disabled; passed',
      interaction_checks: ['street and network views', 'street zoom and fit', 'network zoom and fit', 'keyboard pan and reset', 'layer visibility', 'prediction JSON export', 'mobile width', 'tile failure and recovery fixture', 'connection error and recovery fixture', 'missing-reference fixture'],
      scope: 'local browser client and separate UI/API processes; physical LAN laptop not accessed',
    }, null, 2));
    if (errors.length) throw Error(errors.join('\n'));
    console.log('Street tiles, offline fallback, two live predictions, centered maps, controls, export, mobile and error-state checks passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
