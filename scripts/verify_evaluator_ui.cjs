/* Reserved browser checks: local URLs only; evidence is never written to public results. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'results/evaluator');
const inspection = JSON.parse(fs.readFileSync(path.join(output, 'inspection.json')));
const canonical = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const banned = ['reference_arc_id', 'alignment_status', 'reference_route', 'reference_geometry', 'ground_truth', 'route_position'];
function checkLeaks(value) {
  if (Array.isArray(value)) value.forEach(checkLeaks);
  else if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) {
    assert(!banned.includes(key), 'Reference field exposed: ' + key); checkLeaks(child);
  }
}
(async () => {
  const ui = process.env.PATHSNAP_UI_URL || 'http://127.0.0.1:8001';
  const api = process.env.PATHSNAP_API_URL || 'http://127.0.0.1:8000';
  assert(['localhost', '127.0.0.1'].includes(new URL(ui).hostname));
  assert(['localhost', '127.0.0.1'].includes(new URL(api).hostname));
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PATHSNAP_BROWSER_PATH || chromium.executablePath() });
  const errors = [], external = [], rows = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
    page.on('pageerror', error => errors.push(String(error)));
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (!['localhost', '127.0.0.1'].includes(url.hostname)) { external.push(url.origin); return route.abort(); }
      return route.continue();
    });
    await page.goto(ui + '/?api=' + encodeURIComponent(api));
    await page.waitForFunction(() => !document.querySelector('#run').disabled);
    assert.equal(await page.locator('#mode').getAttribute('data-mode'), 'reserved-evaluator');
    assert.deepEqual(await page.locator('#trip option').evaluateAll(options => options.map(o => o.value)), inspection.trip_ids);
    assert(await page.locator('#street-view').isDisabled());
    fs.mkdirSync(path.join(output, 'ui'), { recursive: true });
    for (const tid of inspection.trip_ids) {
      const tick = Date.now(); const row = { trip_id: tid };
      try {
        await page.selectOption('#trip', tid);
        const responsePromise = page.waitForResponse(response => response.url().endsWith('/predict') && response.request().method() === 'POST', { timeout: 600000 });
        await page.click('#run');
        const response = await responsePromise;
        row.http_status = response.status();
        const data = await response.json();
        checkLeaks(data); assert.equal(data.trip_id, tid); assert.equal(data.mode, 'reserved-evaluator');
        await page.waitForFunction(() => document.querySelector('#status').textContent === 'Reconstruction complete', null, { timeout: 600000 });
        assert((await page.locator('#caption').textContent()).includes(tid));
        assert.equal(await page.locator('#network-view').getAttribute('aria-pressed'), 'true');
        const saved = JSON.parse(fs.readFileSync(path.join(output, 'predictions', tid + '.json')));
        assert.equal(canonical(data.predictions), canonical(saved.predictions), 'Repeated UI inference changed predicted IDs');
        assert.equal(canonical(data.route_runs), canonical(saved.route_runs), 'Repeated UI inference changed route runs');
        assert.equal(await page.locator('#count').textContent(), data.observation_count.toLocaleString('en-US'));
        if (!data.metrics) {
          assert.equal(await page.locator('#hmm').textContent(), 'N/A');
          assert((await page.locator('#detail').textContent()).toLowerCase().includes('accuracy'));
        }
        await page.screenshot({ path: path.join(output, 'ui', tid + '.png'), fullPage: true });
        const downloadPromise = page.waitForEvent('download'); await page.click('#download');
        const download = await downloadPromise;
        const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
        checkLeaks(exported); assert.equal(exported.trip_id, tid);
        assert.equal(canonical(exported.predictions), canonical(saved.predictions));
        row.passed = true; row.prediction_count = data.predictions.length; row.deterministic_predictions = true; row.reference_leak_check = 'passed';
      } catch (error) { row.passed = false; row.error = String(error); }
      row.runtime_s = (Date.now() - tick) / 1000;
      rows.push(row); console.log(JSON.stringify(row));
      fs.writeFileSync(path.join(output, 'ui_results.partial.json'), JSON.stringify(rows, null, 2));
    }
    const report = { trips: rows, trip_count: inspection.trip_count, javascript_errors: errors,
      blocked_external_requests: external, privacy: 'Only localhost requests allowed; external maps disabled',
      passed: rows.every(r => r.passed) && !errors.length && !external.length,
      scope: 'Actual local browser/UI/API; second physical laptop/LAN not tested' };
    fs.writeFileSync(path.join(output, 'ui_verification.json'), JSON.stringify(report, null, 2));
    assert(report.passed, 'Reserved UI verification failed; see private evidence');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
