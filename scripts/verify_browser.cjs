/* Optional browser QA: npm install && node scripts/verify_browser.cjs */
const { chromium } = require('playwright');
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
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(process.env.PATHSNAP_UI_URL || 'http://127.0.0.1:8001');
    await page.waitForFunction(() => !document.querySelector('#run').disabled);
    const uiUrl = page.url();
    const apiUrl = await page.inputValue('#api');
    await page.selectOption('#trip', '00000000');
    await page.click('#run');
    // Playwright's second argument is the page-function argument, not options.
    await page.waitForFunction(
      () => document.querySelector('#status').textContent === 'Reconstruction complete',
      null, { timeout: 180000 },
    );
    await page.screenshot({ path: path.join(out, 'ui_desktop.png'), fullPage: true });
    await page.locator('details').click();
    await page.screenshot({ path: path.join(out, 'api_response.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(out, 'ui_mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto('file://' + path.join(out, 'final_folium_map.html'));
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(out, 'folium_map.png'), fullPage: true });
    fs.writeFileSync(path.join(out, 'browser_verification.json'), JSON.stringify({
      javascript_errors: errors, desktop: [1440, 1000], mobile: [390, 844],
      trip: '00000000', ui_url: uiUrl, api_url: apiUrl,
      ui_request: 'real POST /predict',
      scope: 'local browser client and separate UI/API processes; physical LAN laptop not accessed',
    }, null, 2));
    if (errors.length) throw Error(errors.join('\n'));
    console.log('Browser, live prediction, desktop/mobile and Folium screenshots passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
