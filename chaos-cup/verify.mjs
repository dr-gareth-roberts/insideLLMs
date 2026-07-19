/* ============================================================================
   CHAOS CUP — verification harness (dev-only; uses playwright-core + the
   pre-installed Chromium at /opt/pw-browsers/chromium).

   What it does, headlessly:
     1. Loads index.html and fails on ANY console error / uncaught exception.
     2. Screenshots every scene: title, character select, gameplay, goal
        celebration, and the result screen  ->  chaos-cup/shots/*.png
     3. Drives a real user match via the window.CHAOS debug API.
     4. Measures sustained FPS and asserts a 60fps-class budget.
     5. Prints a JSON report and exits non-zero if any gate fails.

   The LOOP reads the screenshots back (they are the visual ground truth) and
   the JSON report (the mechanical ground truth). Run:  npm run verify
   Flags:  --url <file-or-http>  --shots <dir>  --keep-open  --headed
   ========================================================================== */
import { chromium } from 'playwright-core';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const getFlag = (n, d) => { const i = argv.indexOf(n); return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : d; };
const has = (n) => argv.includes(n);

const EXE = process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium';
const SHOTS = resolve(getFlag('--shots', join(__dirname, 'shots')));
const URL = getFlag('--url', pathToFileURL(join(__dirname, 'index.html')).href);
const HEADED = has('--headed');

mkdirSync(SHOTS, { recursive: true });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const report = { url: URL, ok: false, gates: {}, fps: null, errors: [], shots: [], notes: [] };
const fail = (g, msg) => { report.gates[g] = { pass: false, msg }; report.notes.push(`FAIL ${g}: ${msg}`); };
const pass = (g, msg = '') => { report.gates[g] = { pass: true, msg }; };

let browser;
try {
  browser = await chromium.launch({
    executablePath: EXE, headless: !HEADED,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--hide-scrollbars', '--mute-audio', '--force-color-profile=srgb'],
  });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();

  const consoleErrors = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push('PAGEERROR: ' + e.message));

  await page.goto(URL, { waitUntil: 'load', timeout: 20000 });
  // Wait for the game API to come online (proves the script parsed & ran).
  await page.waitForFunction('window.CHAOS && window.CHAOS.version', null, { timeout: 8000 })
    .then(() => pass('boot', 'window.CHAOS online'))
    .catch(() => fail('boot', 'window.CHAOS never appeared (script crashed on load?)'));

  const shot = async (name) => { const p = join(SHOTS, name); await page.screenshot({ path: p }); report.shots.push(p); };

  // ---- Title ---------------------------------------------------------------
  await sleep(900);
  await shot('01-title.png');

  // ---- Character select ----------------------------------------------------
  await page.evaluate(() => window.CHAOS.skipTo('select'));
  await sleep(500);
  await shot('02-select.png');

  // ---- Gameplay (real user match) -----------------------------------------
  await page.evaluate(() => window.CHAOS.startUserMatch('BOLT'));
  await sleep(3200); // let the countdown finish and play begin
  await page.evaluate(() => { for (let i = 0; i < 3; i++) window.CHAOS.spawnItem(); });
  await sleep(1600);
  const inPlay = await page.evaluate(() => window.CHAOS.match && ['play', 'countdown'].includes(window.CHAOS.match.state));
  inPlay ? pass('gameplay', 'match is live') : fail('gameplay', 'match did not reach play state');
  await shot('03-gameplay.png');

  // ---- Goal celebration ----------------------------------------------------
  await page.evaluate(() => window.CHAOS.forceGoal(0));
  await sleep(700);
  await shot('04-goal.png');
  const scored = await page.evaluate(() => window.CHAOS.match && window.CHAOS.match.teams[0].score >= 1);
  scored ? pass('goal', 'goal registered + celebration') : fail('goal', 'score did not increment');

  // ---- FPS budget (sample during live play) --------------------------------
  await sleep(1600); // return to play after celebration
  await page.evaluate(() => { window.CHAOS.metrics._f0 = window.CHAOS.metrics.frames; });
  await sleep(2000);
  const fps = await page.evaluate(() => window.CHAOS.metrics.fps);
  report.fps = Math.round(fps * 10) / 10;
  (fps >= 45) ? pass('fps', `${report.fps} fps`) : fail('fps', `only ${report.fps} fps (<45)`);

  // ---- Result screen -------------------------------------------------------
  await page.evaluate(() => { const m = window.CHAOS.match; if (m) { m.time = 0.1; } });
  await sleep(900);
  await shot('05-result.png');

  // ---- Console cleanliness -------------------------------------------------
  report.errors = consoleErrors;
  consoleErrors.length === 0 ? pass('console', 'no console errors')
    : fail('console', `${consoleErrors.length} console error(s): ${consoleErrors.slice(0, 3).join(' | ')}`);

  if (HEADED || has('--keep-open')) { report.notes.push('holding browser open 30s'); await sleep(30000); }
} catch (e) {
  fail('harness', e.message);
  report.errors.push('HARNESS: ' + (e.stack || e.message));
} finally {
  if (browser) await browser.close().catch(() => {});
}

report.ok = Object.values(report.gates).every(g => g.pass);
writeFileSync(join(SHOTS, 'report.json'), JSON.stringify(report, null, 2));

// Human summary
const line = (s) => process.stdout.write(s + '\n');
line('\n================ CHAOS CUP — verify ================');
for (const [g, v] of Object.entries(report.gates)) line(`  ${v.pass ? '✅' : '❌'} ${g.padEnd(9)} ${v.msg || ''}`);
line(`  fps: ${report.fps}   shots: ${report.shots.length} -> ${SHOTS}`);
if (report.errors.length) { line('  errors:'); report.errors.slice(0, 6).forEach(e => line('    - ' + e)); }
line(`  RESULT: ${report.ok ? '✅ PASS' : '❌ FAIL'}`);
line('====================================================\n');
process.exit(report.ok ? 0 : 1);
