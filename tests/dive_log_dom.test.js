/* Behavioural tests for the dive log STORE LAYER, in a real DOM.
 *
 * Run: node tests/dive_log_dom.test.js
 *
 * The load-bearing rule guarded here: storage holds the POLYLINE ONLY. Derived
 * scores are re-computed at render, so refitting the model re-scores saved dives
 * instead of leaving them stale. (Render/table tests arrive with the render task.)
 */
const {JSDOM} = require(process.env.CLAUDE_JOB_DIR + '/tmp/node_modules/jsdom');
const fs = require('fs');
const path = require('path');

const APP = path.join(__dirname, '..', 'decostress_app', 'index.html');

function boot(storage) {
  const ENGINE = fs.readFileSync(path.join(__dirname, '..', 'decostress_app', 'deco-engine.js'), 'utf8');
  const html = fs.readFileSync(APP, 'utf8')
    .replace(/<script src="https:\/\/cdnjs[^>]*><\/script>/, '<script></script>')
    .replace('<script src="deco-engine.js"></script>', '<script>' + ENGINE + '</script>');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'http://localhost/',
    beforeParse(w) {
      w.HTMLCanvasElement.prototype.getContext = () =>
        new Proxy({}, {get: () => () => ({addColorStop() {}, width: 10})});
      if (storage !== undefined) {
        w.localStorage.setItem('decostress.scenarios.v1', storage);
      }
    },
  });
  return dom.window;
}

// 30 m for 20 min with a 3-min ascent.
const SQUARE = [{t: 0, d: 0}, {t: 1.7, d: 30}, {t: 21.7, d: 30}, {t: 24.7, d: 0}];

let fails = 0;
const ok = (n, c, x) => {
  console.log((c ? '  PASS  ' : '  FAIL  ') + n + (x !== undefined ? '  [' + x + ']' : ''));
  if (!c) fails++;
};

setTimeout(() => {
  console.log('--- store ---');
  const w = boot();
  ok('log starts empty', w.logLoad().length === 0);
  const s = w.logAdd('test dive', SQUARE);
  ok('logAdd returns a scenario with an id', !!s.id);
  ok('logLoad sees it', w.logLoad().length === 1);
  ok('name persisted', w.logLoad()[0].name === 'test dive');

  console.log('--- the drift rule: inputs only ---');
  const raw = JSON.parse(w.localStorage.getItem('decostress.scenarios.v1'));
  const keys = Object.keys(raw[0]).sort();
  ok('persists ONLY {id,name,poly,savedAt}',
     JSON.stringify(keys) === JSON.stringify(['id', 'name', 'poly', 'savedAt']), keys.join(','));
  for (const derived of ['pct', 'score', 'requiredAscent', 'deficit', 'stress', 'violated']) {
    ok('does NOT persist derived field: ' + derived, !(derived in raw[0]));
  }

  w.logRemove(s.id);
  ok('logRemove works', w.logLoad().length === 0);
  ok('defaultName from profile', /30\s*m/.test(w.defaultName(SQUARE)), w.defaultName(SQUARE));

  console.log('--- edge cases ---');
  const w4 = boot('{{{not json');
  ok('corrupt storage yields an empty log (no crash)', w4.logLoad().length === 0);
  const w4b = boot('{"not":"an array"}');
  ok('non-array storage yields an empty log', w4b.logLoad().length === 0);

  console.log('--- export / import ---');
  const w5 = boot();
  w5.logAdd('a', SQUARE);
  const json = w5.logExport();
  const parsed = JSON.parse(json);
  ok('export contains only input fields',
     JSON.stringify(Object.keys(parsed[0]).sort()) === JSON.stringify(['id', 'name', 'poly', 'savedAt']));

  const w6 = boot();
  const res = w6.logImport(json);
  ok('import round-trips', res.added === 1 && w6.logLoad()[0].name === 'a');
  const again = w6.logImport(json);
  ok('re-importing skips duplicates', again.added === 0 && again.skipped === 1, JSON.stringify(again));

  let threw = false;
  try { w6.logImport('not json at all'); } catch (e) { threw = true; }
  ok('importing garbage throws a clean error', threw);
  ok('  ...and leaves the existing log intact', w6.logLoad().length === 1);

  // A crafted file with a NaN depth must be rejected, not coerced — NaN makes the
  // void guards fail open, so a dangerous dive could import as unflagged.
  const w7 = boot();
  const bad = JSON.stringify([{id: 'x', name: 'nan', savedAt: 1,
    poly: [{t: 0, d: 0}, {t: 5, d: 'deep'}, {t: 10, d: 0}]}]);
  const rb = w7.logImport(bad);
  ok('import rejects non-finite waypoints', rb.added === 0 && rb.skipped === 1, JSON.stringify(rb));

  // A bent / out-of-distribution dive: over the ceiling, unrankable by the model.
  const BENT = [{t:0,d:0},{t:3,d:45},{t:10,d:45},{t:11,d:33},{t:12,d:38},{t:13,d:30},
                {t:15,d:32},{t:17,d:22},{t:18,d:26},{t:20,d:12},{t:21,d:18},{t:23.6,d:0}];

  console.log('--- render: table ---');
  const w2 = boot();
  w2.logAdd('safe dive', SQUARE);
  w2.logAdd('bent dive', BENT);
  w2.renderLog();
  const rows = [...w2.document.querySelectorAll('#logTbody tr')];
  ok('table renders one row per scenario', rows.length === 2, rows.length);

  const cells = r => [...r.children].map(c => c.textContent.trim());
  const bent = rows.find(r => cells(r)[0] === 'bent dive');
  const safe = rows.find(r => cells(r)[0] === 'safe dive');
  ok('bent dive flagged (void row class)', bent.classList.contains('void'));
  ok('safe dive NOT flagged', !safe.classList.contains('void'));

  console.log('--- the honesty rule in the table ---');
  const bentRank = cells(bent)[5];
  const safeRank = cells(safe)[5];
  ok('bent/unrankable dive shows VOID', bentRank === 'VOID', bentRank);
  ok('voided rank contains NO digits', !/\d/.test(bentRank), bentRank);
  ok('safe dive DOES show a numeric rank', /\d/.test(safeRank), safeRank);

  w2.sortLog('rank');
  const order = [...w2.document.querySelectorAll('#logTbody tr')].map(r => r.classList.contains('void'));
  const firstVoid = order.indexOf(true);
  ok('sorting by rank groups voided dives at the end',
     firstVoid === -1 || order.slice(firstVoid).every(v => v === true), JSON.stringify(order));

  console.log('--- drivers ---');
  const drv = w2.document.getElementById('logDrivers').textContent;
  ok('driver breakdown mentions the obligation', /obligation/i.test(drv));
  ok('voided dive gets a refusal, not contributions', /no rank|does not reduce|over your ceiling|outside that world/i.test(drv));

  console.log('--- empty state ---');
  const w3 = boot();
  w3.renderLog();
  ok('empty log shows the empty state', !w3.document.getElementById('logEmpty').hidden);
  ok('empty log hides the table block', w3.document.getElementById('logHas').hidden);

  console.log(fails ? '\nFAILURES: ' + fails : '\nALL PASSED');
  process.exit(fails ? 1 : 0);
}, 900);
