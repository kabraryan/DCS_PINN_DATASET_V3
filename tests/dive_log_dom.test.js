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
  const html = fs.readFileSync(APP, 'utf8')
    .replace(/<script src="https:\/\/cdnjs[^>]*><\/script>/, '<script></script>');
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

  console.log(fails ? '\nFAILURES: ' + fails : '\nALL PASSED');
  process.exit(fails ? 1 : 0);
}, 700);
