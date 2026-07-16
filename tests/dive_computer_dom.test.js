/* The dive-computer page, in a real DOM.
 *
 *   node tests/dive_computer_dom.test.js
 *
 * The page holds no physics — it reads the shared engine and draws. These checks
 * confirm it boots, renders the watch from the engine, shows 16 tissue bars, and
 * that the DecoStress rank obeys voidVerdict (VOID on a bent profile), the same
 * discipline as everywhere else.
 */
const {JSDOM} = require(process.env.CLAUDE_JOB_DIR + '/tmp/node_modules/jsdom');
const fs = require('fs'), path = require('path');
const PAGE = path.join(__dirname, '..', 'decostress_app', 'divecomputer.html');
const ENGINE = path.join(__dirname, '..', 'decostress_app', 'deco-engine.js');

function boot() {
  const eng = fs.readFileSync(ENGINE, 'utf8');
  const html = fs.readFileSync(PAGE, 'utf8')
    .replace('<script src="deco-engine.js"></script>', '<script>' + eng + '</script>');
  const errs = [];
  const w = new JSDOM(html, {runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/',
    beforeParse(w) {
      w.HTMLCanvasElement.prototype.getContext = () =>
        new Proxy({}, {get: () => () => ({addColorStop() {}, setLineDash() {}, width: 10})});
      w.onerror = m => errs.push(String(m)); w.addEventListener('error', e => errs.push(e.message));
    }}).window;
  w.__errs = errs;
  return w;
}

let fails = 0;
const ok = (n, c, x) => { console.log((c ? '  PASS  ' : '  FAIL  ') + n + (x !== undefined ? '  [' + x + ']' : '')); if (!c) fails++; };

setTimeout(() => {
  const w = boot(), d = w.document;
  ok('page boots with zero JS errors', w.__errs.length === 0, w.__errs[0] || '');
  ok('engine is loaded', typeof w.predict === 'function');
  ok('demo dive exists', Array.isArray(w.DEMO) && w.DEMO.length >= 4);

  // at the bottom of the demo dive, depth is positive and NDL or TTS is shown
  const bottomT = w.DEMO[2].t;   // end of bottom phase
  w.renderAt(bottomT);
  const depth = d.getElementById('wDepth').textContent;
  ok('watch shows a positive depth at the bottom', parseFloat(depth) > 5, depth);
  const ndl = d.getElementById('wNdl').textContent;
  ok('watch shows an NDL or TTS value', /\d/.test(ndl) || /∞/.test(ndl), ndl);

  // the 16 tissue bars render
  ok('tissue strip has 16 bars', d.querySelectorAll('#wTissues .tbar').length === 16,
     d.querySelectorAll('#wTissues .tbar').length);

  // mid-dive (submerged), the rank is withheld — you cannot rank a dive not yet over
  ok('rank is "—" while still submerged', d.getElementById('wRank').textContent.trim() === '—',
     d.getElementById('wRank').textContent);

  // once SURFACED, the completed in-distribution demo dive shows a real rank
  const surfacedT = w.DEMO[w.DEMO.length - 1].t + 2;   // 2 min into the surface watch
  w.renderAt(surfacedT);
  ok('rank shows a number once surfaced (in-distribution demo dive)',
     /\d/.test(d.getElementById('wRank').textContent), d.getElementById('wRank').textContent);

  // a bent profile: deep, then straight to surface (over ceiling) -> VOID
  const BENT = [{t:0,d:0},{t:2.5,d:45},{t:22,d:45},{t:23.2,d:0}];
  w.renderProfileAt(BENT, BENT[3].t);
  ok('rank reads VOID on a bent profile', /VOID/i.test(d.getElementById('wRank').textContent),
     d.getElementById('wRank').textContent);

  console.log(fails ? '\nFAILURES: ' + fails : '\nALL PASSED');
  process.exit(fails ? 1 : 0);
}, 700);
