/* The council's central finding: the reported "MILDER THAN MOST NAVY DIVES" bug
 * lived in showAssessment(), NOT the dial — and clicking Scrub reverted the dial
 * to the reassuring number. A predict()-only fix is not enough. The rank must be
 * suppressed on ALL FOUR surfaces: dial, assessment headline, scrub, log table.
 *
 *   node tests/audit_surfaces.test.js
 */
const {JSDOM} = require(process.env.CLAUDE_JOB_DIR + '/tmp/node_modules/jsdom');
const fs = require('fs');
const path = require('path');
const APP = path.join(__dirname, '..', 'decostress_app', 'index.html');

function boot() {
  const ENGINE = fs.readFileSync(path.join(__dirname, '..', 'decostress_app', 'deco-engine.js'), 'utf8');
  const html = fs.readFileSync(APP, 'utf8')
    .replace(/<script src="https:\/\/cdnjs[^>]*><\/script>/, '<script></script>')
    .replace('<script src="deco-engine.js"></script>', '<script>' + ENGINE + '</script>');
  return new JSDOM(html, {
    runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/',
    beforeParse(w) {
      w.HTMLCanvasElement.prototype.getContext = () =>
        new Proxy({}, {get: () => () => ({addColorStop() {}, width: 10})});
    },
  }).window;
}

// The reported bug: 45 m, wandering sawtooth ascent, surfaces over the limit.
const BENT = [{t:0,d:0},{t:3,d:45},{t:10,d:45},{t:11,d:33},{t:12,d:38},{t:13,d:30},
              {t:15,d:32},{t:17,d:22},{t:18,d:26},{t:20,d:12},{t:21,d:18},{t:23.6,d:0}];
const SAFE = [{t:0,d:0},{t:1.5,d:18},{t:40,d:18},{t:43,d:5},{t:46,d:5},{t:47,d:0}];

let fails = 0;
const ok = (n, c, x) => {
  console.log((c ? '  PASS  ' : '  FAIL  ') + n + (x !== undefined ? '  [' + x + ']' : ''));
  if (!c) fails++;
};
const hasRank = s => /\bth\b|\bst\b|\bnd\b|\brd\b|\d+(st|nd|rd|th)/.test(s);

setTimeout(() => {
  const w = boot(), d = w.document;
  const render = () => { w.refreshValues(); w.smooth(1); };

  console.log('--- SAFE dive: rank shown on every surface ---');
  w.loadProfile(SAFE); render();
  ok('dial shows a rank', /\d+(st|nd|rd|th)/.test(d.getElementById('pct').textContent), d.getElementById('pct').textContent.trim());
  ok('assessment headline shows a rank', /\d+(st|nd|rd|th)/.test(d.getElementById('aBig').textContent), d.getElementById('aBig').textContent.trim());
  ok('assessment band is a cohort band', /NAVY DIVES/i.test(d.getElementById('aBand').textContent), d.getElementById('aBand').textContent.trim());

  console.log('--- BENT dive: rank WITHHELD on every surface ---');
  w.loadProfile(BENT); render();

  // 1. the dial
  const dial = d.getElementById('pct').textContent.trim();
  const dialBand = d.getElementById('band').textContent.trim();
  ok('dial does NOT show a reassuring rank', !/\d+(st|nd|rd|th)/.test(dial), dial);
  ok('dial band flags the problem, not "MILDER"', !/MILDER/i.test(dialBand) && /VOID|CEILING|VIOLATION|CAN.?T SEE/i.test(dialBand), dialBand);

  // 2. the assessment headline — where the reported bug actually lived
  const big = d.getElementById('aBig').textContent.trim();
  const aband = d.getElementById('aBand').textContent.trim();
  ok('assessment headline does NOT show a green rank', !/\d+(st|nd|rd|th)/.test(big), big);
  ok('assessment band is NOT "MILDER THAN MOST NAVY DIVES"', !/MILDER/i.test(aband), aband);

  // 3. scrub — one click must not revert to the reassuring number
  w.enterScrub(); render();
  const sdial = d.getElementById('pct').textContent.trim();
  const sband = d.getElementById('band').textContent.trim();
  ok('scrub does NOT revert the dial to a rank', !/\d+(st|nd|rd|th)/.test(sdial), sdial);
  ok('scrub band still flags the problem', !/MILDER/i.test(sband), sband);

  console.log(fails ? '\nFAILURES: ' + fails : '\nALL PASSED');
  process.exit(fails ? 1 : 0);
}, 800);
