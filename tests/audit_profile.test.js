/* Core violation-audit tests, run against the app's real functions.
 *
 *   node tests/audit_profile.test.js
 *
 * Every dangerous profile here was CONSTRUCTED by an adversarial reviewer that
 * ported the app's own physics and confirmed the number. The old checks (deficit
 * of ascent minutes; final surface load) passed several of these. auditProfile
 * must catch all of them and must NOT false-flag the legitimate dives.
 *
 * Two facts, deliberately distinct:
 *   - violated       : physically over the ceiling at some instant, or surfaced
 *                      over the surfacing M-value. A hard physical fact.
 *   - inDistribution : the 3-scalar reduction loses no material physics, so the
 *                      model's rank means something. An epistemic fact.
 * The rank is trustworthy only when inDistribution AND not violated.
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

let fails = 0;
const ok = (n, c, x) => {
  console.log((c ? '  PASS  ' : '  FAIL  ') + n + (x !== undefined ? '  [' + x + ']' : ''));
  if (!c) fails++;
};
const sq = (dm, bt, at) => {
  const d = Math.max(dm / 18, 0.3);
  return [{t: 0, d: 0}, {t: d, d: dm}, {t: d + bt, d: dm}, {t: d + bt + at, d: 0}];
};

// --- the adversarial fleet: dangerous OR unrankable, must NOT be trusted ---
const DANGEROUS = {
  'reported sawtooth (surfaces 1.03x)':
    [{t:0,d:0},{t:3,d:45},{t:10,d:45},{t:11,d:33},{t:12,d:38},{t:13,d:30},{t:15,d:32},{t:17,d:22},{t:18,d:26},{t:20,d:12},{t:21,d:18},{t:23.6,d:0}],
  'blows stops, 3m/25min (surfaces 0.94x, peak 2.9m over)':
    [{t:0,d:0},{t:2,d:40},{t:25,d:40},{t:26.3,d:3},{t:50,d:3},{t:51,d:0}],
  '44m bolt to 2m/15min (surfaces 0.91x, peak 3.5m over)':
    [{t:0,d:0},{t:2.44,d:44},{t:18.44,d:44},{t:19.64,d:2},{t:34.64,d:2},{t:35.04,d:0}],
  '45m bolt to 4m/30min (surfaces 0.96x, peak 3.9m over)':
    [{t:0,d:0},{t:2.5,d:45},{t:24.5,d:45},{t:25.7,d:4},{t:55.7,d:4},{t:56.2,d:0}],
  'monotone multilevel (surfaces 1.13x)':
    [{t:0,d:0},{t:2.5,d:45},{t:12.5,d:45},{t:13.5,d:25},{t:38.5,d:25},{t:40,d:5},{t:43,d:5},{t:44,d:0}],
};

// --- legitimate dives: must be trusted (rankable, not flagged violated) ---
const LEGIT = {
  'rec 18m/40min + safety stop': [{t:0,d:0},{t:1.5,d:18},{t:40,d:18},{t:43,d:5},{t:46,d:5},{t:47,d:0}],
  'square 40m/8min bounce': sq(40, 8, 2),
  'shallow 12m/60min': sq(12, 60, 3),
  // A deeper rec dive WITH a safety stop: the stop off-gasses more than a linear
  // ascent, so its square reconstruction disagrees by ~0.04 load. That is a
  // genuine square dive (Simplicity measured such dives at 0.03-0.08), NOT out of
  // distribution — a safety stop must not make a dive unrankable.
  '28m/18min + 5m safety stop': [{t:0,d:0},{t:1.4,d:28},{t:19,d:28},{t:21.4,d:9},{t:22,d:5},{t:25,d:5},{t:26,d:0}],
};

setTimeout(() => {
  const w = boot();
  ok('auditProfile is exposed', typeof w.auditProfile === 'function');

  console.log('--- dangerous / unrankable: rank must be WITHHELD ---');
  for (const [name, poly] of Object.entries(DANGEROUS)) {
    const a = w.auditProfile(poly);
    ok('withheld: ' + name, a && a.trustRank === false,
       a ? `violated=${a.violated} inDist=${a.inDistribution}` : 'null');
  }

  console.log('--- legitimate: rank must be TRUSTED ---');
  for (const [name, poly] of Object.entries(LEGIT)) {
    const a = w.auditProfile(poly);
    ok('trusted: ' + name, a && a.trustRank === true,
       a ? `violated=${a.violated} inDist=${a.inDistribution} over=${a.maxOverM.toFixed(2)} L=${a.finalLoad.toFixed(3)}` : 'null');
  }

  console.log('--- the specific facts ---');
  // surfaced-bent detection (final load >= 1.0)
  const saw = w.auditProfile(DANGEROUS['reported sawtooth (surfaces 1.03x)']);
  ok('sawtooth: finalLoad >= 1.0 detected', saw.finalLoad >= 1.0, saw.finalLoad.toFixed(3));
  // at-depth ceiling breach that off-gasses before surfacing (surface load < 1.0!)
  const bolt = w.auditProfile(DANGEROUS['44m bolt to 2m/15min (surfaces 0.91x, peak 3.5m over)']);
  ok('bolt: surfaces UNDER 1.0x yet still caught', bolt.finalLoad < 1.0 && bolt.violated,
     `L=${bolt.finalLoad.toFixed(3)} over=${bolt.maxOverM.toFixed(2)}`);
  // a PROPERLY STAGED deco dive (built with the app's own ceiling scheduler,
  // stopping at each ceiling) must NOT be called violated. A staged deco dive
  // surfaces AT the M-value limit (~1.02x) by design, so a naive finalLoad>=1.0
  // test would false-alarm on it; the ceiling test must not.
  const P_SURFACE = 1.01325, FSW_TO_BAR = 0.030643, FN2 = 0.79;   // const, not on window
  const M2F = 1 / (10 * FSW_TO_BAR), F2M = 10 * FSW_TO_BAR;
  const staged = (dm, bt) => {
    const depthFsw = dm * M2F;
    const desc = Math.max(depthFsw / 60, 0.5);
    let poly = [{t:0,d:0},{t:desc,d:dm},{t:desc+bt,d:dm}];
    let P = w.loadToBottom(depthFsw, bt), d = depthFsw, tnow = desc + bt;
    for (let it=0; it<4000; it++) {
      if (d <= 0) break;
      const target = Math.min(Math.ceil(w.ceilingFsw(P)/10)*10, d);
      if (target < d) {
        const dt=(d-target)/30;
        P = w.haldane(P, (P_SURFACE + ((d+target)/2)*FSW_TO_BAR)*FN2, dt);
        tnow += dt; d = target; poly.push({t:tnow, d:d*F2M});
      } else {
        P = w.haldane(P, (P_SURFACE + d*FSW_TO_BAR)*FN2, 0.5);
        tnow += 0.5;
        if (poly.length && Math.abs(poly[poly.length-1].d - d*F2M) < 1e-9) poly[poly.length-1].t = tnow;
        else poly.push({t:tnow, d:d*F2M});
      }
    }
    return poly;
  };
  for (const [dm, bt] of [[40,25],[45,20],[50,30]]) {
    const a = w.auditProfile(staged(dm, bt));
    ok(`properly-staged ${dm}m/${bt}min deco not falsely flagged violated`, !a.violated,
       `over=${a.maxOverM.toFixed(2)} L=${a.finalLoad.toFixed(3)}`);
  }

  console.log('--- Devil bug: bottom-time collapse is backstopped by inDistribution ---');
  // A hand-flown dive whose max depth is a single sample: diveScalars collapses
  // bottom time. The square reconstruction then can't match reality -> OOB ->
  // rank withheld, so the corrupted scalar cannot produce a false confident rank.
  const nudged = [{t:0,d:0},{t:1.5,d:18},{t:1.6,d:18.9},{t:1.7,d:18},{t:35,d:18},{t:37,d:0}];
  const nud = w.auditProfile(nudged);
  ok('single-sample-max-depth dive is not confidently ranked', nud.trustRank === false,
     `inDist=${nud.inDistribution}`);

  console.log(fails ? '\nFAILURES: ' + fails : '\nALL PASSED');
  process.exit(fails ? 1 : 0);
}, 700);
