# Dive-Computer UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the DecoStress physics into a shared `deco-engine.js`, then build `divecomputer.html` — a Shearwater-style dive watch beside a console driving an auto-playing demo dive.

**Architecture:** Classic scripts share one global scope, so `deco-engine.js` loaded before a page's inline script exposes `predict()`, `ceilingFsw()`, etc. unchanged. `index.html` loses its inline copies (top-level `const` cannot be redeclared across two scripts, so the copies MUST be deleted). The new page holds no physics — it reads the engine and draws.

**Tech Stack:** Vanilla JS + inline CSS. Shared `deco-engine.js`. jsdom for JS tests, pytest for Python guards. Served over `http://localhost:8899` (external `<script src>` needs a server).

## Global Constraints

- **Single source of truth.** The physics and fitted-model constants live ONCE, in `deco-engine.js`. No page re-declares them. A second copy is Correction 10.
- **The existing 76-test suite must still pass after extraction** — it is the guard that the engine is intact.
- **The rank obeys `voidVerdict`** everywhere it appears: percentile when `trustRank`, else `VOID` (never a probability).
- **Not a dive planner.** The "never use for real dive decisions" disclaimer appears on the new page.
- Python: `/opt/miniconda3/bin/python3`. jsdom at `$CLAUDE_JOB_DIR/tmp/node_modules/jsdom`.

## File Structure

| File | Responsibility |
|------|----------------|
| `decostress_app/deco-engine.js` | CREATE. All DOM-free physics + fitted-model constants. The single source of truth. |
| `decostress_app/index.html` | MODIFY. Load the engine; delete the now-shared inline definitions. Everything else unchanged. |
| `decostress_app/divecomputer.html` | CREATE. The watch + console page. Loads the engine. |
| `tests/*.test.js` (existing) | MODIFY. `boot()` helpers must inline `deco-engine.js` (jsdom can't fetch `<script src>`). |
| `tests/dive_computer_dom.test.js` | CREATE. Boots the new page, checks the watch + rank void. |
| `tests/test_web_model_sync.py` | MODIFY. Drift guards read `deco-engine.js` (the constants moved there). |

---

## Task 1: Extract the shared engine

**Files:**
- Create: `decostress_app/deco-engine.js`
- Modify: `decostress_app/index.html`, all four existing `tests/*.test.js` boot helpers, `tests/test_web_model_sync.py`

**Interfaces:**
- Produces (globals, unchanged signatures): `ZHL, HALF, A_C, B_C, NC, FN2, P_SURFACE, FSW_TO_BAR, M_TO_FSW, ambBar, inspN2, mValue, mSurface, RM, RM_Q, ASCENT_FSW_PER_MIN, DESCENT_FSW_PER_MIN, STOP_FSW, DT_SCHED, K_C, ceilingFsw, haldane, alvAtFsw, requiredAscentMin, loadToBottom, diveScalars, realScore, CEIL_TOL_M, walkAudit, squareFromScalars, auditProfile, predict, percentileOf, realDrivers, cohortDecileRate, voidVerdict, simulatePolyline`.
- `simulatePolyline(poly) -> {st, li, tissue}`; `predict(poly) -> {pct, trustRank, violated, inDistribution, maxOverM, finalLoad, requiredAscent, ...}`; `voidVerdict(r) -> {voided, band, reason}`.

- [ ] **Step 1: Create `deco-engine.js` by copying the pure definitions**

Create `decostress_app/deco-engine.js`. Copy — verbatim — from `index.html` the whole block from the `/* REAL MODEL ... */` header through the end of `cohortDecileRate()` (the contiguous pure region), then also `voidVerdict()` and `simulatePolyline()` (which sit later, interleaved with page state). Prepend this header:

```js
/* ================================================================
   deco-engine.js — the DecoStress physics, shared by every page.

   DOM-free and page-free: every function here is a pure function of its
   arguments. Loaded by index.html and divecomputer.html as a classic script
   BEFORE their own inline script, so they share global scope and call these
   names directly. It lives in exactly one file on purpose — a second copy is
   how compartment 16's b-coefficient once diverged across four files
   (Correction 10). scripts/export_web_model.py regenerates the fitted
   constants (RM, RM_Q); do not hand-tune them.
   ================================================================ */
```

Do NOT include: `MIN_PER_SEC`, `depthRate`, `flags`, `keys`, `currentStress` (reads the global `tissue`), `LOG_*`, or anything that references `$(...)` or page globals — those stay in `index.html`.

- [ ] **Step 2: Load the engine in `index.html` and delete the inline copies**

In `index.html`, immediately before the main inline `<script>` that begins `const $=id=>document.getElementById(id);`, add:

```html
<script src="deco-engine.js"></script>
```

Then delete from `index.html`'s inline script every definition now in `deco-engine.js`: the `/* REAL MODEL */` block through `cohortDecileRate()`, plus `voidVerdict()` and `simulatePolyline()`. Leave `MIN_PER_SEC`, `depthRate`, `flags`, `keys`, `currentStress`, and all DOM/log code in place. (Top-level `const` redeclaration across two scripts throws "already declared", so a missed deletion fails loudly — that is the check.)

- [ ] **Step 3: Update the JS test boot helpers to inline the engine**

jsdom cannot fetch `<script src>` in-test. In EACH of `tests/audit_profile.test.js`, `tests/audit_surfaces.test.js`, `tests/dive_log_dom.test.js`, and any other `*.test.js` with a `boot()` that reads `index.html`, replace the `<script src="...cdnjs...">` removal line so it ALSO inlines the engine. Change:

```js
  const html = fs.readFileSync(APP, 'utf8')
    .replace(/<script src="https:\/\/cdnjs[^>]*><\/script>/, '<script></script>');
```

to:

```js
  const ENGINE = fs.readFileSync(path.join(__dirname, '..', 'decostress_app', 'deco-engine.js'), 'utf8');
  const html = fs.readFileSync(APP, 'utf8')
    .replace(/<script src="https:\/\/cdnjs[^>]*><\/script>/, '<script></script>')
    .replace('<script src="deco-engine.js"></script>', '<script>' + ENGINE + '</script>');
```

- [ ] **Step 4: Point the Python drift guards at the engine file**

In `tests/test_web_model_sync.py`, the constants `ZHL`, `RM` fields (`mean/scale/coef/intercept/n/nDcs`), and `RM_Q` now live in `deco-engine.js`, not `index.html`. Add an engine reader and use it in the three drift guards:

```python
ENGINE = os.path.join(HERE, "..", "decostress_app", "deco-engine.js")


def _engine() -> str:
    with open(ENGINE) as f:
        return f.read()
```

In `test_zhl16c_table_matches_the_shared_module`, `test_fitted_coefficients_match_a_fresh_fit`, `test_cohort_quantiles_match_a_fresh_fit`, and `test_cohort_size_claims_match_the_data`, change the `text = _html()` line to `text = _engine()`. Leave the UI-behaviour guards (`test_app_emits_no_probability_claim`, `test_violation_is_a_continuous_ceiling_audit...`, the dive-log guards, `test_ascent_coefficient...`) reading `_html()` — those assert page behaviour, which stays in `index.html`. For any that reference BOTH engine constants and page strings, read `_html() + _engine()`.

- [ ] **Step 5: Run the full suite — the extraction is correct iff nothing changed**

Run: `node tests/audit_profile.test.js && node tests/audit_surfaces.test.js && node tests/dive_log_dom.test.js`
Expected: `ALL PASSED` for each. A `ReferenceError: X is not defined` means a definition was deleted but not moved; an `already been declared` means a copy was left behind.

Run: `/opt/miniconda3/bin/python3 -m pytest -q`
Expected: `76 passed` (same as before the extraction).

- [ ] **Step 6: Confirm the app still renders in a browser (smoke)**

Run: `node -e "const {JSDOM}=require(process.env.CLAUDE_JOB_DIR+'/tmp/node_modules/jsdom');const fs=require('fs'),p='decostress_app/';const eng=fs.readFileSync(p+'deco-engine.js','utf8');let h=fs.readFileSync(p+'index.html','utf8').replace(/<script src=\"https:\/\/cdnjs[^>]*><\/script>/,'<script></script>').replace('<script src=\"deco-engine.js\"></script>','<script>'+eng+'</script>');const errs=[];const w=new JSDOM(h,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://localhost/',beforeParse(w){w.HTMLCanvasElement.prototype.getContext=()=>new Proxy({},{get:()=>()=>({addColorStop(){}})});w.onerror=m=>errs.push(m);}}).window;setTimeout(()=>{console.log('predict is',typeof w.predict);console.log(errs.length?'ERRORS '+errs:'no errors');process.exit(errs.length||typeof w.predict!=='function'?1:0)},600);"`
Expected: `predict is function` and `no errors`.

- [ ] **Step 7: Commit**

```bash
git add decostress_app/deco-engine.js decostress_app/index.html tests/
git commit -m "refactor: extract the physics into shared deco-engine.js (single source of truth)"
```

---

## Task 2: The dive-computer page

**Files:**
- Create: `decostress_app/divecomputer.html`
- Test: `tests/dive_computer_dom.test.js`

**Interfaces:**
- Consumes from the engine: `simulatePolyline`, `predict`, `voidVerdict`, `ceilingFsw`, `requiredAscentMin`, `mSurface`, `HALF`, `NC`, `M_TO_FSW`, `inspN2`.
- Produces (page globals, for tests): `renderAt(t)` renders both devices at dive-time `t` minutes; `DEMO` the demo polyline; `ndlTtsAt(tissue, depthM)` -> `{ndl, tts, ceilingM}`.

- [ ] **Step 1: Write the failing test**

```js
// tests/dive_computer_dom.test.js
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
        new Proxy({}, {get: () => () => ({addColorStop(){}, setLineDash(){}, width: 10})});
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

  // in-distribution point -> a rank; scrub past the ceiling -> VOID
  ok('rank shows a number on the in-distribution demo dive', /\d/.test(d.getElementById('wRank').textContent),
     d.getElementById('wRank').textContent);

  // a hand-made bent state: deep, then straight to surface (over ceiling)
  w.DEMO_OVERRIDE = [{t:0,d:0},{t:2.5,d:45},{t:22,d:45},{t:23.2,d:0}];
  w.renderProfileAt(w.DEMO_OVERRIDE, w.DEMO_OVERRIDE[3].t);
  ok('rank reads VOID on a bent profile', /VOID/i.test(d.getElementById('wRank').textContent),
     d.getElementById('wRank').textContent);

  console.log(fails ? '\nFAILURES: ' + fails : '\nALL PASSED');
  process.exit(fails ? 1 : 0);
}, 700);
```

- [ ] **Step 2: Run it — verify it fails**

Run: `node tests/dive_computer_dom.test.js`
Expected: FAIL — `ENOENT` (page not created yet).

- [ ] **Step 3: Create `divecomputer.html`**

Create `decostress_app/divecomputer.html`. It is a self-contained page whose body has two columns: `.watch` (left) and `.console` (right). It loads the engine and defines the render functions. Full file:

```html
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DecoStress — dive computer</title>
<style>
  :root{--bg:#05080c;--face:#0a1017;--ink:#e8f2f6;--dim:#6b8595;--line:#16242f;
        --cyan:#37d0e0;--safe:#2fbf9e;--amber:#f2a63a;--risk:#e2574c;--mono:'IBM Plex Mono',ui-monospace,monospace}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);font-family:var(--mono);
       min-height:100vh;display:flex;align-items:center;justify-content:center;gap:40px;flex-wrap:wrap;padding:32px}
  .banner{position:fixed;top:0;left:0;right:0;background:var(--risk);color:#fff;padding:8px;text-align:center;font-size:13px;display:none;z-index:9}
  /* ---- the watch ---- */
  .watch{width:340px;background:linear-gradient(160deg,#0c141c,#060a0e);border:1px solid var(--line);
         border-radius:34px;padding:26px 24px;box-shadow:0 30px 80px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.04);
         transition:box-shadow .4s}
  .watch.amber{box-shadow:0 0 60px rgba(242,166,58,.28),inset 0 0 60px rgba(242,166,58,.08)}
  .watch.risk{box-shadow:0 0 70px rgba(226,87,76,.4),inset 0 0 70px rgba(226,87,76,.12)}
  .wtop{display:flex;justify-content:space-between;align-items:baseline;color:var(--dim);font-size:11px;letter-spacing:.15em;text-transform:uppercase}
  .wprimary{margin:14px 0 4px;display:flex;align-items:flex-end;gap:8px}
  .wprimary .n{font-size:76px;font-weight:600;line-height:.9;letter-spacing:-.02em}
  .wprimary .u{font-size:20px;color:var(--dim);margin-bottom:10px}
  .wrow{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}
  .cell{background:var(--face);border:1px solid var(--line);border-radius:12px;padding:11px 13px}
  .cell .lbl{font-size:9.5px;letter-spacing:.16em;color:var(--dim);text-transform:uppercase}
  .cell .v{font-size:26px;font-weight:600;margin-top:3px}
  .cell .v small{font-size:13px;color:var(--dim)}
  .arate{margin-top:16px}
  .arate .lbl{font-size:9.5px;letter-spacing:.16em;color:var(--dim);text-transform:uppercase;display:flex;justify-content:space-between}
  .arbar{height:8px;background:var(--face);border-radius:5px;margin-top:6px;overflow:hidden;position:relative}
  .arfill{height:100%;width:0;background:var(--safe);transition:width .12s,background .12s}
  .armark{position:absolute;top:-2px;bottom:-2px;left:50%;width:1px;background:var(--dim)}
  .tissues{margin-top:16px}
  .tissues .lbl{font-size:9.5px;letter-spacing:.16em;color:var(--dim);text-transform:uppercase;margin-bottom:7px}
  #wTissues{display:flex;gap:3px;align-items:flex-end;height:56px}
  .tbar{flex:1;background:var(--safe);border-radius:2px 2px 0 0;min-height:2px;transition:height .12s,background .12s}
  .wrank{margin-top:18px;border-top:1px solid var(--line);padding-top:14px;display:flex;justify-content:space-between;align-items:baseline}
  .wrank .lbl{font-size:9.5px;letter-spacing:.14em;color:var(--dim);text-transform:uppercase;max-width:180px;line-height:1.4}
  #wRank{font-size:30px;font-weight:700}
  #wRank.void{color:var(--risk);font-size:22px;letter-spacing:.08em}
  /* ---- the console ---- */
  .console{width:520px;max-width:92vw;background:var(--face);border:1px solid var(--line);border-radius:16px;padding:22px}
  .console h1{font-size:15px;letter-spacing:.04em;font-weight:600}
  .console .sub{color:var(--dim);font-size:12px;margin:6px 0 18px;line-height:1.5}
  .ctrls{display:flex;gap:10px;align-items:center;margin-bottom:16px}
  .ctrls button{background:#0d1a22;border:1px solid var(--line);color:var(--ink);border-radius:9px;
                 padding:9px 15px;font-family:var(--mono);font-size:13px;cursor:pointer}
  .ctrls button:hover{border-color:var(--cyan);color:var(--cyan)}
  .ctrls .spd{margin-left:auto;display:flex;gap:6px}
  .ctrls .spd button.on{border-color:var(--cyan);color:var(--cyan)}
  #scrub{width:100%;margin:4px 0 14px;accent-color:var(--cyan)}
  #profile{width:100%;height:200px;display:block;background:#060b0f;border:1px solid var(--line);border-radius:10px}
  .disc{color:var(--dim);font-size:11px;margin-top:14px;line-height:1.5;border-top:1px solid var(--line);padding-top:12px}
  .disc b{color:var(--risk)}
</style></head><body>

<div class="banner" id="banner">Engine not loaded — open this page over http:// (the local server), not as a file://</div>

<div class="watch" id="watch">
  <div class="wtop"><span>DecoStress · ZH-L16C</span><span id="wClock">0:00</span></div>
  <div class="wprimary"><span class="n" id="wDepth">0</span><span class="u">m</span></div>
  <div class="wrow">
    <div class="cell"><div class="lbl" id="wNdlLbl">No-deco</div><div class="v" id="wNdl">∞</div></div>
    <div class="cell"><div class="lbl">Ceiling</div><div class="v" id="wCeil">—</div></div>
  </div>
  <div class="arate">
    <div class="lbl"><span>Ascent rate</span><span id="wRateN">0 m/min</span></div>
    <div class="arbar"><span class="arfill" id="wRateBar"></span><span class="armark"></span></div>
  </div>
  <div class="tissues"><div class="lbl">Tissue loading — 16 compartments</div><div id="wTissues"></div></div>
  <div class="wrank">
    <div class="lbl" id="wRankLbl">DecoStress rank<br><span style="color:var(--dim)">vs 1,948 real dives</span></div>
    <div id="wRank">—</div>
  </div>
</div>

<div class="console">
  <h1>Dive console</h1>
  <p class="sub">A demo dive, played through the same Bühlmann engine as the explorer.
     Play it, change speed, or scrub. The watch reflects the diver in real time.</p>
  <div class="ctrls">
    <button id="play">⏸ Pause</button>
    <button id="restart">↺ Restart</button>
    <div class="spd">
      <button data-spd="1">1×</button><button data-spd="4" class="on">4×</button><button data-spd="10">10×</button>
    </div>
  </div>
  <input type="range" id="scrub" min="0" max="1000" value="0">
  <canvas id="profile" width="960" height="400"></canvas>
  <p class="disc"><b>Not a dive planner.</b> A teaching demo of the physics, not a device to
     make in-water decisions with. The rank is a weak ranking (AUC 0.64), shown VOID whenever
     the dive goes over its ceiling or off the model's map.</p>
</div>

<script src="deco-engine.js"></script>
<script>
const $=id=>document.getElementById(id);
if(typeof predict!=='function'){ $('banner').style.display='block'; }

/* A realistic recreational dive: descent, bottom, ascent, 5 m safety stop, surface.
   Chosen to stay in-distribution so the rank shows a real number for most of it. */
const DEMO=[{t:0,d:0},{t:1.4,d:28},{t:19,d:28},{t:21.4,d:9},{t:22,d:5},{t:25,d:5},{t:26,d:0}];
const T_END=DEMO[DEMO.length-1].t;
const SURFACE_WATCH=8;   // minutes of post-surface off-gassing shown

function polyUpTo(poly,t){
  const out=[]; for(let i=0;i<poly.length;i++){
    if(poly[i].t<=t) out.push(poly[i]);
    else { const a=poly[i-1]; if(a){ const f=(t-a.t)/(poly[i].t-a.t); out.push({t, d:a.d+(poly[i].d-a.d)*f}); } break; }
  }
  if(!out.length) out.push({t:0,d:0});
  return out;
}
function depthAt(poly,t){ const s=polyUpTo(poly,t); return s[s.length-1].d; }

/* NDL and TTS from the current tissue state. NDL: forward-search how long you could
   stay at this depth before the ceiling leaves the surface. TTS: the obligation now. */
function ndlTtsAt(tissue,depthM){
  const ceilFsw=ceilingFsw(tissue), ceilingM=ceilFsw/M_TO_FSW;
  const tts=requiredAscentMin(tissue,depthM*M_TO_FSW).total;
  let ndl=Infinity;
  if(depthM>0.5){
    let P=tissue.slice(); const alv=inspN2(depthM);
    for(let m=0;m<=99;m++){
      if(ceilingFsw(P)>0.5){ ndl=m; break; }
      P=haldane(P,alv,1);
    }
    if(ndl===Infinity && ceilingFsw(P)>0.5) ndl=99;
  }
  return {ndl,tts,ceilingM};
}

function renderProfileAt(poly,t){
  const s=polyUpTo(poly,t);
  const {tissue}=simulatePolyline(s.length>=2?s:[{t:0,d:0},{t:0.01,d:0}]);
  const depthM=depthAt(poly,t);
  const {ndl,tts,ceilingM}=ndlTtsAt(tissue,depthM);

  $('wDepth').textContent=Math.round(depthM);
  $('wClock').textContent=Math.floor(t)+':'+String(Math.floor((t%1)*60)).padStart(2,'0');

  // NDL while no obligation; TTS + ceiling once in deco
  if(ceilingM>0.5){
    $('wNdlLbl').textContent='TTS'; $('wNdl').innerHTML=Math.ceil(tts)+'<small> min</small>';
    $('wCeil').textContent=Math.ceil(ceilingM)+' m';
  }else{
    $('wNdlLbl').textContent='No-deco';
    $('wNdl').innerHTML = ndl===Infinity?'∞':(ndl>=99?'99+':ndl)+'<small> min</small>';
    $('wCeil').textContent='—';
  }

  // ascent rate from the last segment
  let rate=0; for(let i=1;i<s.length;i++){ const seg=s[i], a=s[i-1]; if(seg.t>t-0.2){ const dt=seg.t-a.t; if(dt>0) rate=Math.max(rate,(a.d-seg.d)/dt); } }
  const rec=9;   // recommended max m/min
  $('wRateN').textContent=Math.max(0,rate).toFixed(0)+' m/min';
  const bar=$('wRateBar'); bar.style.width=Math.min(100,Math.max(0,rate)/(rec*2)*100)+'%';
  bar.style.background=rate>rec*1.6?'var(--risk)':rate>rec?'var(--amber)':'var(--safe)';

  // tissue strip
  const box=$('wTissues');
  if(box.children.length!==NC){ box.innerHTML=''; for(let i=0;i<NC;i++){ const b=document.createElement('div'); b.className='tbar'; box.appendChild(b);} }
  let peak=0;
  for(let c=0;c<NC;c++){ const ratio=tissue[c]/mSurface(c); peak=Math.max(peak,ratio);
    const el=box.children[c]; el.style.height=Math.min(100,ratio/1.4*100)+'%';
    el.style.background=ratio>=1.0?'var(--risk)':ratio>=0.85?'var(--amber)':'var(--safe)'; }

  // face tint from peak loading
  const w=$('watch'); w.classList.toggle('risk',peak>=1.0); w.classList.toggle('amber',peak>=0.85&&peak<1.0);

  // the DecoStress rank, obeying voidVerdict
  const r=predict(s);
  const rankEl=$('wRank');
  if(!r || depthM<=0.5 && t<0.2){ rankEl.textContent='—'; rankEl.classList.remove('void'); }
  else{
    const v=voidVerdict(r);
    if(v.voided){ rankEl.textContent = r.violated?'VOID · OVER CEILING':'VOID · UNRANKABLE'; rankEl.classList.add('void'); }
    else{ const p=Math.round(r.pct*100); const s2=['th','st','nd','rd'],vv=p%100;
      rankEl.textContent=p+(s2[(vv-20)%10]||s2[vv]||s2[0]); rankEl.classList.remove('void'); }
  }

  drawProfile(poly,t);
}
function renderAt(t){ renderProfileAt(DEMO,t); }

function drawProfile(poly,t){
  const cv=$('profile'); if(!cv) return; const ctx=cv.getContext('2d'); if(!ctx) return;
  const W=cv.width,H=cv.height,PADL=42,PADR=14,PADT=14,PADB=26;
  ctx.clearRect(0,0,W,H);
  const maxT=T_END+SURFACE_WATCH, maxD=Math.max(10,...poly.map(p=>p.d));
  const X=tt=>PADL+(tt/maxT)*(W-PADL-PADR), Y=dd=>PADT+(dd/maxD)*(H-PADT-PADB);
  ctx.strokeStyle='#16242f';ctx.fillStyle='#6b8595';ctx.font='11px monospace';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const dd=maxD*i/4;ctx.beginPath();ctx.moveTo(PADL,Y(dd));ctx.lineTo(W-PADR,Y(dd));ctx.stroke();
    ctx.textAlign='right';ctx.fillText(Math.round(dd)+'m',PADL-5,Y(dd)+4);}
  // full profile faint, travelled part bright
  ctx.strokeStyle='#24404f';ctx.lineWidth=1.5;ctx.beginPath();
  poly.forEach((p,i)=>i?ctx.lineTo(X(p.t),Y(p.d)):ctx.moveTo(X(p.t),Y(p.d)));ctx.stroke();
  const s=polyUpTo(poly,t);
  ctx.strokeStyle='#37d0e0';ctx.lineWidth=2.4;ctx.beginPath();
  s.forEach((p,i)=>i?ctx.lineTo(X(p.t),Y(p.d)):ctx.moveTo(X(p.t),Y(p.d)));ctx.stroke();
  const dNow=depthAt(poly,t);
  ctx.fillStyle='#37d0e0';ctx.beginPath();ctx.arc(X(t),Y(dNow),4,0,7);ctx.fill();
}

/* ---- playback clock ---- */
let playing=true, speed=4, simT=0, lastMs=null;
function frame(ms){
  if(lastMs==null) lastMs=ms; const dt=(ms-lastMs)/1000; lastMs=ms;
  if(playing){ simT+=dt*speed*0.5; if(simT>T_END+SURFACE_WATCH){ simT=0; } $('scrub').value=Math.round(simT/(T_END+SURFACE_WATCH)*1000); }
  if(typeof predict==='function') renderAt(simT);
  requestAnimationFrame(frame);
}
$('play').onclick=()=>{ playing=!playing; $('play').textContent=playing?'⏸ Pause':'▶ Play'; lastMs=null; };
$('restart').onclick=()=>{ simT=0; lastMs=null; };
document.querySelectorAll('.spd button').forEach(b=>b.onclick=()=>{
  speed=+b.dataset.spd; document.querySelectorAll('.spd button').forEach(x=>x.classList.toggle('on',x===b)); });
$('scrub').oninput=e=>{ playing=false; $('play').textContent='▶ Play'; simT=(+e.target.value/1000)*(T_END+SURFACE_WATCH); if(typeof predict==='function') renderAt(simT); };

if(typeof predict==='function') requestAnimationFrame(frame);
</script></body></html>
```

- [ ] **Step 4: Run the test — verify it passes**

Run: `node tests/dive_computer_dom.test.js`
Expected: `ALL PASSED` (7 checks). If the bent-profile check fails, confirm `renderProfileAt` is a global (declared with `function`, not inside a closure).

- [ ] **Step 5: Commit**

```bash
git add decostress_app/divecomputer.html tests/dive_computer_dom.test.js
git commit -m "feat: dive-computer page — Shearwater-style watch + console on the shared engine"
```

---

## Task 3: Link the pages and add the new-page guard

**Files:**
- Modify: `decostress_app/index.html` (a link to the dive computer), `tests/test_web_model_sync.py`

**Interfaces:** none (terminal task).

- [ ] **Step 1: Add a Python guard that the new page uses the engine, not a copy**

Append to `tests/test_web_model_sync.py`:

```python
def test_dive_computer_page_uses_the_shared_engine_not_a_copy():
    """The new page must LOAD deco-engine.js, never re-declare the physics.

    A second inline copy of the engine is the Correction 10 failure mode. The
    page is allowed to read predict()/simulatePolyline()/voidVerdict(); it must
    not define ZHL, RM, or predict itself.
    """
    page_path = os.path.join(HERE, "..", "decostress_app", "divecomputer.html")
    with open(page_path) as f:
        page = f.read()
    assert '<script src="deco-engine.js">' in page, "the page must load the shared engine"
    assert "const ZHL" not in page and "const RM " not in page and "const RM_Q" not in page, (
        "the page must not re-declare the fitted constants -- that is a second "
        "copy that can drift from the source"
    )
    assert "function predict(" not in page and "function auditProfile(" not in page, (
        "the page must call the engine's predict(), not define its own"
    )
    # the rank on the watch must obey voidVerdict, not print a raw percentile
    assert "voidVerdict(" in page, "the watch rank must run through voidVerdict"
```

- [ ] **Step 2: Run it**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/test_web_model_sync.py -q`
Expected: all pass.

- [ ] **Step 3: Add a link from the main app to the dive computer**

In `index.html`, find the tab bar `<div class="tabs" role="tablist" ...>` and, immediately AFTER the closing `</div>` of that tab bar, add:

```html
  <div style="text-align:right;max-width:1100px;margin:0 auto;padding:0 20px">
    <a href="divecomputer.html" style="color:#37d0e0;font-size:12px;text-decoration:none">Open the dive-computer view →</a>
  </div>
```

- [ ] **Step 4: Full suite**

Run: `/opt/miniconda3/bin/python3 -m pytest -q && node tests/audit_profile.test.js && node tests/audit_surfaces.test.js && node tests/dive_log_dom.test.js && node tests/dive_computer_dom.test.js`
Expected: `77 passed` (python) and `ALL PASSED` (each js).

- [ ] **Step 5: Commit**

```bash
git add decostress_app/index.html tests/test_web_model_sync.py
git commit -m "feat: link the dive-computer view; guard it against a second engine copy"
```

---

## Self-Review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| Extract DOM-free engine to `deco-engine.js`, both pages load it | 1 |
| `index.html` loses inline copies; classic-script global scope | 1 |
| Existing 76-suite passes after extraction | 1 (Step 5) |
| Drift guards read the engine file | 1 (Step 4) |
| Test harnesses inline the engine (jsdom can't fetch src) | 1 (Step 3) |
| Watch: depth, NDL/TTS, ceiling, ascent-rate bar, 16 tissue bars | 2 |
| Watch: DecoStress rank via `voidVerdict`, VOID when bent/OOD | 2 |
| Face tint green/amber/red by peak loading | 2 |
| Console: play/pause, speed 1/4/10×, scrub, profile graph | 2 |
| Auto-playing looping demo dive | 2 |
| Shearwater-black aesthetic | 2 |
| Engine-not-loaded banner; scrub-to-0; over-ceiling VOID; no canvas | 2 (banner, `renderAt`, void check) |
| "Not a dive planner" disclaimer | 2 |
| New page guard: no second engine copy | 3 |

**Placeholder scan:** none. Every code step is complete and runnable.

**Type consistency:** `predict(poly)` returns `{pct, trustRank, violated, inDistribution, ...}` used in Task 2's rank logic and matched by Task 1's Produces block. `voidVerdict(r) -> {voided, band, reason}` used verbatim. `simulatePolyline(s) -> {tissue}` used in `renderProfileAt`. `ndlTtsAt` / `renderProfileAt` / `renderAt` / `DEMO` are page globals the test consumes. `mSurface`, `ceilingFsw`, `requiredAscentMin`, `haldane`, `inspN2`, `M_TO_FSW`, `NC`, `HALF` are all engine exports listed in Task 1.

**Deviation from spec:** `LOG_COLOURS` stays in `index.html` (it is a dive-log UI constant, used only by the log chart) rather than moving to the engine — moving it would be churn for no reuse. Noted so the engine's exported list is accurate.
