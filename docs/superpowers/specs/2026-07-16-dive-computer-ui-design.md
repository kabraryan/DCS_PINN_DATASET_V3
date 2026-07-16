# Dive-Computer UI — Design Spec

**Date:** 2026-07-16
**Component:** shared engine `decostress_app/deco-engine.js` + new page `decostress_app/divecomputer.html`
**Status:** approved, ready for implementation plan

## Purpose

A second page that presents the DecoStress physics as a realistic **dive
computer**: a wrist-watch device (the decompression status display) next to a
console that drives the dive. A scripted demo dive auto-plays and loops; the
console can pause, change speed, and scrub.

The point is to show the same engine rendered as an authentic instrument — and to
carry the project's honesty discipline onto a device that *looks* authoritative.

## The engine extraction (the load-bearing change)

The physics currently lives inline in `decostress_app/index.html`. To genuinely
reuse it — not copy it — the pure, DOM-free engine moves to
`decostress_app/deco-engine.js`, loaded by **both** pages via `<script src>`
before their page script. Classic scripts share one global scope, so the app's
remaining code calls `predict()`, `ceilingFsw()`, etc. unchanged — no export /
import rewiring, only deletion of the now-shared definitions from `index.html`.

**Why this and not a second copy:** Correction 10 — a coefficient hand-copied
across four files that then diverged — is this project's recurring wound. A second
inline copy would triple the drift surface. One module is the only option that
reuses rather than duplicates.

**De-risking:** the existing 76-test suite (including the web-model-sync guards
that assert the app's constants match the fitted Python model) must still pass
after the extraction. If the engine changed, those fail.

### What moves to `deco-engine.js` (pure — functions of their arguments only)

Constants: `ZHL`, `HALF`, `A_C`, `B_C`, `NC`, `FN2`, `P_SURFACE`, `FSW_TO_BAR`,
`M_TO_FSW`, `RM`, `RM_Q`, `CEIL_TOL_M`, `ASCENT_FSW_PER_MIN`,
`DESCENT_FSW_PER_MIN`, `STOP_FSW`, `DT_SCHED`, `K_C`, `LOG_COLOURS`.
Helpers: `ambBar`, `inspN2`, `mValue`, `mSurface`, `alvAtFsw`.
Physics: `ceilingFsw`, `haldane`, `requiredAscentMin`, `loadToBottom`.
Profile/model: `diveScalars`, `squareFromScalars`, `realScore`, `walkAudit`,
`auditProfile`, `predict`, `percentileOf`, `realDrivers`, `cohortDecileRate`,
`voidVerdict`, `simulatePolyline`.

### What STAYS in `index.html` (reads app globals or the DOM)

`currentStress` (reads the global `tissue` array), everything under the dive-log
store (`LOG_KEY`, `logLoad`, …), `liveRisk`, `refreshValues`, `smooth`,
`showAssessment`, `renderLog`, presets, input handlers, the 3-D diver. These
depend on page state and are not shared.

### Test-harness consequence

jsdom cannot fetch an external `<script src>` (no server in-test). The existing
JS test harnesses read `index.html`; they must be updated to **inline
`deco-engine.js`** into the HTML before booting (read both files, splice the
engine into a `<script>` tag). One shared `boot()` helper does this.

## `divecomputer.html` — components

Single self-contained page (plus the shared engine + the three.js-free layout).
No 3-D. Loads `deco-engine.js`.

### 1. The watch (hero, left) — Shearwater-black instrument

A square device face, high-contrast, one accent colour, showing live:

- **Depth** — large primary number (metres).
- **NDL** (no-decompression limit, min) when within no-stop; when the ceiling
  demands stops, this flips to **TTS** (time-to-surface) + **ceiling depth**.
- **Ascent-rate bar** — current rate vs the 9 m/min recommended; amber over, red
  well over.
- **Tissue-loading strip** — the 16 ZHL-16C compartments as bars, each vs its
  surfacing M-value (green / amber ≥0.85 / red ≥1.0), the same rendering as the
  explorer's compartment view.
- **DecoStress rank** — a secondary readout. Runs through `predict()` +
  `voidVerdict()` on the profile-so-far. Shows the percentile when trustworthy;
  shows **VOID** (with a one-word reason: OVER CEILING / UNRANKABLE) otherwise.
  Never a probability.
- **Face tint** — the whole face shifts green → amber → red as peak loading
  approaches / exceeds the surfacing limit. Device-native danger signal.

### 2. The console (right) — the control surface

- **Play / Pause** toggle.
- **Speed** — 1× / 4× / 10× real time.
- **Scrub timeline** — drag to any point in the dive; the watch follows.
- **Profile graph** — depth vs time, the diver's current position marked, drawn
  from the same profile the watch reads.

### 3. The demo dive

A fixed, realistic scripted profile (descent → bottom → ascent → 5 m safety stop
→ surface) chosen to stay in-distribution so the rank shows a real number for
most of it. Defined as a polyline constant. Auto-plays on load, loops.

## Data flow

One clock, owned by the console. The playhead is a time `t`. Each frame:
`t → slice the profile polyline up to t → simulatePolyline(slice) → {tissue, st}`
→ the watch renders depth/NDL/TTS/ascent/tissues/tint from that state, and the
rank readout from `predict(slice)` + `voidVerdict`. The page holds no physics; it
reads the engine and draws. NDL and TTS come from `requiredAscentMin` on the
current tissue state (TTS) and a forward search for when the no-stop limit is hit
(NDL).

## Error handling

| Case | Behaviour |
|------|-----------|
| Engine fails to load (`predict` undefined) | The page shows a clear "engine not loaded — serve over http, not file://" banner rather than a blank device. |
| Scrub to t=0 (no dive yet) | Depth 0, NDL ∞, tissues at surface saturation, rank shown as "—" (no dive to score). |
| A scrub position that is over-ceiling | Rank reads VOID; watch face red; TTS/ceiling shown. This is the honesty carry-over working. |
| Canvas 2D context unavailable | Graph area is skipped; numeric readouts still render. |

## Testing

- **Extraction regression:** the full existing suite (76 py + 4 js) passes
  unchanged after the engine moves. This is the primary guard that the physics
  is intact.
- **New `tests/dive_computer_dom.test.js`:**
  - the page boots with zero JS errors and the engine is present (`predict` is a function);
  - at a bottom-of-dive time, the watch shows a positive depth and a finite NDL or TTS;
  - the tissue strip renders 16 bars;
  - scrubbing to an over-ceiling position makes the rank readout show VOID, not a number;
  - the demo dive at a normal in-distribution point shows a numeric rank.
- **New Python guard** (extend `test_web_model_sync.py`): `deco-engine.js` is the
  file that carries the fitted-model constants (`RM`, `RM_Q`, `ZHL`), and
  `index.html` no longer defines them inline — so the drift guards read the engine
  file. This keeps the single-source-of-truth property enforced.

## Non-goals (YAGNI)

- No 3-D, no sound, no gas switching / trimix / multi-gas, no user-editable
  profile on this page (the explorer already does that).
- Not a dive planner; the same "not for real dive decisions" disclaimer applies.
- No new physics — this page only *renders* the existing engine.

## Open questions

None.
