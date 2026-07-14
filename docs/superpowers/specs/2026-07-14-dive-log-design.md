# Dive Log — Design Spec

**Date:** 2026-07-14
**Component:** `decostress_app/index.html` (fourth tab)
**Status:** approved, ready for implementation plan

## Purpose

Save dive scenarios built in the Risk explorer, and review them together: compare
them in a table, overlay their profiles on one chart, and see why each one scored
as it did.

This is a **scratchpad for scenarios you construct**, not a record of real dives
you performed. It makes no claim to be a personal diving record, and it collects
no physiology (there is nothing to fit it to — see the honesty rule below).

## Non-goals (YAGNI)

- **Not** a log of real dives (no site, buddy, gas mix, water temp, equipment).
- **Not** reopening a saved scenario back into the explorer to re-fly or edit it.
  Considered and explicitly cut.
- **Not** a shared or synced log. Storage is local to the browser.
- **No** physiology fields. The real data has no such columns; a field we cannot
  fit is a field that invites an invented conclusion.

## Architecture

A fourth tab (`Dive log` → panel `p-log`) in the existing single-file app. No
backend, no build step, no new dependencies.

### Data model — store inputs, derive outputs

A saved scenario is:

```js
{
  id:      "s_1721030400000_3f2a",   // stable, unique
  name:    "45 m / 18 min",           // editable, defaulted from the profile
  savedAt: 1721030400000,             // epoch ms
  poly:    [{t:0,d:0}, {t:2.5,d:45}, {t:18,d:45}, {t:21.4,d:0}]
}
```

**That is the whole record.** Model rank, decompression obligation, Bühlmann
loading, violation flag, deficit and driver contributions are **all re-derived at
render time** by calling the same `predict()` and `simulatePolyline()` the
explorer uses.

This is the load-bearing decision. Caching the derived scores would freeze them:
refit the model and every saved dive would keep serving a stale number, with
nothing to catch it. That is exactly the drift that produced Correction 10 (a
hand-copied coefficient diverging across four files) and exactly what
`tests/test_web_model_sync.py` now guards against everywhere else. The log must
not reintroduce it one screen over.

Storing only the polyline also means saved scenarios automatically re-score when
the model or the physics changes — which is the behaviour you want.

### Storage

- `localStorage`, key `decostress.scenarios.v1` (versioned; unknown versions are
  ignored rather than crashing).
- **Export** → downloads `decostress-log.json`.
- **Import** → merges a JSON file back in, by `id`, skipping duplicates.

Export/import exists because `localStorage` is scoped to the exact origin:
`file://` and `http://localhost:8899` are *different* origins with *different*
logs. Without export, opening the app a different way silently shows an empty
log and looks like data loss.

Corrupt or unparseable storage must not brick the page: it is logged, discarded,
and the log starts empty.

## Features

### 1. Saving

A **Save to log** button in the explorer's post-dive bar (alongside
*Scrub / replay* and *Edit & re-analyse*), enabled once the diver has surfaced
and a profile exists. It seeds the name from the profile (e.g. `45 m / 18 min`)
and the user may edit it.

### 2. Comparison table

One row per scenario, sortable by any column:

| Column | Source |
|---|---|
| Name | stored |
| Max depth | `predict().depthFsw` → metres |
| Bottom time | `predict().bottom` |
| Ascent taken | `predict().actualAscent` |
| Deco obligation | `predict().requiredAscent` |
| Deficit | `predict().deficit` (minutes of obligation skipped) |
| Model rank | `predict().pct` — **or `VOID`**, see the honesty rule |
| Bühlmann | `simulatePolyline().st` (× the surfacing limit) |
| Status | `OK` / `DECO VIOLATION` |

Rows for voided dives are visually distinct (red tint).

### 3. Overlay chart

All saved profiles drawn on one depth/time axis, colour-coded, with a legend.
Voided profiles drawn in the violation colour. This is what makes the effect of a
safety stop visible as a *shape*, not just a number.

### 4. Driver breakdown

Select a scenario → show the standardised log-odds contribution of each of the
three fitted features (max depth, bottom time, deco obligation), the same
decomposition the explorer's assessment shows.

For a **voided** dive, the breakdown is replaced by the refusal: the model has
never seen a dive that skipped its obligation, so no contribution is shown.

## The honesty rule (load-bearing)

**A voided dive has no model rank. The table shows `VOID`, never a number.**

A sortable table is precisely where the bug found in use would creep back. The
explorer already voids its dial on a decompression violation — because the fitted
model's ascent coefficient is positive, so on a dive that skipped its deco it
would call a *faster* ascent *safer*. If the log printed `42nd` in a tidy column
next to legitimate dives, that number would read as comparable, and voiding the
dial would have been pointless.

Therefore:

- Voided dives render `VOID` in the rank column.
- Sorting by rank **groups voided dives separately** (they sort to the end, as a
  block), rather than interleaving them as though their rank were meaningful.
- The driver breakdown for a voided dive shows the refusal, not contributions.
- No probability is ever displayed. The rank is a percentile within the 1,948
  real NMRC 99-02 dives; nothing more.

## Error handling

| Case | Behaviour |
|---|---|
| `localStorage` unavailable (private mode, disabled) | Log works in-memory for the session; a banner says it will not persist. |
| Corrupt / unparseable stored JSON | Discarded, logged to console, log starts empty. Never a blank page. |
| Import file is not valid JSON, or wrong shape | Rejected with a message. Existing log untouched. |
| Import contains an id already present | Skipped (no duplicate, no overwrite). |
| Scenario with < 2 waypoints | Rejected at save time; `predict()` cannot score it. |
| Empty log | Table and chart show an explanatory empty state, not a broken layout. |

## Testing

Extend `tests/test_web_model_sync.py` (static guards on the shipped HTML):

- The log stores `poly`, and does **not** persist derived fields (`pct`, `score`,
  `requiredAscent`) into `localStorage`. Guards the drift rule.
- A voided dive can never render a numeric rank in the table.
- The storage key is versioned.

Add jsdom behavioural tests:

- Save two scenarios → both appear in the table.
- A dive that skips its obligation renders `VOID`, not a percentile.
- Sorting by rank does not interleave voided dives among valid ones.
- Export produces JSON containing only `{id, name, savedAt, poly}`.
- Import of that JSON round-trips to the same table.
- Corrupt `localStorage` does not throw and yields an empty log.

## Open questions

None.
