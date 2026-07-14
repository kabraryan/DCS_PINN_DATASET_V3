# DecoStress — the explorer

A single self-contained `index.html`. Open it in a browser; no build, no server.
(One external dependency: three.js from a CDN, for the 3D diver. Everything else
is inline.)

## What it shows

**The physics is real.** Genuine Bühlmann ZHL-16C — all 16 compartments, the
published *a* and *b* coefficients, real M-values. Not a simplification. It also
runs the ceiling-driven ascent scheduler ported from `benchmark/profile.py`, so
the "deco obligation (TTS)" readout is a real time-to-surface.

**The prediction is real, and weak.** The dial is a logistic fitted to 1,948 real
NMRC 99-02 dives (305 DCS cases). Its honest out-of-sample skill is **AUC 0.64 ±
0.05** from nested grouped CV. It emits a **percentile rank**, never a
probability.

## The one thing you must not break

The fitted model's ascent-time coefficient is **positive (+0.537)**. Fed an ascent
time the *diver chose*, it calls a fast ascent safer — it ranks a 40 m / 8 min
bounce straight to the surface above a by-the-book 18 m / 40 min dive with a
safety stop. Read as advice, it tells you to skip your safety stop.

That is not a bug in the fit. It is confounding: in the Navy trials `ascent_time`
was the **prescribed decompression schedule**, so the model learned "long ascent
⇒ provocative dive". True as a marker of severity. Lethal as advice.

**So the app never feeds it a diver-chosen ascent.** It computes the
decompression *obligation* from the tissue state and scores that — the same
quantity the training data actually contained. This makes the model monotonic
(deeper or longer always raises the rank; verified over 103 checks) and
ungameable (the diver's ascent cannot move the score at all).

And if a dive **skips** its obligation, the model **refuses to score it**. No
diver in the training set did, so it is pure extrapolation, and the positive
coefficient would wrongly call it safe. The app defers to the physics instead:
you are over your ceiling.

`tests/test_web_model_sync.py::test_ascent_coefficient_is_positive_so_the_obligation_guard_must_exist`
enforces all of this. If you rewire the app to score the diver's ascent, it fails.

## Physiology is inert, on purpose

The PFO / hydration / sleep / fitness / age controls **do nothing**, and the app
says so when you click them. The real dive records contain depth, bottom time,
ascent time and outcome — and nothing else. There is no column to fit. Anything
the app told you about physiology would be invented.

## The numbers are generated, not hand-written

The constants in the `RM`, `RM_Q` and `ZHL` blocks come from:

```bash
python -m scripts.export_web_model
```

They are pasted into `index.html`. **That is a drift path** — the same one that
produced Correction 10, where compartment 16's b-coefficient was hand-copied into
four files, diverged, and reached a README, two design docs, and a unit test that
asserted the bug.

So it is guarded. `tests/test_web_model_sync.py` refits from the real CSV and
asserts the HTML still agrees — the table, the coefficients, the quantiles, the
cohort size, and the no-probability rule. If you change the model, that test tells
you the app is stale.

## What this is not

Not a dive planner. It emits no probability. Every profile you can build here sits
far outside the training distribution — those Navy dives averaged 127 fsw for 61
minutes. Read the rank as *"how provocative is this exposure, relative to dives
that were designed to provoke DCS"* — and nothing more.
