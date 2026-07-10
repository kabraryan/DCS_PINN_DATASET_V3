# What this thing actually does — in plain English

No maths. No jargon. If a word has to be technical, it gets explained the first time.

---

## The problem

Divers breathe air under pressure. Pressure squeezes nitrogen out of the air and into the
body — into blood, fat, muscle, everything. The deeper you go and the longer you stay, the
more nitrogen goes in.

When a diver comes back up, the pressure drops and the nitrogen wants to come back out. If it
comes out slowly, it leaves through the lungs and nothing happens. If it comes out too fast,
it forms bubbles inside the body.

That is **the bends** — decompression sickness. It ranges from aching joints to paralysis to
death.

The question everyone wants answered: **can you tell in advance who is going to get it?**

---

## The two piles of data

### Pile 1: 50,000 made-up divers

Someone wrote a computer program to invent 50,000 divers, and to decide which of them got hurt.

The program decided using a rule that a human typed in. Roughly: *deep dive, add points; older
diver, add points; lots of points means hurt.*

Then people trained prediction machines on these invented divers. The machines did brilliantly.

**Of course they did.** They were rediscovering the rule a human had typed in. It's like hiding
an Easter egg, finding it, and announcing you've discovered where eggs come from.

We measured exactly how brilliant. The theoretical maximum score on this data is **0.935**
(where 0.5 means coin-flipping and 1.0 means perfect). A tiny, simple machine looking at **one
number** scored **0.922**. A large, sophisticated machine looking at 37 numbers scored **0.928**.

The sophisticated machine was worth **six thousandths**. On this pile, the choice of machine
does not matter, because the answer was written on the back of the card.

**Use for this pile:** checking that your plumbing works. Not for learning anything true.

### Pile 2: 2,700 real Navy divers

Between 1944 and 1997, the US Navy took thousands of real people diving and recorded who got
hurt. Real dives, real injuries, nobody making up the answers.

**This is the pile that can teach us something**, because nobody hid an egg in it.

The project's own plan said, in writing: *"There is no real dive-outcome data and none will be
available."* That was false. The file was on the desk the whole time.

---

## The hidden trap in the real data

Those 2,700 dives came from **38 separate Navy experiments**. The experiments were wildly
different. In one, 4 divers in 100 got hurt. In another, 35 in 100 did — because that
experiment was deliberately punishing.

Now picture a lazy machine. It learns nothing about diving. It learns:

> *"This dive is from Experiment 12. Lots of people got hurt in Experiment 12. I'll say hurt."*

That machine scores beautifully on a test and is worthless on a new diver, because a new diver
isn't from Experiment 12. It's the student who noticed every answer on last week's quiz was "B."

We measured the trap. The lazy machine scores **0.77**. The honest machine scores **0.70**. That
seven-point gap is pure cheating.

**So the benchmark hides the experiment name.** Train on 30 experiments; test on 8 the machine
has never seen. Memorising stops helping.

---

## What the benchmark actually is

Not a tool that tells a diver what to do. **A referee.**

Scientists have competing theories about why the bends happens. The referee lines each theory up
against 2,700 real divers and asks one question:

> *"Did you actually predict who got hurt — or are you just talking?"*

Each theory has to produce two numbers for every dive.

**"How much decompression did you owe?"** Every theory can say how long a diver should have
paused on the way up. Compare that to how long they actually paused. If they owed 20 minutes and
took 5, they're 15 minutes short. *(We call this the* deficit*.)*

**"How overloaded did the diver get?"** How far past the safe limit did the nitrogen push?
*(We call this the* risk index*.)*

Then we check whether either number actually lines up with who got hurt.

---

## The four theories on trial

**Raw numbers.** Not a theory at all — just how deep, how long, how fast up. This is the
**number to beat**. Any theory that can't outperform three plain measurements has not earned
its physics.

**Bühlmann.** The classic. Models the body as 16 imaginary tissues, each soaking up nitrogen at
its own speed — blood fast, bone slow. Warns when any tissue is overloaded. This is what most
dive computers run.

**Bühlmann with gradient factors.** The same, but more cautious. Instead of "don't cross the
line," it says "don't get closer than 70% of the way to the line." The modern default.

**The bubble model.** Actually simulates a single microscopic bubble growing and shrinking
inside the diver, minute by minute, and reports how big it got.

---

## Why the referee is built to be hard to fool

Because during this project, we fooled ourselves **twice**, and both times every check we had
said the result was real.

### It hides the team jersey
Test only on experiments the machine has never seen. Erases a fake 7-point head start.

### It shuffles the answers
Scramble who-got-hurt into random nonsense and re-run everything. A machine trained on nonsense
**must** score 0.5 — a coin flip. Ours scores 0.507. That proves our ruler is straight, not
bent.

### It checks the arrow points the right way
Once, the bubble theory "predicted" the bends **backwards** — the more the bubble grew, the
*fewer* injuries. It wasn't finding bubbles at all. It had quietly noticed *"this was a deep,
short dive."*

That backwards result passed the size check. It passed the statistics check. It passed the
random-shuffle check. It was **still wrong.**

So now: if a theory predicts the bends backwards, it is thrown out, however good its score.

### It asks the question twice, two different ways
The old Navy records only kept **three numbers per dive**: how deep, how long, how fast up. They
never wrote down the *shape* of the ascent. So we rebuild the missing shape two different ways —
a straight climb, and a climb with pauses — and we only believe an answer if **both** agree.

We know this matters, because the answer *changes* depending which we use. Our biggest weakness,
turned into our best guard dog.

### It writes the answer down before running
The plan says, in advance: **no theory will pass.** If we're right, that's an honest result. If a
theory *does* pass, we've promised to attack it before celebrating.

That's the difference between finding something and fooling yourself.

---

## What we already know it will say

The bubble physics **doesn't help.**

Three plain numbers — depth, time, ascent speed — beat every clever physics idea we tried. We
checked this **four completely different ways**, and got the same answer each time. Feeding the
physics to a trained machine made it **worse**, not better.

That is not a failure. **Finding out that something doesn't work is a real discovery**, and it's
the kind that rarely gets published. It only counts because we checked it honestly.

---

## Why this will never be an app for divers

You asked whether this could help real recreational divers. We did the arithmetic. The answer
is **no**, and it isn't close.

The bends is **rare** — very roughly 1 dive in 2,000.

Take our best machine, scoring 0.71, and set it to catch 80% of real injuries. Here is what
happens in the water:

- It shouts **DANGER on half of all dives.**
- Of every **1,309 warnings, 1,308 are false alarms.** Nobody listens after a week.
- And of the dives it calls **safe**, one in five real injuries is hiding in there.

A machine that tells a diver *"you're fine"* just before they get hurt is **worse than no
machine at all.**

This cannot be fixed by trying harder. Even a machine far better than anything this data can
produce — 0.90 — is still wrong 414 times out of 415. When the thing you're hunting is that
rare, **even an excellent guesser is wrong nearly every time it speaks.**

To do it properly you would need roughly **200,000 real recreational dives** with recorded
outcomes. We have **419 injuries**, from Navy tests deliberately designed to injure people so
the limits could be found. Those divers were young, screened, fit military personnel doing deep
single dives on air. The file records nothing about age, temperature, exertion, or the heart
defect that a quarter of the population has and that drives much real-world risk.

So the honest answer: **a diver's dive computer already does this, and does it better.** It was
built from this kind of data plus decades more, and it gives a *schedule*, not a fake percentage.

**This benchmark never prints a probability. That is enforced in the code, not just promised.**

---

## The one lesson, if you remember nothing else

Every mistake in this project had the same cause.

Someone wrote something that *looked* right, and nobody ran it.

- A typo in a physics table sat there for months. A test was **guarding the typo.**
- A document was corrected, and the correction invented a number that exists in **no file
  anywhere.**
- A bubble simulation was designed, specified, and planned in 1,500 lines — and the bubble it
  simulated **could never grow, on any dive, ever.**

That last one took **twenty lines of code** to expose. Twenty lines, a few seconds to run.

They just had to exist *before* fifteen hundred lines were built on top of a bubble that was
never going to grow.

That's the whole thing. **Stop believing what looks right. Run it.**
