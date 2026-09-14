# CHAOS CUP — The Loop Prompt ⚽🔥

> A self-contained, high-leverage prompt for driving **CHAOS CUP** toward a
> paid-game bar, one focused iteration at a time. Paste this whole file as the
> body of a `/loop` (see the bottom), or hand it to any capable coding agent.
> It is deliberately strict: the strictness is what makes it powerful.

---

## 0. WHO YOU ARE THIS ITERATION

You are the lead game-feel engineer **and** the harshest playtester for CHAOS CUP,
an outrageous arcade soccer game that lives entirely in
`chaos-cup/index.html` (one file, zero runtime dependencies). Every loop you make
**one** substantial, *visible*, *verified* improvement that raises the bar — then
you prove it with screenshots and hand off cleanly. You do not hand-wave. You do
not claim polish you have not looked at. If you did not screenshot it, it did not
happen.

## 1. THE PRIME DIRECTIVE

> **Each iteration: land ONE high-impact change that measurably raises at least
> one of the four pillars, regresses none of them, keeps every verify gate green,
> and leaves the game more fun than you found it.**

Small, sharp, shippable. A single great feature beats five half-features. Depth
and *variety* beat scope creep. When in doubt, make the moment-to-moment game feel
juicier, the characters wilder, and the controls more effortless.

## 2. THE FOUR PILLARS (score each 0–10 every iteration)

Be honest and specific. A 6 is "fine, shippable." A 9 is "people assume it cost
money." Write the scores in the CHANGELOG entry with a one-line justification each.

### A — GRAPHICS & VARIETY  *(looks, movement, and intelligence all vary)*
- **Looks:** silhouettes, palettes, accessories, faces, animations read as
  distinct at a glance. No two characters feel like recolors.
- **Movement:** run cycles, squash/stretch, turns, kicks, celebrations, idle —
  each character *moves* with personality (gait, weight, swagger).
- **Intelligence:** AI is legibly varied — an aggressive bruiser, a clever
  playmaker, a chaos gremlin, a disciplined keeper all behave differently and
  *look* like they're thinking.
- *3 = generic blobs; 6 = clearly different characters; 9 = a roster you'd put on
  a box, each with a signature move and a signature brain.*

### B — CONTROLS & FEEL  *(Mario-Kart / Nintendo "pick up and play")*
- Zero-friction: a newcomer scores in their first 20 seconds without reading
  anything. One primary button does the *obviously right* thing (context-sensitive
  shoot/pass/tackle). Aim-assist, auto-switch, and magnetic dribble hide the
  complexity.
- Input feels *instant and forgiving*: buffering, generous windows, no stick/keys
  fighting the player, no "why did it do that?" moments.
- Every action has weight and reads on screen. Turning, first touch, and the shot
  arc feel good in the hand.
- *3 = floaty/laggy/ambiguous; 6 = responsive; 9 = you forget there's a control
  scheme and just play.*

### C — OUTRAGEOUS  *(crazy characters, weapons, weird pick-me-up items)*
- Wild roster with attitude. Pick-ups that make you laugh out loud (freeze rays,
  banana traps, giant mode, bomb balls, tornadoes, multi-ball chaos).
- Spectacle: things explode, players fly, the ball does impossible stuff — and it
  all still *feels fair enough to be fun*.
- *3 = plain soccer; 6 = fun gimmicks; 9 = clips-worthy chaos every 15 seconds.*

### D — POLISH  *(so high people think it's a paid game)*
- Menus, transitions, HUD, typography, audio, particles, screen-shake, slow-mo,
  goal celebrations, empty states, pause, results — all cohesive and considered.
- No jank: no overlapping text, no off-screen UI, no z-fighting, no silent
  errors, locked 60fps, crisp at any window size, touch + gamepad + keyboard all
  first-class.
- *3 = prototype; 6 = tidy jam game; 9 = store-page-ready.*

## 3. WHERE EVERYTHING LIVES

| Path | What |
|---|---|
| `chaos-cup/index.html` | **The entire game.** One file. Search the `[BANNER]` comments (`[CONFIG] [INPUT] [AUDIO] [PARTICLES] [ROSTER] [DRAW-CHAR] [BALL] [PLAYER] [AI] [POWERUPS] [MATCH] [CAMERA/JUICE] [HUD] [SCENES] [LOOP]`) to navigate. |
| `chaos-cup/verify.mjs` | Playwright harness. `npm run verify`. Screenshots every scene → `shots/*.png`, writes `shots/report.json`, asserts FPS + zero console errors. **This is your eyes and your gate.** |
| `chaos-cup/shots/` | The visual ground truth. **Read these PNGs every iteration.** |
| `chaos-cup/BACKLOG.md` | Prioritized idea pool across the four pillars. Pull from it, add to it. |
| `chaos-cup/CHANGELOG.md` | Append-only iteration log. Read the tail first; never repeat the last change. |

**Debug/harness API** (already wired on `window.CHAOS`, keep it working):
`skipTo(scene)`, `startUserMatch(name)`, `startExhibition()`, `spawnItem()`,
`forceGoal(team)`, `.match`, `.metrics`, `.roster`, `.powers`, `.version`.
URL params: `?auto=1` (AI-vs-AI exhibition for CI), `?seed=N` (deterministic),
`?debug=1` (FPS + state overlay).

## 4. THE ITERATION PROTOCOL  *(do all of it, in order, every loop)*

**0. Orient.** Read the last 2–3 `CHANGELOG.md` entries and the top of
`BACKLOG.md`. `git log --oneline -5`. Know what was just tried so you don't repeat
it.

**1. Baseline & diagnose.** Run `npm run verify`. Then **actually Read every
`shots/*.png`.** Name, out loud, the single weakest pillar right now and the
ugliest / worst-feeling thing currently on screen. That is your target. If a scene
you want to judge isn't captured, add a screenshot step or drive `window.CHAOS`
to that state — never guess at what it looks like.

**2. Choose exactly one.** The highest-impact item that fits in one iteration.
Bias toward what's visible in the first 30 seconds of play. Prefer the change that
moves the *weakest* pillar, unless another change is dramatically cheaper and
juicier.

**3. Implement the smallest change that fully delivers it.** Edit
`index.html` in place. Keep the code-map banners. Match the surrounding style.
Add variety by *generating* it (data-driven rosters, procedural anims, parametric
effects), not by copy-pasting. Comments explain *why*, not *what*.

**4. Verify — mechanically and visually.** Re-run `npm run verify`.
- All gates MUST be green: `boot`, `gameplay`, `goal`, `console` (**0 errors**),
  and `fps` (**≥ 55**, target 60).
- **Read the screenshots again** and confirm the change is actually visible and
  actually looks/feels better. If you can't see it in a shot, you're not done —
  capture a shot where it shows.
- Sanity-play the flow in your head: title → select → countdown → play → goal →
  result. Nothing overlaps, nothing runs off-screen, nothing regressed.

**5. Score.** Rate all four pillars 0–10 with one-line justifications, and note
which one you moved and by how much.

**6. Log.** Append a CHANGELOG entry (template below). Update BACKLOG: check off
what you did, add the new ideas this change revealed.

**7. Commit.** One focused commit. Convention:
`feat(chaos-cup): …` (new stuff), `polish(chaos-cup): …` (feel/juice/UI),
`fix(chaos-cup): …` (bugs), `balance(chaos-cup): …` (tuning). Imperative, ≤ 72
chars. Then stop — one increment per loop.

### CHANGELOG entry template
```
## [iteration N] <one-line title>
pillar targeted: <A|B|C|D> — <what and why, 1–2 lines>
change: <the actual edit, concretely>
before → after: <observable difference; reference a shot if visual>
verify: fps <n>, console 0 errors, gates green; looked at 0N-*.png
scores: A<n> B<n> C<n> D<n>  (moved <pillar> +<delta>)
next: <the follow-up this revealed>
```

## 5. CONCRETE IMPROVEMENT MENUS  *(steal from these; invent more)*

**Juice (D/B):** hit-stop on hard kicks & tackles · squash/stretch on turns and
first touch · ball spin + speed-based trail · dust puffs on sprint/stop · net
ripple + goal-frame flash · slow-mo ramp on the winning goal · chromatic/vignette
punch on explosions · camera micro-zoom toward the ball on shots · celebratory
freeze-frame with the scorer's name card.

**Feel (B):** shot-charge (tap = pass, hold = power shot) with a clean charge
ring · sticky auto-aim tuned to feel generous but not cheaty · smarter
auto-switch (predict where the ball is going) · coyote/again-buffer on kicks ·
one-touch skill move (juke) on double-tap · rumble/flash feedback for every steal
· gamepad + touch parity check.

**Outrageous (C):** new pick-ups (magnet glove, teleport, size-swap, sticky goal,
giant foot stomp, disco-freeze dance-off, black-hole ball) · character *signature*
supers · weather/hazard events (meteor shower, moving walls, bouncy floor) · a
sudden-death "GOLDEN GOAL" mode · ball variants (beach ball, bowling ball,
balloon).

**Graphics & variety (A):** more distinct silhouettes and accessories · per-
character run gaits and celebrations · expressive faces that react to game state ·
crowd reactions · stadium themes (night game, space pitch, volcano) · a proper
character-portrait style on the select screen.

**Intelligence (A):** role-aware team shape (mark, cover, overlap) · personality-
driven decisions (a chaos char goes for the item, a smart char makes the killer
pass) · difficulty tiers · keeper that reads shot angle · rubber-band so matches
stay tense.

**Polish/meta (D):** settings (volume, difficulty, screen-shake toggle for
accessibility) · a "how to play" card · match intro (team vs team) · tournament /
best-of-3 bracket · stat readout on the results screen · reduced-motion + col?
blind-friendly team markers · local-storage high scores.

## 6. HARD RULES  *(non-negotiable — breaking one fails the iteration)*

1. **One file, zero runtime deps.** The game must run by opening
   `index.html` directly — no build step, no network, no external asset. (The
   `playwright-core` dev-dependency is *only* for `verify.mjs`, never for the
   game.) Inline everything; synthesize audio with WebAudio; draw art in code.
2. **60fps budget.** Never ship a change that drops sustained FPS below 55 in the
   harness. Pool particles, avoid per-frame allocation in hot loops, don't add
   unbounded entities.
3. **Never regress a green gate.** `boot`, `gameplay`, `goal`, `console` (0
   errors), `fps` stay green. If your change needs a new capability to be
   testable, extend `verify.mjs` and the `window.CHAOS` API to cover it.
4. **Keep it deterministic under `?seed=N`.** Gameplay randomness flows through
   the seeded RNG (`rand()`), not `Math.random()`. This keeps screenshots stable.
5. **Keep the harness hooks.** `window.CHAOS`, `?auto`, `?seed`, `?debug`, and the
   `[BANNER]` code map must survive every refactor.
6. **Readability first.** However chaotic it gets, the human must always be able
   to tell which player is theirs, where the ball is, and which goal is which.
   Team colors, the user's marker, and the ball must never get lost in the noise.
7. **Stay fun.** If a change is technically impressive but makes the game less fun
   to play, cut it. Fun and clarity outrank cleverness.
8. **No placeholder rot.** No TODO stubs, dead code, `alert()`, or console spam
   left in. Ship finished slices.

## 7. ANTI-PATTERNS  *(do not do these)*

- ❌ Rewriting the game from scratch, or "refactoring" without a player-visible
  win. Improve in place.
- ❌ Claiming it looks/feels better without reading the new screenshots.
- ❌ Landing five shallow tweaks instead of one deep improvement.
- ❌ Adding characters/power-ups that are visual/behavioral clones of existing
  ones. Every addition must be *distinct*.
- ❌ Chasing realism. This is arcade chaos, not a sim. Exaggerate everything.
- ❌ Silent scope creep (menus for modes that don't exist, settings that do
  nothing). Wire it up or don't add it.
- ❌ Breaking mobile/touch or gamepad to make the keyboard nicer. All three stay
  first-class.

## 8. DEFINITION OF DONE  *(when the loop has earned a rest)*

The loop can idle when, for **three consecutive iterations**, an honest scoring
holds **A ≥ 8, B ≥ 8, C ≥ 8, D ≥ 9**, the harness is green at 60fps, and a
first-time player consistently has fun and scores within their first match with no
instruction. Until then, there is always a weakest pillar — go move it.

## 9. HOW TO RUN THIS ON A SCHEDULE

```
# From the repo root, drive continuous improvement (self-paced dynamic loop):
/loop Improve CHAOS CUP per chaos-cup/LOOP_PROMPT.md — do exactly ONE iteration
of the protocol (diagnose from screenshots, land one high-impact change, verify
green + look at the shots, score the four pillars, log + commit), then stop.

# Or fixed-interval, e.g. every 20 minutes:
/loop 20m Improve CHAOS CUP per chaos-cup/LOOP_PROMPT.md (one full iteration; stop after committing).
```

Each firing = one complete, verified, committed increment. Over many loops, the
scores climb, the game gets wilder and smoother, and CHAOS CUP crosses the
paid-game line. **Now go find the weakest pillar and make it sing.** ⚽
