# CHAOS CUP — Iteration Log

Append-only. Newest at the bottom. One entry per loop iteration (see
`LOOP_PROMPT.md` §4 for the template). Scores are honest 0–10 per pillar:
**A** graphics/variety · **B** controls/feel · **C** outrageous · **D** polish.

---

## [iteration 0] v1 scaffold — a game worth polishing
pillar targeted: all four — stand up a genuinely playable, already-juicy base so
the loop has real ground to improve, not an empty canvas.
change: single-file `index.html` (zero runtime deps) implementing:
- **[A]** 16 procedurally-drawn characters (distinct bodies, palettes, 15+
  accessory types, expressive faces) with per-gait run cycles, squash/stretch,
  kick wind-ups and celebrations; personality-driven AI (chaser/support/defend
  roles + keeper).
- **[B]** one-button context-sensitive kick (shoot/pass) with aim-assist toward
  the goal and lead passing, magnetic dribble, automatic player-switch, input
  buffering; keyboard + gamepad + on-screen touch stick/buttons.
- **[C]** 12 power-ups + banana trap (rocket boots, magnet, giant, freeze ray,
  shrink, bomb ball, multi-ball, ghost, lightning, tornado, homing ball), item
  crates, spectacle on every activation.
- **[D]** animated title with character parade, character select with stat cards,
  3-2-1 countdown, slow-mo goal celebrations, confetti, screen-shake, hit-punch
  zoom, WebAudio-synth SFX + music, scoreboard/timer/power-up HUD, pause + results.
- harness: `verify.mjs` (Playwright) screenshots every scene, asserts FPS + zero
  console errors; `window.CHAOS` debug API; `?auto/?seed/?debug` params.
before → after: nothing → a complete match loop that runs at 60fps.
verify: fps 60, console 0 errors, all gates green; captured 01–05 shots.
scores: A6 B6 C7 D7
next: bug-sweep from the first real screenshots.

## [iteration 1] screenshot bug-sweep — trust the pixels
pillar targeted: D (polish) + A — the first screenshots exposed concrete jank.
change:
- **fix** giant/tiny scale was applied twice (`p.r` already encodes it *and*
  `drawCharacter` re-multiplied `scaleMod`) → giants rendered ~3.6× instead of
  1.9×. Now scale once.
- **fix** AI activated ball-only power-ups (homing/bomb) with no possession,
  wasting them and spamming "NEED THE BALL!" toasts. `maybeUsePower` now gates
  ball-dependent items on ownership and offensive items on a rival in range.
- **[D]** character-select grid ran off the bottom (16 cards, 4 rows, last row
  below the viewport) → recompacted to a fully-visible 4×4.
- **[D]** announcements stacked over the center and collided with the goal banner
  → small toasts now stack upward from the lower third; the big goal banner owns
  the center.
- **[D]** the HTML control-hint bled over the pitch during play → hidden outside
  the menus.
before → after: compare `02-select.png` (all 16 now visible) and `03-gameplay.png`
(correct character sizes, clean bottom toasts, unobstructed pitch).
verify: fps 60, console 0 errors, all gates green; re-read 02 & 03 shots.
scores: A7 B7 C8 D8  (moved D +1, A +1)
next: P0 backlog — team spacing so AI stops clumping, then shot-charge for feel.

## [iteration 2] team shape — stop the clump
pillar targeted: A (intelligence) + readability — outfield AI converged on the ball
so the pitch looked like a scrum and passing lanes vanished.
change: gave each outfielder a stable `slot` (0..2); supporters now hold distinct
vertical lanes and staggered depth (`laneY`/`laneBias`) when attacking or defending,
instead of all chasing `Ball.y`. Chaser logic unchanged, so pressing still happens.
before → after: `03-gameplay.png` — players fan across the full width into a real
team shape; only a natural contest remains around the ball.
verify: fps 60, console 0 errors, all gates green; re-read 03-gameplay.png.
scores: A8 B7 C8 D8  (moved A +1)
next: shot-charge (tap pass / hold power shot) for the biggest feel win, then a
settings panel with a reduced-motion toggle.
