# CHAOS CUP — Improvement Backlog

Prioritized idea pool for the loop (see `LOOP_PROMPT.md`). Pull the highest-impact
item that fits one iteration, then check it off and add what the change revealed.
Tags: **[A]** graphics/variety · **[B]** controls/feel · **[C]** outrageous ·
**[D]** polish.

Priority: 🔴 do-next / known gap · 🟠 high value · 🟡 nice-to-have.

---

## 🔴 P0 — known gaps (do these first)

- [x] **[A] Team spacing / formation shape.** ✅ *iter 2* — slot-based lanes +
  staggered depth so supporters fan out instead of clumping.
- [ ] **[B] Shot charge.** Tap = pass / placed shot, hold = power shot, with a clean
  charge ring on the user player and a satisfying release. The single biggest
  feel upgrade available.
- [ ] **[A] Per-character celebrations & signature move.** Right now everyone
  celebrates the same. Give each archetype a distinct celebration and one signature
  on-pitch flourish so the roster feels authored.
- [ ] **[D] Settings panel.** Volume slider, difficulty, and a screen-shake /
  reduced-motion toggle (accessibility). Wire them to real state.
- [ ] **[C] Two more marquee power-ups** with unique spectacle (e.g. TELEPORT to the
  ball, BLACK-HOLE ball that warps nearby players). Must look/behave distinct from
  the existing 12.

## 🟠 P1 — high value

- [ ] **[B] Predictive auto-switch.** Switch the user to the player who will *reach*
  the ball first, not merely the nearest — fewer "controlling the wrong guy"
  moments.
- [ ] **[B] One-touch juke** on double-tap of a direction (short burst + tight turn),
  with a dust puff. Skill ceiling without new buttons.
- [ ] **[A] Keeper intelligence.** Read the shot angle and commit to dives; parries
  and catches with a "SAVE!" beat. Make the keeper feel alive.
- [ ] **[A] Difficulty tiers + light rubber-banding** so matches stay tense to the
  final whistle.
- [ ] **[D] Match intro & bracket.** "TEAM A vs TEAM B" splash; optional best-of-3 /
  4-team tournament with a bracket screen.
- [ ] **[D] Results screen depth.** Possession %, shots, biggest hit, MVP with a
  portrait — reads like a real post-match screen.
- [ ] **[C] Hazard events.** Timed chaos: meteor shower, moving walls, bouncy floor,
  sudden GOLDEN GOAL overtime.
- [ ] **[A] Stadium themes.** Night game (floodlights), space pitch, volcano —
  swap palette + crowd + backdrop per match.

## 🟡 P2 — nice-to-have

- [ ] **[D] Local high-scores / longest win streak** via localStorage.
- [ ] **[C] Ball variants** (beach ball floaty, bowling ball heavy, balloon).
- [ ] **[A] Crowd character variety** (banners, wave, pitch invasion on a big win).
- [ ] **[B] Remappable keys + full gamepad glyphs** in the hint line.
- [ ] **[D] Colorblind-friendly team markers** (shape as well as color).
- [ ] **[D] Subtle dynamic music** that intensifies in the last 15 seconds.
- [ ] **[C] Announcer voice-style callouts** (synth "vox" chirps tied to events).

---

## ✅ Done (mirrored in CHANGELOG.md)

- [x] **[iter 0]** v1 scaffold: full game — 16 procedural characters, one-button
  aim-assisted controls, auto-switch, magnetic dribble, 12 power-ups + banana trap,
  personality AI, keeper, title parade, character select, countdown, slow-mo goals,
  confetti, screen-shake, synth audio, HUD, pause, results, touch + gamepad support.
- [x] **[iter 1]** Screenshot bug-sweep: fixed double-applied giant/tiny scale,
  stopped AI wasting ball-only power-ups (announcement spam), fit all 16 characters
  in the select grid, moved toasts to a bottom stack, hid the HTML hint during play.
- [x] **[iter 0]** Playwright verification harness (`verify.mjs`) + `window.CHAOS`
  debug API + `?auto/?seed/?debug` params.
