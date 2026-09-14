# CHAOS CUP ⚽🔥

**Outrageous arcade soccer** — 16 wild characters, 12+ absurd power-ups, one-button
Nintendo-easy controls, and paid-game-grade juice. Single self-contained HTML file,
**zero runtime dependencies**. Just open it and play.

> This game is built to be improved by an autonomous loop. The design brief and the
> strict iteration protocol live in **[`LOOP_PROMPT.md`](./LOOP_PROMPT.md)** — that
> prompt is the real deliverable; this game is the substrate it polishes.

## ▶️ Play

Open `index.html` in any modern browser. That's it — no build, no server, no assets.

Or with a tiny static server (nice for mobile testing on your LAN):

```bash
cd chaos-cup && npm run serve   # → http://localhost:8123
```

### Controls
| | Keyboard | Gamepad | Touch |
|---|---|---|---|
| Move | WASD / Arrows | Left stick / D-pad | Left thumb-stick |
| Kick (shoot/pass, aim-assisted) | Space | A | **KICK** button |
| Use power-up | Shift | B / X | **POWER** button |
| Switch player | Tab | Y / bumpers | (auto) |
| Pause | Esc / P | Start | — |
| Mute | M | — | — |

**It's designed so you score in your first 20 seconds.** One button does the right
thing: near the ball it shoots (curving toward goal) or threads a pass; away from
it, it tackles. The game auto-switches you to the most useful player.

## 🎮 What's in it
- **16 procedurally-drawn characters** — El Toro, Robo-9000, Granny Slam, Yeti, Zap,
  Cap Cactus, Lava, Disco, Chomp, Wizbeard, Shadow, Rex, Sir Kicks, Bolt, Prof.
  Honk, Frostbite — each with its own look, gait, and AI personality.
- **12 power-ups + banana traps** — 🚀 rocket boots, 🧲 ball magnet, 🍄 giant mode,
  ❄️ freeze ray, 🐜 shrink ray, 💣 bomb ball, ⚽ multi-ball, 👻 ghost dash, ⚡
  lightning, 🌪️ tornado, 🎯 homing ball, 🍌 banana slip.
- **Full juice** — slow-mo goals, confetti, screen-shake, hit-punch zoom, particle
  bursts, ball trails, squash & stretch, synthesized audio + music (WebAudio).

## 🧪 Verify (dev only)

The loop's "eyes." Launches the game in headless Chromium, screenshots every scene,
and asserts 60fps + zero console errors.

```bash
cd chaos-cup
npm install          # dev-only: playwright-core (the game itself needs nothing)
npm run verify       # → shots/*.png + shots/report.json, exits non-zero on any failure
```

`verify.mjs` uses the pre-installed Chromium (`CHROMIUM_PATH`, default
`/opt/pw-browsers/chromium`). Flags: `--headed`, `--keep-open`, `--url <file|http>`,
`--shots <dir>`.

## 🔁 Improve it with a loop

```
/loop Improve CHAOS CUP per chaos-cup/LOOP_PROMPT.md — one full iteration
(diagnose from screenshots, land one high-impact change, verify green + look at the
shots, score the four pillars, log + commit), then stop.
```

Each firing lands one verified, committed improvement. See `LOOP_PROMPT.md` for the
four pillars and the rules, `BACKLOG.md` for the idea pool, and `CHANGELOG.md` for
the running log.

## 📁 Structure
```
chaos-cup/
├─ index.html      # the entire game (open this)
├─ LOOP_PROMPT.md  # ⭐ the powerful loop prompt — the deliverable
├─ BACKLOG.md      # prioritized improvement ideas
├─ CHANGELOG.md    # append-only iteration log
├─ verify.mjs      # Playwright verification harness (the loop's eyes)
├─ package.json    # dev-only playwright-core; the game has no runtime deps
└─ shots/          # screenshots + report.json from the last verify (gitignored)
```

## Design pillars
**A.** rich variety in looks, movement & intelligence · **B.** Mario-Kart /
Nintendo ease · **C.** outrageous characters, weapons & pick-me-ups · **D.** polish
so high people think it's a paid game.
