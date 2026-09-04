# web — Policy Reader interface

Vite + React + TypeScript + Tailwind 4. Talks to the FastAPI backend in `../api`.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api to 127.0.0.1:8000
npm run build
npm run shots    # drive the real UI with a real policy, capture every screen
```

The backend must be running for anything but the upload screen to work:

```bash
cd ../api && .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

## Before changing anything visual

Read **[`../DESIGN.md`](../DESIGN.md)**. It defines the aesthetic direction, the
type scale, the colour tokens, the 2px radius, the motion policy, and a list of
forbidden anti-patterns.

Two rules are easy to break by accident:

1. **The policy's own words are always set in Source Serif 4** (the `.verbatim`
   class); everything the app says is set in Instrument Sans. The typeface is
   what tells a reader which voice is speaking, so the two must never be mixed.
2. **Severity is never carried by colour alone.** Every severity indicator pairs
   its hue with a numeral and a text label — including at narrow widths, where
   the impact rail is hidden and the score moves inline.

## Layout

```
src/
  index.css              design tokens; light and dark, mapped into Tailwind
  types.ts               mirrors api/app/schemas.py
  api.ts                 typed client
  clauseMeta.ts          clause type -> label, severity band, consequence text
                         and the buriedness -> "why you'd miss this" sentences
  components/Chrome.tsx  masthead, metadata strip, theme toggle, disclaimer
  components/RiskCard.tsx    one clause in the ranked list
  components/ClausePanel.tsx plain language beside the policy's own words
  routes/Upload.tsx      upload
  routes/Policy.tsx      processing, the ranked report, filters
scripts/screenshots.mjs  Playwright capture of every screen, light and dark
```
