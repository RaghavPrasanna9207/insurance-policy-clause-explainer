/**
 * Capture every screen of the app, light and dark, by driving a real browser
 * against a real policy.
 *
 * Why this is a committed script rather than a one-off: a design you have not
 * looked at has not been reviewed. The design system forbids a specific list of
 * anti-patterns, and the only way to check the built UI against that list is to
 * see it. Keeping this runnable means the check is repeatable after any change.
 *
 * Requires both servers running:
 *   cd api && .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
 *   cd web && npm run dev
 *
 * Then:  npm run shots [outputDir]
 */
import { mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from 'playwright'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = resolve(process.argv[2] ?? resolve(HERE, '../.screenshots'))
const PDF = resolve(HERE, '../../evals/golden/synthetic-health-policy.pdf')

mkdirSync(OUT, { recursive: true })

const browser = await chromium.launch()
const page = await browser.newPage({
  viewport: { width: 1440, height: 960 },
  // Retina-density capture, so type rendering can actually be judged.
  deviceScaleFactor: 2,
})

const shot = async (name, opts = {}) => {
  await page.screenshot({ path: `${OUT}/${name}.png`, ...opts })
  console.log('shot:', name)
}

const setTheme = async (mode) => {
  await page.evaluate((m) => {
    localStorage.setItem('theme', m)
    document.documentElement.classList.toggle('dark', m === 'dark')
  }, mode)
  await page.waitForTimeout(250)
}

// --- Upload ---
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.waitForTimeout(700) // let webfonts settle before judging type
await shot('01-upload-light')
await setTheme('dark')
await shot('02-upload-dark')
await setTheme('light')

// --- Upload a real policy and watch it process ---
await page.setInputFiles('input[type=file]', PDF)
await page.waitForURL(/\/policy\//, { timeout: 30000 })
await page.waitForTimeout(1500)
await shot('03-processing')

console.log('waiting for analysis (up to 10 min on a cold cache)...')
await page.waitForSelector('article', { timeout: 600000 })
await page.waitForTimeout(1200)

await shot('04-report-light')
await shot('04b-report-full', { fullPage: true })
await setTheme('dark')
await shot('05-report-dark')
await setTheme('light')

// --- The clause panel: plain language beside the policy's own words ---
await page.locator('article button').first().click()
await page.waitForSelector('blockquote', { timeout: 20000 })
await page.waitForTimeout(700)
await shot('06-clause-panel-light')
await setTheme('dark')
await shot('07-clause-panel-dark')
await setTheme('light')

// --- Scenario simulator ---
await page.keyboard.press('Escape')
await page.waitForTimeout(300)
const box = page.locator('#scenario')
await box.scrollIntoViewIfNeeded()
await page.waitForTimeout(400)
await shot('09-scenario-empty')

// Use an example chip: it fires a real question against the real policy.
await page.locator('button', { hasText: 'I have had diabetes for years' }).first().click()
await page.waitForSelector('blockquote', { timeout: 300000 })
await page.waitForTimeout(800)
await box.scrollIntoViewIfNeeded()
await page.waitForTimeout(300)
await shot('10-scenario-answer-light')
await setTheme('dark')
await shot('11-scenario-answer-dark')
await setTheme('light')

// --- Narrow viewport ---
await page.setViewportSize({ width: 430, height: 900 })
await page.waitForTimeout(500)
await shot('08-narrow-report')

await browser.close()
console.log('DONE ->', OUT)
