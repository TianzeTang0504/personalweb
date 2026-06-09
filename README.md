# Personal Web Command Center

A personal static website and private-tool workspace built with vanilla HTML/CSS/JS, Firebase Auth, Firestore, Firebase Storage, and Cloud Functions.

The public site lives in `index.html`. Private tools live as standalone pages and use the same multi-user pattern: users sign in with Firebase Auth, and each user's data is stored under `users/{uid}/...`.

## Current Features

- Public portfolio/blog homepage (`index.html`)
  - Geek-style personal site with projects, blog/content modules, contact links, custom cursor, and audio/background effects.
- Admin command center (`admin_portal.html`)
  - Project, task, event, and memo management.
  - Uses Firebase Auth and Firestore private user data.
  - Legacy Gemini/Gmail daily report code still exists in Cloud Functions, but the scheduled report currently returns early and is disabled.
- Writing practice room (`writer.html`)
  - Private multi-user workspace for drafts, scene exercises, materials, reading breakdowns, weekly reviews, and writing stats.
  - Google sign-in plus email/password fallback.
  - Stores data under `users/{uid}/writingDrafts`, `writingExercises`, `writingMaterials`, `readingBreakdowns`, `writingWeeklyReviews`, `writingStats`, and `writingTaxonomy`.
  - OpenAI-powered structured feedback for one-time scene evaluations and completed weekly insights.
- Food Lab calorie tracker (`calorie.html`)
  - Private multi-user calorie tracker for daily meals, food amounts, optional kcal/100g overrides, body logs, targets, and trend charts.
  - Google sign-in plus email/password fallback, matching the writing room's auth behavior.
  - Stores data under `users/{uid}/calorieSettings`, `calorieDays`, and `bodyLogs`.
  - OpenAI-powered structured calorie/macronutrient estimates through a callable Cloud Function.
  - Estimate Report includes a short daily diet assessment about target adequacy, balance, and practical next-step suggestions.
  - No image input: estimates are based on food names, amounts, units, raw/cooked/packaged state, notes, and optional kcal/100g labels.

## Tech Stack

- Frontend: vanilla HTML, CSS, and JavaScript
- Backend: Firebase Cloud Functions v2
- Auth: Firebase Authentication
- Database: Cloud Firestore
- Storage: Firebase Storage for star journal photos only
- AI:
  - OpenAI Responses API for writing feedback and calorie estimates
  - Google Gemini API for legacy admin daily report code
- Email: Nodemailer/Gmail SMTP for legacy daily report code

## Setup

### 1. Prerequisites

- Node.js compatible with the Cloud Functions engine in `functions/package.json`
- Firebase CLI
- A Firebase project with Blaze enabled if you deploy Cloud Functions that call external APIs

### 2. Firebase Services

Enable:

- Authentication
  - Email/Password for fallback login
  - Google provider for the private tools
- Firestore Database
- Cloud Functions
- Firebase Storage if using star journal photo uploads

Firestore rules already protect private user data:

```text
users/{uid}/...
```

Only the owner user or an admin user can read/write that user's private subcollections.

### 3. Frontend Firebase Config

The shared Firebase web config is in:

```text
js/firebase-config.js
```

The private pages load that file directly. If you fork the project, replace the config object with your Firebase web app config.

### 4. Cloud Function Environment

Create `functions/.env` for local development. Do not commit real secrets.

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_WRITING_MODEL=gpt-5.5
OPENAI_CALORIE_MODEL=gpt-5.4-mini
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GMAIL_USER=your_email@gmail.com
GMAIL_PASS=your_google_app_password
```

Notes:

- `OPENAI_API_KEY` is required for writing AI and calorie AI.
- `OPENAI_WRITING_MODEL` is optional. It defaults to `gpt-5.5`.
- `OPENAI_CALORIE_MODEL` is optional. Calorie estimates default to `gpt-5.4-mini` unless a whitelisted per-user `aiModel` is set.
- `GEMINI_API_KEY`, `GMAIL_USER`, and `GMAIL_PASS` are only needed for the legacy admin daily report code.
- In production, configure the same values as Firebase Functions environment variables or secrets before deploying.

## Private Tool Pages

### Writing Room

Open:

```text
writer.html
```

Cloud Functions:

- `generateWritingExerciseEvaluation({ exerciseId })`
  - Reads one scene exercise from the signed-in user's `writingExercises`.
  - Writes `aiEvaluation` back to that exercise.
  - Refuses to overwrite an existing evaluation.
- `generateWritingWeeklyInsight({ weekId })`
  - Reads the selected completed week's writing data.
  - Writes `aiInsight` and `aiSummary` to `writingWeeklyReviews/{weekId}`.
  - Refuses to generate for the current week or overwrite an existing insight.

### Food Lab

Open:

```text
calorie.html
```

Firestore paths:

- `users/{uid}/calorieSettings/main`
- `users/{uid}/calorieDays/{YYYY-MM-DD}`
- `users/{uid}/bodyLogs/{YYYY-MM-DD}`

Cloud Function:

- `estimateCalorieDay({ dateId })`
  - Reads one day from the signed-in user's `calorieDays`.
  - Validates food names and amounts.
  - Uses kcal/100g entries and notes as strong evidence.
  - Reads each user's `aiModel` setting; supported values are `gpt-5.5`, `gpt-5.4`, and `gpt-5.4-mini`.
  - Sends food rows to OpenAI for structured calorie and macronutrient range estimates.
  - Writes a short `dailyAssessment` into the Estimate Report.
  - Writes `aiEstimate` and `inputHash` back to that day.

Default Food Lab targets are:

```text
3000 kcal
125 g protein
425 g carbs
90 g fat
```

Users can edit targets in the page. The app treats AI estimates as stale when foods change after the last estimate.

## Development

Install functions dependencies:

```bash
cd functions
npm install
```

Useful checks:

```bash
node --check js/writer.js
node --check js/calorie.js
node --check functions/index.js
```

Deploy all Firebase resources:

```bash
firebase deploy
```

Deploy only Cloud Functions after backend changes:

```bash
firebase deploy --only functions
```

Static frontend changes normally require deploying hosting/static files as configured for your Firebase project.

## Notes For Maintenance

- Do not expose OpenAI or Gemini API keys in frontend code.
- Keep AI calls behind callable Cloud Functions.
- Keep new private tool data under `users/{uid}/...` so existing Firestore rules protect it.
- If you add or rename private tool pages, update this README and `AGENTS.md`.
- The repository currently does not include a license file. Add one before publishing or distributing beyond personal use.
