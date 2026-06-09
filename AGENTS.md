# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project Shape

- This is a vanilla HTML/CSS/JS Firebase site.
- Public homepage: `index.html`.
- Private tool pages:
  - `writer.html` with `js/writer.js` and `css/writer.css`.
  - `calorie.html` with `js/calorie.js` and `css/calorie.css`.
- Backend Cloud Functions: `functions/index.js`.
- Shared Firebase web config: `js/firebase-config.js`.
- Firestore security rules: `firestore.rules`.

## Data And Auth

- Private tools must use Firebase Auth.
- User-private data belongs under `users/{uid}/...`.
- Firestore rules already allow users to read/write their own private subcollections and admins to access all user docs.
- Do not introduce top-level private collections for a single user's data unless rules are updated deliberately.
- Keep browser localStorage keys namespaced by `uid` for multi-user pages.

## AI Boundaries

- Never put API keys in frontend files.
- OpenAI calls must go through callable Cloud Functions.
- Existing OpenAI helper: `createOpenAIJsonResponse` in `functions/index.js`.
- Writing AI functions:
  - `generateWritingExerciseEvaluation`
  - `generateWritingWeeklyInsight`
- Calorie AI function:
  - `estimateCalorieDay`
- Food Lab estimates are food-name/amount based. Do not add image-based calorie estimation unless explicitly requested.
- Food entry should stay lightweight: do not add manual protein/carbs/fat inputs unless explicitly requested.
- Food Lab stores the per-user calorie AI model in `users/{uid}/calorieSettings/main.aiModel`; keep backend model selection whitelisted.
- Calorie estimates should return structured JSON ranges and include assumptions, warnings, and missing information.
- Calorie estimates should also include a short Chinese `dailyAssessment` for target adequacy, balance, and practical adjustment advice.

## Frontend Style

- `writer.html` intentionally has a warm paper/writing-room style.
- `calorie.html` intentionally has a distinct Food Lab/data-workbench style.
- Do not make the calorie UI visually identical to the writing room.
- Keep forms responsive and table-heavy calorie input horizontally scrollable on small screens.
- Avoid adding large frontend frameworks unless the user explicitly asks.

## Checks

Run focused syntax checks after JS or functions edits:

```bash
node --check js/writer.js
node --check js/calorie.js
node --check functions/index.js
```

If editing HTML IDs used by JS, verify the JS selectors still exist in the corresponding HTML file.

## Documentation

- Update `README.md` when adding pages, functions, environment variables, or Firestore paths.
- Keep this file current for future agents.
- Do not document secrets or private credentials.
