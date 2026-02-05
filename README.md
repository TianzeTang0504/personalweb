# Personal Command Center & AI Portfolio

A "Geek-style" personal website featuring a high-precision Task/Schedule Manager (Command Center) and an AI-powered portfolio/blog system. Built with **HTML/JS (Vanilla)** and **Firebase**.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Command+Center+Dashboard)

## Features

- **Cyberpunk/Geek UI**: Glassmorphism, terminal aesthetics, and responsive animations.
- **Admin Command Center**:
  - **Project Management**: Track projects with start/end dates and subtask sequences.
  - **Task & Event Tracking**: Urgent items highlight automatically.
  - **Secure Vault**: Memo/Note storage.
- **Authentication**: Fully integrated Email/Password login with "Forgot Password" flow.
- **AI Intelligence Reports (Cloud Functions)**:
  - Automatically generates a daily briefing summarizing your tasks and schedules.
  - Powered by **Google Gemini API**.
  - **Customizable Schedule**: Users can set their preferred delivery time (hour) and Timezone.
  - Sent directly to your email via Gmail SMTP.

---

## Replication & Setup Guide

If you wish to fork and deploy this system for yourself, follow these steps.

### 1. Prerequisites
- **Node.js** (v18+ recommended)
- **Firebase CLI** (`npm install -g firebase-tools`)
- A **Google Cloud / Firebase** Account

### 2. Firebase Project Setup
1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Create a new project.
3. **Upgrade to Blaze Plan** (Pay-as-you-go): This is **REQUIRED** for Cloud Functions to make external network requests (to Gmail SMTP and Gemini API).
   - *Note: The free usage tier is usually sufficient for personal use, but the plan upgrade is needed to unlock the network firewall.*
4. **Enable Services**:
   - **Authentication**: Enable **Email/Password**.
   - **Firestore Database**: Create a database (Production mode).
   - **Cloud Functions**: (Will be enabled upon deploy).

### 3. Client-Side Configuration
You need to connect the frontend to your new Firebase project.

1. In Firebase Console, go to *Project Settings > General*.
2. Register a **Web app**.
3. Copy the `firebaseConfig` object (keys, IDs, etc.).
4. Open the following files in your code editor and find the `firebaseConfig` section to replace:
   - `admin_portal.html`
   - `index.html` (if applicable)
   - `js/schedule.js` (Verify if config is initialized here)

### 4. Backend Configuration (Environment)
The AI reporting features live in the `functions/` directory.

1. Navigate to the functions folder:
   ```bash
   cd functions
   npm install
   ```
2. Create your environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and fill in your secrets:
   - `GEMINI_API_KEY`: Get your free API key from [Google AI Studio](https://aistudio.google.com/).
   - `GMAIL_USER`: Your full Gmail address (e.g., `youremail@gmail.com`). This is the account that will *send* the emails.
   - `GMAIL_PASS`: Your Google **App Password**.
     - *How to get this*: Go to [Google Account Security](https://myaccount.google.com/security) > 2-Step Verification > App passwords. Create one named "Firebase" and copy the 16-character code. **Do not use your login password.**

### 5. Deploy
1. Login to Firebase CLI:
   ```bash
   firebase login
   ```
2. Initialize project (select your newly created project):
   ```bash
   firebase use --add
   ```
3. Deploy everything (Frontend + Backend):
   ```bash
   firebase deploy
   ```

---

## 🛠 Usage Notes

- **Initial Access**: Open your deployed `admin_portal.html` URL.
- **Registration**: Use the "Initial User Registration" or "Register" button to create your account.
- **Email Verification**: You cannot access the dashboard until you verify your email (click the link sent by Firebase).
- **Settings**: Once logged in, click the **Gear Icon** to configure:
  - **Daily Report**: Enable/Disable.
  - **Time & Timezone**: Set when you want to receive your AI briefing.

## 📄 License
[MIT](LICENSE)
