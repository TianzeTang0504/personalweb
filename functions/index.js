require("dotenv").config();
const { onSchedule } = require("firebase-functions/v2/scheduler");
const { onRequest } = require("firebase-functions/v2/https");
const { setGlobalOptions } = require("firebase-functions/v2");
const admin = require("firebase-admin");
const { GoogleGenAI } = require("@google/genai");
const nodemailer = require("nodemailer");
const { marked } = require("marked"); // 引入 Markdown 解析器

admin.initializeApp();

// cd functions
// npm install (如果有新依赖)
// cd ..
// firebase deploy --only functions

// 设置全局配置
setGlobalOptions({ timeoutSeconds: 300, memory: "256MiB" });

/**
 * 🔐 安全配置读取
 */
const getAIInstance = () => {
    const key = process.env.GEMINI_API_KEY;
    if (!key) return null;
    return new GoogleGenAI({ apiKey: key });
};

const GMAIL_USER = process.env.GMAIL_USER;
const GMAIL_PASS = process.env.GMAIL_PASS;

/**
 * Core Logic: Generate and send report for a single user
 */
async function processUserReport(uid, userEmail, transporter) {
    const db = admin.firestore();
    const userRef = db.collection("users").doc(uid);

    const [projectsSnap, tasksSnap, eventsSnap, memosSnap] = await Promise.all([
        userRef.collection("projects").get(),
        userRef.collection("tasks").get(),
        userRef.collection("events").get(),
        userRef.collection("memos").get()
    ]);

    if (projectsSnap.empty && tasksSnap.empty && eventsSnap.empty) return;

    const todayStr = new Date().toLocaleDateString('zh-CN');
    let dataContext = `今日日期: ${todayStr}\n\n`;

    // 数据聚合 (简化版以减小 Context 压力)
    projectsSnap.forEach(doc => {
        const p = doc.data();
        dataContext += `- 项目: ${p.name} (DDL: ${p.deadline})\n`;
        if (p.subtasks) p.subtasks.forEach(s => {
            if (s.status !== 'done') dataContext += `    * [${s.status}] ${s.name} (DDL: ${s.deadline || '无'})\n`;
        });
    });
    tasksSnap.forEach(doc => {
        const t = doc.data();
        if (t.status !== 'done') dataContext += `- 任务: ${t.name} (DDL: ${t.deadline || '无'})\n`;
    });
    memosSnap.forEach(doc => { dataContext += `- 备忘: ${doc.data().name} (${doc.data().content}) [日期: ${doc.data().date || '无'}]\n`; });
    eventsSnap.forEach(doc => {
        const e = doc.data();
        if (e.status !== 'done') dataContext += `- 事件: ${e.name} (日期: ${e.date || '无'})\n`;
    });

    try {
        const aiInstance = getAIInstance();
        if (!aiInstance) throw new Error("AI Engine failure.");

        const response = await aiInstance.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: [{
                role: "user",
                parts: [{
                    text: `你是一位专业且高效的私人助手。今天是${todayStr}。
                    请根据提供的数据生成一份结构严谨、排版优雅的邮件日报。
                    
                    要求：
                    1. 严禁使用 Emoji。
                    2. 严格按照以下四个板块生成，每个板块独立清晰，不要互相渗透：
                       - [PROJECTS]：列出重点项目的状态和子任务进度。
                       - [TASKS]：列出待办列表中的关键任务。
                       - [EVENTS]：列出日程表中的重要事件。
                       - [MEMO]：列出相关的备忘或提醒。
                    3. **重要判断逻辑**：不要机械地列出所有内容。仅当任务/事件“即将到期”（如1-3天内）或“看起来非常复杂需要提前准备”时才重点提及。如果一个任务还有一周以上且看起来很简单，请忽略或仅极简提及。
                    4. 语气专业、冷静、精炼。
                    
                    Markdown格式，中文。数据如下：\n${dataContext}`
                }]
            }],
            config: { thinkingConfig: { thinkingLevel: "low" } }
        });

        // 将 Markdown 转换为 HTML
        const reportHtml = marked.parse(response.text);

        await transporter.sendMail({
            from: `"Intelligence Core [Jarvis]" <${GMAIL_USER}>`,
            to: userEmail,
            subject: `Daily Briefing :: ${todayStr}`,
            html: `
                <div style="background-color: #fdfaf6; padding: 40px 20px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 40px; border-radius: 8px; border: 1px solid #e8e1d5; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                        <div style="color: #9a8c7d; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 20px; border-bottom: 1px solid #f0ede9; padding-bottom: 10px;">
                            PROTOCOL_UPDATE // ${todayStr}
                        </div>
                        <div style="color: #333333; line-height: 1.8; font-size: 15px;">
                            ${reportHtml}
                        </div>
                        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #f0ede9; color: #b0a495; font-size: 12px; font-style: italic;">
                            -- End of Daily Report --
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px; color: #b0a495; font-size: 10px;">
                        Intelligence Core v1.4.2 // Multi-User Secured
                    </div>
                </div>
            `
        });
    } catch (e) {
        console.error(`Execution failed for ${userEmail}:`, e);
    }
}

/**
 * Scheduled Task: Individualized Delivery
 * Runs every hour to check which users need their report at this specific hour in their specific timezone.
 */
exports.scheduledDailyReport = onSchedule({
    schedule: "every 1 hours", // Run hourly
    timeZone: "UTC", // Server time reference
    timeoutSeconds: 300
}, async (event) => {
    const db = admin.firestore();
    const usersSnapshot = await db.collection("users").get();
    const transporter = nodemailer.createTransport({
        service: "gmail",
        auth: { user: GMAIL_USER, pass: GMAIL_PASS }
    });

    const tasks = [];
    const now = new Date();

    for (const doc of usersSnapshot.docs) {
        const userData = doc.data();
        const uid = doc.id;

        // Skip if disabled or no email
        if (!userData.email || userData.receiveReport === false) continue;

        // 1. Determine User's Target Time
        const targetHour = userData.reportTime !== undefined ? userData.reportTime : 8; // Default 8 AM
        const userTimezone = userData.timezone || 'Asia/Shanghai'; // Default Shanghai

        try {
            // 2. Get User's Current Local Time
            // We use toLocaleString to convert server 'now' to user's wall-clock time
            const localDateStr = now.toLocaleDateString("en-CA", { timeZone: userTimezone }); // "YYYY-MM-DD" in user land
            const localHourStr = now.toLocaleTimeString("en-US", { timeZone: userTimezone, hour12: false, hour: 'numeric' });

            // Handle edge case where "24" might be returned or similar quirks, although numeric usually gives 0-23
            let currentLocalHour = parseInt(localHourStr);
            if (isNaN(currentLocalHour)) continue;
            if (currentLocalHour === 24) currentLocalHour = 0;

            console.log(`Checking user ${uid} (${userTimezone}): Now ${currentLocalHour}:00, Target ${targetHour}:00. Last sent: ${userData.lastReportSentDate}`);

            // 3. Match Logic
            if (currentLocalHour === targetHour) {
                // Check Idempotency (Don't send if already sent today in their timezone)
                if (userData.lastReportSentDate === localDateStr) {
                    console.log(`Skipping ${uid}: Already sent for ${localDateStr}`);
                    continue;
                }

                // 4. Send & Update
                const sendPromise = processUserReport(uid, userData.email, transporter)
                    .then(async () => {
                        // Mark as sent for this local date
                        await db.collection("users").doc(uid).update({
                            lastReportSentDate: localDateStr
                        });
                        console.log(`Report sent to ${userData.email} for ${localDateStr}`);
                    })
                    .catch(err => {
                        console.error(`Failed to send report to ${uid}:`, err);
                    });

                tasks.push(sendPromise);
            }
        } catch (err) {
            console.error(`Error processing user ${uid} time logic:`, err);
        }
    }

    await Promise.all(tasks);
});

/**
 * HTTP Test
 */
exports.testMultiUserReport = onRequest(async (req, res) => {
    try {
        const db = admin.firestore();
        const usersSnapshot = await db.collection("users").get();
        const transporter = nodemailer.createTransport({
            service: "gmail",
            auth: { user: GMAIL_USER, pass: GMAIL_PASS }
        });

        for (const doc of usersSnapshot.docs) {
            const userData = doc.data();
            if (userData.email && userData.receiveReport !== false) {
                await processUserReport(doc.id, userData.email, transporter);
            }
        }
        res.status(200).send("New styled report dispatched to active subscribers.");
    } catch (error) {
        res.status(500).send(error.message);
    }
});