require("dotenv").config();
const { onSchedule } = require("firebase-functions/v2/scheduler");
const { onRequest } = require("firebase-functions/v2/https");
const { setGlobalOptions } = require("firebase-functions/v2");
const admin = require("firebase-admin");
const { GoogleGenAI } = require("@google/genai");
const nodemailer = require("nodemailer");
const { marked } = require("marked"); // 引入 Markdown 解析器

admin.initializeApp();

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
        if (p.subtasks) p.subtasks.forEach(s => dataContext += `    * [${s.status}] ${s.name}\n`);
    });
    tasksSnap.forEach(doc => { if (doc.data().status !== 'done') dataContext += `- 任务: ${doc.data().name}\n`; });
    memosSnap.forEach(doc => { dataContext += `- 备忘: ${doc.data().name} (${doc.data().content}) [${doc.data().date}]\n`; });
    eventsSnap.forEach(doc => { if (doc.data().status !== 'done') dataContext += `- 事件: ${doc.data().name} (${doc.data().date})\n`; });

    try {
        const aiInstance = getAIInstance();
        if (!aiInstance) throw new Error("AI Engine failure.");

        const response = await aiInstance.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: [{
                role: "user",
                parts: [{
                    text: `你是一位全能且温和的智能助手。今天是${todayStr}。
                    请生成一份简洁、排版优雅的今日简报并不要使用emoji，包括：
                    1. [今日日程]：今日重点。
                    2. [临期提示]：即将到期的风险。
                    3. [备忘录提示]：如果备忘录里有临近今天的内容，请给出温馨提示。
                    4. [激励指令]：一句温柔而有力量的话。
                    Markdown格式，中文。数据：\n${dataContext}`
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
                        Intelligence Core v3.0 // Multi-User Secured
                    </div>
                </div>
            `
        });
    } catch (e) {
        console.error(`Execution failed for ${userEmail}:`, e);
    }
}

/**
 * Scheduled Task
 */
exports.scheduledDailyReport = onSchedule({
    schedule: "0 8 * * *",
    timeZone: "Europe/Paris"
}, async (event) => {
    const db = admin.firestore();
    const usersSnapshot = await db.collection("users").get();
    const transporter = nodemailer.createTransport({
        service: "gmail",
        auth: { user: GMAIL_USER, pass: GMAIL_PASS }
    });

    const tasks = [];
    for (const doc of usersSnapshot.docs) {
        if (doc.data().email) tasks.push(processUserReport(doc.id, doc.data().email, transporter));
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
            if (doc.data().email) await processUserReport(doc.id, doc.data().email, transporter);
        }
        res.status(200).send("New styled report dispatched.");
    } catch (error) {
        res.status(500).send(error.message);
    }
});