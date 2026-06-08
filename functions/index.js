const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env"), quiet: true });
const { onSchedule } = require("firebase-functions/v2/scheduler");
const { onRequest, onCall, HttpsError } = require("firebase-functions/v2/https");
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
const OPENAI_MODEL = process.env.OPENAI_WRITING_MODEL || "gpt-5.5";
const OPENAI_CALORIE_MODEL = process.env.OPENAI_CALORIE_MODEL || OPENAI_MODEL;
const WRITING_PROMPT_VERSION = "writing-coach-v1";
const CALORIE_PROMPT_VERSION = "calorie-estimator-v1";
const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";

const WRITING_COACH_INSTRUCTIONS = `
你是一个小说写作训练教练。
你的任务不是代写、改写、续写或全文润色，而是基于用户自己的文本和统计数据，给出短、具体、可执行的训练反馈。

原则：
1. 只基于输入数据判断，不编造用户没有写过的内容。
2. 必须指出具体证据，避免泛泛夸奖。
3. 关注小说训练：人物欲望、冲突、场景推进、对白、节奏、氛围、转折。
4. 引用用户文本只能使用很短摘录。
5. 输出必须严格符合 JSON Schema，不要输出 Markdown。
`;

const CALORIE_ESTIMATOR_INSTRUCTIONS = `
你是一个谨慎的营养估算助手，服务于个人增重记录。
你只根据用户给出的原料、重量、单位、生熟状态、品牌备注和可选营养标签估算热量与三大营养素。

原则：
1. 不处理图片，不假装能从菜名精确知道热量。
2. 有明确营养标签并可按重量换算的项目由系统确定性计算，你不要重新估算这些项目。
3. 对缺少品牌、做法或净含量的信息，给出合理范围而不是单点值。
4. 对单位为“份、碗、杯、袋、盒”等模糊量的项目，必须在 basis 或 missingInfo 中说明换算假设。
5. 不给医疗建议，不评价健康风险，只做记录估算。
6. 输出必须严格符合 JSON Schema，不要输出 Markdown。
`;

const EXERCISE_EVALUATION_SCHEMA = {
    type: "object",
    additionalProperties: false,
    required: [
        "summary",
        "score",
        "strengths",
        "weaknesses",
        "evidence",
        "revisionTargets",
        "nextPrompt"
    ],
    properties: {
        summary: { type: "string" },
        score: {
            type: "object",
            additionalProperties: false,
            required: [
                "characterDesire",
                "conflictClarity",
                "sceneProgression",
                "proseControl",
                "overall"
            ],
            properties: {
                characterDesire: { type: "number" },
                conflictClarity: { type: "number" },
                sceneProgression: { type: "number" },
                proseControl: { type: "number" },
                overall: { type: "number" }
            }
        },
        strengths: {
            type: "array",
            items: { type: "string" }
        },
        weaknesses: {
            type: "array",
            items: { type: "string" }
        },
        evidence: {
            type: "array",
            items: {
                type: "object",
                additionalProperties: false,
                required: ["quote", "point"],
                properties: {
                    quote: { type: "string" },
                    point: { type: "string" }
                }
            }
        },
        revisionTargets: {
            type: "array",
            items: {
                type: "object",
                additionalProperties: false,
                required: ["target", "why", "exercise"],
                properties: {
                    target: { type: "string" },
                    why: { type: "string" },
                    exercise: { type: "string" }
                }
            }
        },
        nextPrompt: { type: "string" }
    }
};

const WEEKLY_INSIGHT_SCHEMA = {
    type: "object",
    additionalProperties: false,
    required: [
        "headline",
        "summaryText",
        "statsRead",
        "comparedToLastWeek",
        "rhythm",
        "strongestProgress",
        "mainBlocker",
        "evidence",
        "nextWeekPlan"
    ],
    properties: {
        headline: { type: "string" },
        summaryText: { type: "string" },
        statsRead: { type: "string" },
        comparedToLastWeek: { type: "string" },
        rhythm: { type: "string" },
        strongestProgress: { type: "string" },
        mainBlocker: { type: "string" },
        evidence: {
            type: "array",
            items: {
                type: "object",
                additionalProperties: false,
                required: ["source", "point"],
                properties: {
                    source: { type: "string" },
                    point: { type: "string" }
                }
            }
        },
        nextWeekPlan: {
            type: "array",
            items: {
                type: "object",
                additionalProperties: false,
                required: ["focus", "task"],
                properties: {
                    focus: { type: "string" },
                    task: { type: "string" }
                }
            }
        }
    }
};

const NUTRIENT_RANGE_SCHEMA = {
    type: "object",
    additionalProperties: false,
    required: ["low", "mid", "high"],
    properties: {
        low: { type: "number" },
        mid: { type: "number" },
        high: { type: "number" }
    }
};

const CALORIE_ESTIMATE_SCHEMA = {
    type: "object",
    additionalProperties: false,
    required: [
        "items",
        "assumptions",
        "warnings",
        "missingInfo",
        "confidence"
    ],
    properties: {
        items: {
            type: "array",
            items: {
                type: "object",
                additionalProperties: false,
                required: [
                    "ingredientId",
                    "name",
                    "kcal",
                    "protein",
                    "carbs",
                    "fat",
                    "basis",
                    "confidence"
                ],
                properties: {
                    ingredientId: { type: "string" },
                    name: { type: "string" },
                    kcal: NUTRIENT_RANGE_SCHEMA,
                    protein: NUTRIENT_RANGE_SCHEMA,
                    carbs: NUTRIENT_RANGE_SCHEMA,
                    fat: NUTRIENT_RANGE_SCHEMA,
                    basis: { type: "string" },
                    confidence: { type: "string" }
                }
            }
        },
        assumptions: {
            type: "array",
            items: { type: "string" }
        },
        warnings: {
            type: "array",
            items: { type: "string" }
        },
        missingInfo: {
            type: "array",
            items: { type: "string" }
        },
        confidence: { type: "string" }
    }
};

function assertAuthedUid(request) {
    const uid = request.auth?.uid;
    if (!uid) {
        throw new HttpsError("unauthenticated", "请先登录。");
    }
    return uid;
}

function getOpenAIKey() {
    const key = process.env.OPENAI_API_KEY;
    if (!key) {
        throw new HttpsError("failed-precondition", "OPENAI_API_KEY 未配置。");
    }
    return key;
}

function cleanId(value, fieldName) {
    const clean = String(value || "").trim();
    if (!clean || clean.length > 120 || !/^[A-Za-z0-9_-]+$/.test(clean)) {
        throw new HttpsError("invalid-argument", `${fieldName} 无效。`);
    }
    return clean;
}

function cleanWeekId(value) {
    const raw = String(value || "").trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
        throw new HttpsError("invalid-argument", "weekId 无效。");
    }
    const normalized = getWeekRange(parseDateId(raw)).weekId;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(normalized)) {
        throw new HttpsError("invalid-argument", "weekId 无效。");
    }
    return normalized;
}

function truncateText(value, maxLength = 4000) {
    const text = String(value || "").trim();
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength)}\n...[已截断]`;
}

function countWords(text) {
    const value = String(text || "").trim();
    if (!value) return 0;
    const cjkMatches = value.match(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g) || [];
    const latinText = value.replace(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g, " ");
    const latinMatches = latinText.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g) || [];
    return cjkMatches.length + latinMatches.length;
}

function getDocData(doc) {
    return { id: doc.id, ...doc.data() };
}

function timestampToDate(value) {
    if (!value) return null;
    if (value instanceof Date) return value;
    if (typeof value.toDate === "function") return value.toDate();
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function parseDateId(value) {
    if (value instanceof Date) {
        return new Date(value.getFullYear(), value.getMonth(), value.getDate());
    }
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
        return new Date();
    }
    const [year, month, day] = value.split("-").map(Number);
    const parsed = new Date(year, month - 1, day);
    return Number.isNaN(parsed.getTime()) ? new Date() : parsed;
}

function getDateParts(date) {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, "0");
    const d = String(date.getDate()).padStart(2, "0");
    return {
        id: `${y}-${m}-${d}`,
        label: `${m}/${d}`
    };
}

function getWeekRange(anchorDate = new Date()) {
    const now = anchorDate instanceof Date ? anchorDate : parseDateId(anchorDate);
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const day = start.getDay() || 7;
    start.setDate(start.getDate() - day + 1);
    const end = new Date(start);
    end.setDate(start.getDate() + 6);
    end.setHours(23, 59, 59, 999);
    const startParts = getDateParts(start);
    const endParts = getDateParts(end);
    return {
        weekId: startParts.id,
        weekStart: start,
        weekEnd: end,
        weekStartDate: startParts.id,
        weekEndDate: endParts.id,
        label: `${startParts.label} - ${endParts.label}`
    };
}

function shiftWeek(range, offset) {
    const next = new Date(range.weekStart);
    next.setDate(next.getDate() + offset * 7);
    return getWeekRange(next);
}

function isDateInRange(value, range) {
    const date = timestampToDate(value);
    return Boolean(date && date >= range.weekStart && date <= range.weekEnd);
}

function extractResponseText(responseJson) {
    if (typeof responseJson.output_text === "string") {
        return responseJson.output_text;
    }
    const chunks = [];
    (responseJson.output || []).forEach((item) => {
        (item.content || []).forEach((content) => {
            if (typeof content.text === "string") chunks.push(content.text);
            if (typeof content.output_text === "string") chunks.push(content.output_text);
        });
    });
    return chunks.join("\n").trim();
}

function parseStructuredJson(text) {
    const clean = String(text || "")
        .trim()
        .replace(/^```json\s*/i, "")
        .replace(/^```\s*/i, "")
        .replace(/\s*```$/i, "");
    return JSON.parse(clean);
}

async function createOpenAIJsonResponse({
    input,
    schema,
    schemaName,
    maxOutputTokens,
    instructions = WRITING_COACH_INSTRUCTIONS,
    model = OPENAI_MODEL
}) {
    const response = await fetch(OPENAI_RESPONSES_URL, {
        method: "POST",
        headers: {
            Authorization: `Bearer ${getOpenAIKey()}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model,
            instructions,
            input,
            reasoning: { effort: "medium" },
            max_output_tokens: maxOutputTokens,
            text: {
                format: {
                    type: "json_schema",
                    name: schemaName,
                    strict: true,
                    schema
                }
            }
        })
    });

    const rawBody = await response.text();
    let responseJson = null;
    try {
        responseJson = JSON.parse(rawBody);
    } catch (error) {
        console.error("OpenAI returned non-JSON response", rawBody.slice(0, 500));
    }

    if (!response.ok) {
        console.error("OpenAI request failed", response.status, responseJson || rawBody.slice(0, 500));
        throw new HttpsError("unavailable", "AI 服务暂时不可用，请稍后再试。");
    }

    try {
        return {
            data: parseStructuredJson(extractResponseText(responseJson || {})),
            responseId: responseJson?.id || "",
            usage: responseJson?.usage || null
        };
    } catch (error) {
        console.error("Failed to parse OpenAI structured output", error, rawBody.slice(0, 500));
        throw new HttpsError("internal", "AI 返回格式解析失败。");
    }
}

function normalizeStatContributions(value) {
    const result = {};
    if (!value || typeof value !== "object") return result;
    Object.entries(value).forEach(([dateId, amount]) => {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(dateId)) return;
        const cleanAmount = Number(amount || 0);
        if (!cleanAmount) return;
        result[dateId] = cleanAmount;
    });
    return result;
}

function getFallbackStatDate(item) {
    const date = timestampToDate(item.updatedAt || item.createdAt);
    return getDateParts(date || new Date()).id;
}

function getItemContributionSource(item) {
    const contributions = normalizeStatContributions(item.statsContributions);
    if (Object.keys(contributions).length) return contributions;
    const wordCount = Number(item.wordCount || 0);
    if (!wordCount) return {};
    return {
        [getFallbackStatDate(item)]: wordCount
    };
}

function addItemsToStatsMap(dayMap, items, field) {
    items.forEach((item) => {
        Object.entries(getItemContributionSource(item)).forEach(([dateId, amount]) => {
            const day = dayMap.get(dateId);
            if (!day) return;
            const cleanAmount = Number(amount || 0);
            day[field] += cleanAmount;
            day.totalWords += cleanAmount;
        });
    });
}

function getWritingDayTotals(drafts, exercises) {
    const totals = new Map();
    const addItem = (item) => {
        Object.entries(getItemContributionSource(item)).forEach(([dateId, amount]) => {
            totals.set(dateId, Number(totals.get(dateId) || 0) + Number(amount || 0));
        });
    };
    drafts.forEach(addItem);
    exercises.forEach(addItem);
    return totals;
}

function buildStatsSnapshot(drafts, exercises, range) {
    const days = [];
    for (let i = 0; i < 7; i += 1) {
        const date = new Date(range.weekStart);
        date.setDate(range.weekStart.getDate() + i);
        const parts = getDateParts(date);
        days.push({
            date: parts.id,
            draftWords: 0,
            exerciseWords: 0,
            totalWords: 0
        });
    }

    const dayMap = new Map(days.map((day) => [day.date, day]));
    addItemsToStatsMap(dayMap, drafts, "draftWords");
    addItemsToStatsMap(dayMap, exercises, "exerciseWords");

    const writtenDays = new Set(
        Array.from(getWritingDayTotals(drafts, exercises).entries())
            .filter(([, total]) => total > 0)
            .map(([dateId]) => dateId)
    );
    let streak = 0;
    const anchor = range.weekEnd > new Date() ? new Date() : range.weekEnd;
    const cursor = new Date(anchor.getFullYear(), anchor.getMonth(), anchor.getDate());
    while (writtenDays.has(getDateParts(cursor).id)) {
        streak += 1;
        cursor.setDate(cursor.getDate() - 1);
    }

    const rawDays = Array.from(dayMap.values());
    const visibleDays = rawDays.map((day) => ({
        ...day,
        draftWords: Math.max(0, day.draftWords),
        exerciseWords: Math.max(0, day.exerciseWords),
        totalWords: Math.max(0, day.totalWords)
    }));

    return {
        draftWords: Math.max(0, rawDays.reduce((sum, day) => sum + day.draftWords, 0)),
        exerciseWords: Math.max(0, rawDays.reduce((sum, day) => sum + day.exerciseWords, 0)),
        totalWords: Math.max(0, rawDays.reduce((sum, day) => sum + day.totalWords, 0)),
        completedExercises: exercises.filter((exercise) => (
            exercise.status === "done" && isDateInRange(exercise.completedAt, range)
        )).length,
        streak,
        days: visibleDays
    };
}

function pickWeeklyItems(items, range, includeFn, mapper, limit = 8) {
    return items
        .filter((item) => includeFn(item, range))
        .sort((a, b) => {
            const aDate = timestampToDate(a.updatedAt || a.completedAt || a.createdAt)?.getTime() || 0;
            const bDate = timestampToDate(b.updatedAt || b.completedAt || b.createdAt)?.getTime() || 0;
            return bDate - aDate;
        })
        .slice(0, limit)
        .map(mapper);
}

async function loadUserWritingContext(userRef, range, previousRange) {
    const [
        draftsSnap,
        exercisesSnap,
        materialsSnap,
        readingsSnap,
        previousReviewSnap
    ] = await Promise.all([
        userRef.collection("writingDrafts").get(),
        userRef.collection("writingExercises").get(),
        userRef.collection("writingMaterials").get(),
        userRef.collection("readingBreakdowns").get(),
        userRef.collection("writingWeeklyReviews").doc(previousRange.weekId).get()
    ]);

    const drafts = draftsSnap.docs.map(getDocData);
    const exercises = exercisesSnap.docs.map(getDocData);
    const materials = materialsSnap.docs.map(getDocData);
    const readings = readingsSnap.docs.map(getDocData);

    return {
        stats: buildStatsSnapshot(drafts, exercises, range),
        previousStats: buildStatsSnapshot(drafts, exercises, previousRange),
        previousReview: previousReviewSnap.exists ? previousReviewSnap.data() : null,
        drafts: pickWeeklyItems(
            drafts,
            range,
            (item) => isDateInRange(item.updatedAt || item.createdAt, range),
            (item) => ({
                title: item.title || "未命名草稿",
                wordCount: Number(item.wordCount || 0),
                tags: Array.isArray(item.tags) ? item.tags.slice(0, 8) : [],
                excerpt: truncateText(item.body, 900)
            }),
            6
        ),
        exercises: pickWeeklyItems(
            exercises,
            range,
            (item) => isDateInRange(item.updatedAt || item.completedAt || item.createdAt, range),
            (item) => ({
                prompt: truncateText(item.prompt, 400),
                focus: item.focus || "未分类",
                status: item.status || "draft",
                wordCount: Number(item.wordCount || 0),
                body: truncateText(item.body, 900),
                aiSummary: item.aiEvaluation?.summary || ""
            }),
            8
        ),
        materials: pickWeeklyItems(
            materials,
            range,
            (item) => isDateInRange(item.updatedAt || item.createdAt, range),
            (item) => ({
                type: item.type || "未分类",
                title: item.title || "未命名素材",
                tags: Array.isArray(item.tags) ? item.tags.slice(0, 8) : [],
                content: truncateText(item.content, 500)
            }),
            6
        ),
        readings: pickWeeklyItems(
            readings,
            range,
            (item) => isDateInRange(item.updatedAt || item.createdAt, range),
            (item) => ({
                sourceTitle: item.sourceTitle || "未命名作品",
                author: item.author || "",
                lens: item.lens || "未分类",
                excerpt: truncateText(item.excerpt, 500),
                notes: truncateText(item.notes, 500),
                tags: Array.isArray(item.tags) ? item.tags.slice(0, 8) : []
            }),
            6
        )
    };
}

function limitString(value, maxLength = 600) {
    return truncateText(value, maxLength).replace(/\s+/g, " ").trim();
}

function limitStringArray(values, maxItems = 4, maxLength = 180) {
    return (Array.isArray(values) ? values : [])
        .map((value) => limitString(value, maxLength))
        .filter(Boolean)
        .slice(0, maxItems);
}

function normalizeScore(score = {}) {
    return {
        characterDesire: Number(score.characterDesire || 0),
        conflictClarity: Number(score.conflictClarity || 0),
        sceneProgression: Number(score.sceneProgression || 0),
        proseControl: Number(score.proseControl || 0),
        overall: Number(score.overall || 0)
    };
}

function normalizeExerciseEvaluation(raw, responseMeta) {
    return {
        summary: limitString(raw.summary, 260),
        score: normalizeScore(raw.score),
        strengths: limitStringArray(raw.strengths, 3, 180),
        weaknesses: limitStringArray(raw.weaknesses, 3, 180),
        evidence: (Array.isArray(raw.evidence) ? raw.evidence : []).slice(0, 3).map((item) => ({
            quote: limitString(item?.quote, 80),
            point: limitString(item?.point, 180)
        })).filter((item) => item.quote || item.point),
        revisionTargets: (Array.isArray(raw.revisionTargets) ? raw.revisionTargets : []).slice(0, 3).map((item) => ({
            target: limitString(item?.target, 160),
            why: limitString(item?.why, 180),
            exercise: limitString(item?.exercise, 220)
        })).filter((item) => item.target || item.exercise),
        nextPrompt: limitString(raw.nextPrompt, 260),
        generatedAt: admin.firestore.Timestamp.now(),
        model: OPENAI_MODEL,
        promptVersion: WRITING_PROMPT_VERSION,
        responseId: responseMeta.responseId || "",
        usage: responseMeta.usage || null
    };
}

function normalizeWeeklyInsight(raw, responseMeta) {
    return {
        headline: limitString(raw.headline, 160),
        summaryText: limitString(raw.summaryText, 500),
        statsRead: limitString(raw.statsRead, 360),
        comparedToLastWeek: limitString(raw.comparedToLastWeek, 360),
        rhythm: limitString(raw.rhythm, 360),
        strongestProgress: limitString(raw.strongestProgress, 360),
        mainBlocker: limitString(raw.mainBlocker, 360),
        evidence: (Array.isArray(raw.evidence) ? raw.evidence : []).slice(0, 4).map((item) => ({
            source: limitString(item?.source, 120),
            point: limitString(item?.point, 220)
        })).filter((item) => item.source || item.point),
        nextWeekPlan: (Array.isArray(raw.nextWeekPlan) ? raw.nextWeekPlan : []).slice(0, 4).map((item) => ({
            focus: limitString(item?.focus, 120),
            task: limitString(item?.task, 260)
        })).filter((item) => item.focus || item.task),
        generatedAt: admin.firestore.Timestamp.now(),
        model: OPENAI_MODEL,
        promptVersion: WRITING_PROMPT_VERSION,
        responseId: responseMeta.responseId || "",
        usage: responseMeta.usage || null
    };
}

function cleanDateId(value, fieldName = "dateId") {
    const raw = String(value || "").trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
        throw new HttpsError("invalid-argument", `${fieldName} 无效。`);
    }
    return raw;
}

function cleanCalorieId(value, fallback = "") {
    const clean = String(value || fallback || "").replace(/[^A-Za-z0-9_-]/g, "").slice(0, 80);
    return clean || fallback;
}

function normalizeCalorieLabel(label = {}) {
    const result = {};
    ["kcal", "protein", "carbs", "fat"].forEach((key) => {
        const value = Number(label?.[key]);
        if (Number.isFinite(value) && value >= 0) {
            result[key] = value;
        }
    });
    return result;
}

function normalizeCalorieMeals(meals) {
    return (Array.isArray(meals) ? meals : [])
        .map((meal, index) => ({
            id: cleanCalorieId(meal?.id, `meal_${index}`),
            name: limitString(meal?.name || `餐次 ${index + 1}`, 40),
            order: Number.isFinite(Number(meal?.order)) ? Number(meal.order) : index
        }))
        .sort((a, b) => a.order - b.order);
}

function normalizeCalorieIngredients(ingredients) {
    return (Array.isArray(ingredients) ? ingredients : [])
        .map((item, index) => ({
            id: cleanCalorieId(item?.id, `ingredient_${index}`),
            mealId: cleanCalorieId(item?.mealId, ""),
            name: limitString(item?.name || "", 80),
            amount: Number.isFinite(Number(item?.amount)) ? Number(item.amount) : null,
            unit: limitString(item?.unit || "g", 20),
            state: limitString(item?.state || "unknown", 30),
            note: limitString(item?.note || "", 180),
            labelPer100g: normalizeCalorieLabel(item?.labelPer100g),
            order: Number.isFinite(Number(item?.order)) ? Number(item.order) : index
        }))
        .sort((a, b) => a.order - b.order);
}

function makeCalorieInputHash(day) {
    const meals = normalizeCalorieMeals(day.meals).map((meal) => ({
        id: meal.id,
        name: meal.name,
        order: meal.order
    }));
    const ingredients = normalizeCalorieIngredients(day.ingredients).map((item) => ({
        id: item.id,
        mealId: item.mealId,
        name: item.name,
        amount: item.amount,
        unit: item.unit,
        state: item.state,
        note: item.note,
        labelPer100g: item.labelPer100g,
        order: item.order
    }));
    return simpleHash(JSON.stringify({
        meals,
        ingredients,
        note: String(day.note || "")
    }));
}

function simpleHash(value) {
    let hash = 5381;
    for (let i = 0; i < value.length; i += 1) {
        hash = ((hash << 5) + hash) ^ value.charCodeAt(i);
    }
    return `h${(hash >>> 0).toString(36)}`;
}

function makeNutrientRange(value) {
    const clean = Math.max(0, Number(value || 0));
    return { low: clean, mid: clean, high: clean };
}

function makeEmptyCalorieTotals() {
    return {
        kcal: makeNutrientRange(0),
        protein: makeNutrientRange(0),
        carbs: makeNutrientRange(0),
        fat: makeNutrientRange(0)
    };
}

function normalizeNutrientRange(value = {}) {
    if (typeof value === "number") return makeNutrientRange(value);
    const values = [
        Number(value.low || 0),
        Number(value.mid || 0),
        Number(value.high || 0)
    ].map((number) => (Number.isFinite(number) ? Math.max(0, number) : 0)).sort((a, b) => a - b);
    return {
        low: values[0],
        mid: values[1],
        high: values[2]
    };
}

function addCalorieTotals(target, addition) {
    ["kcal", "protein", "carbs", "fat"].forEach((key) => {
        const range = normalizeNutrientRange(addition?.[key]);
        target[key].low += range.low;
        target[key].mid += range.mid;
        target[key].high += range.high;
    });
}

function getDeterministicNutrition(item) {
    const label = item.labelPer100g || {};
    const hasFullLabel = ["kcal", "protein", "carbs", "fat"].every((key) => Number.isFinite(Number(label[key])));
    const canScaleBy100 = Number(item.amount) > 0 && ["g", "ml"].includes(item.unit);
    if (!hasFullLabel || !canScaleBy100) return null;

    const scale = Number(item.amount) / 100;
    return {
        ingredientId: item.id,
        name: item.name,
        kcal: makeNutrientRange(Number(label.kcal) * scale),
        protein: makeNutrientRange(Number(label.protein) * scale),
        carbs: makeNutrientRange(Number(label.carbs) * scale),
        fat: makeNutrientRange(Number(label.fat) * scale),
        basis: `按营养标签每 100${item.unit} 和 ${item.amount}${item.unit} 确定性换算。`,
        confidence: "high",
        source: "label"
    };
}

function normalizeCalorieEstimateItem(raw, fallbackItem = null) {
    return {
        ingredientId: cleanCalorieId(raw?.ingredientId, fallbackItem?.id || ""),
        name: limitString(raw?.name || fallbackItem?.name || "未命名原料", 80),
        kcal: normalizeNutrientRange(raw?.kcal),
        protein: normalizeNutrientRange(raw?.protein),
        carbs: normalizeNutrientRange(raw?.carbs),
        fat: normalizeNutrientRange(raw?.fat),
        basis: limitString(raw?.basis || "", 260),
        confidence: limitString(raw?.confidence || "medium", 40),
        source: raw?.source || "ai"
    };
}

function combineCalorieEstimate({ rawAi, deterministicItems, aiItems, responseMeta, inputHash }) {
    const totals = makeEmptyCalorieTotals();
    const ingredientEstimates = [];

    deterministicItems.forEach((item) => {
        const normalized = normalizeCalorieEstimateItem(item);
        ingredientEstimates.push(normalized);
        addCalorieTotals(totals, normalized);
    });

    const aiItemsById = new Map((Array.isArray(rawAi?.items) ? rawAi.items : []).map((item) => [
        cleanCalorieId(item?.ingredientId, ""),
        item
    ]));

    aiItems.forEach((sourceItem) => {
        const raw = aiItemsById.get(sourceItem.id);
        if (!raw) {
            ingredientEstimates.push({
                ingredientId: sourceItem.id,
                name: sourceItem.name,
                kcal: makeNutrientRange(0),
                protein: makeNutrientRange(0),
                carbs: makeNutrientRange(0),
                fat: makeNutrientRange(0),
                basis: "AI 未返回该原料估算。",
                confidence: "low",
                source: "missing"
            });
            return;
        }
        const normalized = normalizeCalorieEstimateItem(raw, sourceItem);
        ingredientEstimates.push(normalized);
        addCalorieTotals(totals, normalized);
    });

    return {
        totals,
        ingredientEstimates,
        assumptions: limitStringArray(rawAi?.assumptions, 8, 220),
        warnings: limitStringArray(rawAi?.warnings, 8, 220),
        missingInfo: limitStringArray(rawAi?.missingInfo, 8, 220),
        confidence: limitString(rawAi?.confidence || "medium", 40),
        inputHash,
        generatedAt: admin.firestore.Timestamp.now(),
        model: OPENAI_CALORIE_MODEL,
        promptVersion: CALORIE_PROMPT_VERSION,
        responseId: responseMeta?.responseId || "",
        usage: responseMeta?.usage || null
    };
}

exports.estimateCalorieDay = onCall(async (request) => {
    const uid = assertAuthedUid(request);
    const dateId = cleanDateId(request.data?.dateId);
    const db = admin.firestore();
    const dayRef = db.collection("users").doc(uid).collection("calorieDays").doc(dateId);
    const daySnap = await dayRef.get();

    if (!daySnap.exists) {
        throw new HttpsError("not-found", "找不到这一天的热量记录。");
    }

    const day = { id: daySnap.id, ...daySnap.data() };
    const meals = normalizeCalorieMeals(day.meals);
    const mealNameById = new Map(meals.map((meal) => [meal.id, meal.name]));
    const ingredients = normalizeCalorieIngredients(day.ingredients)
        .filter((item) => item.name || item.amount || item.note || Object.keys(item.labelPer100g || {}).length);

    if (!ingredients.length) {
        throw new HttpsError("failed-precondition", "这一天还没有可估算的原料。");
    }

    const namedIngredients = ingredients.filter((item) => item.name);
    if (!namedIngredients.length) {
        throw new HttpsError("failed-precondition", "请至少填写一条原料名称。");
    }

    const missingAmount = namedIngredients.find((item) => !(Number(item.amount) > 0));
    if (missingAmount) {
        throw new HttpsError("invalid-argument", `“${missingAmount.name}” 缺少数量，无法估算。`);
    }

    const inputHash = makeCalorieInputHash({
        meals,
        ingredients,
        note: day.note || ""
    });
    const deterministicItems = [];
    const aiItems = [];

    namedIngredients.forEach((item) => {
        const deterministic = getDeterministicNutrition(item);
        if (deterministic) {
            deterministicItems.push(deterministic);
        } else {
            aiItems.push({
                id: item.id,
                mealName: mealNameById.get(item.mealId) || "未分类",
                name: item.name,
                amount: item.amount,
                unit: item.unit,
                state: item.state,
                note: item.note,
                labelPer100g: item.labelPer100g
            });
        }
    });

    let responseMeta = { data: { items: [], assumptions: [], warnings: [], missingInfo: [], confidence: "high" } };
    if (aiItems.length) {
        const promptInput = {
            task: "请只估算 estimateIngredients 中的原料热量与三大营养素范围。deterministicItems 已由系统根据营养标签换算，不要重复估算。",
            dateId,
            dayNote: truncateText(day.note, 800),
            units: "metric",
            deterministicItems: deterministicItems.map((item) => ({
                ingredientId: item.ingredientId,
                name: item.name,
                kcal: item.kcal,
                protein: item.protein,
                carbs: item.carbs,
                fat: item.fat,
                basis: item.basis
            })),
            estimateIngredients: aiItems,
            outputLanguage: "中文"
        };

        responseMeta = await createOpenAIJsonResponse({
            input: JSON.stringify(promptInput, null, 2),
            schema: CALORIE_ESTIMATE_SCHEMA,
            schemaName: "calorie_day_estimate",
            maxOutputTokens: 1800,
            instructions: CALORIE_ESTIMATOR_INSTRUCTIONS,
            model: OPENAI_CALORIE_MODEL
        });
    }

    const aiEstimate = combineCalorieEstimate({
        rawAi: responseMeta.data,
        deterministicItems,
        aiItems,
        responseMeta,
        inputHash
    });

    await dayRef.set({
        inputHash,
        aiEstimate,
        updatedAt: admin.firestore.FieldValue.serverTimestamp()
    }, { merge: true });

    return { aiEstimate, inputHash };
});

exports.generateWritingExerciseEvaluation = onCall(async (request) => {
    const uid = assertAuthedUid(request);
    const exerciseId = cleanId(request.data?.exerciseId, "exerciseId");
    const db = admin.firestore();
    const exerciseRef = db.collection("users").doc(uid).collection("writingExercises").doc(exerciseId);
    const exerciseSnap = await exerciseRef.get();

    if (!exerciseSnap.exists) {
        throw new HttpsError("not-found", "找不到这条练习。");
    }

    const exercise = exerciseSnap.data();
    if (exercise.aiEvaluation) {
        throw new HttpsError("already-exists", "这条练习已经生成过 AI 评估。");
    }

    const body = String(exercise.body || "").trim();
    const wordCount = Number(exercise.wordCount || countWords(body));
    if (wordCount < 30) {
        throw new HttpsError("failed-precondition", "正文太短，先写到 30 字以上再评估。");
    }

    const promptInput = {
        task: "请评估一条小说场景练习，给出训练教练式反馈。分数使用 1-5 分，5 分最好。",
        exercise: {
            focus: exercise.focus || "未分类",
            prompt: truncateText(exercise.prompt, 800),
            status: exercise.status || "draft",
            wordCount,
            body: truncateText(body, 8000)
        },
        outputLanguage: "中文"
    };

    const responseMeta = await createOpenAIJsonResponse({
        input: JSON.stringify(promptInput, null, 2),
        schema: EXERCISE_EVALUATION_SCHEMA,
        schemaName: "writing_exercise_evaluation",
        maxOutputTokens: 1600
    });
    const aiEvaluation = normalizeExerciseEvaluation(responseMeta.data, responseMeta);

    await exerciseRef.set({
        aiEvaluation,
        aiLockedAt: admin.firestore.FieldValue.serverTimestamp(),
        updatedAt: admin.firestore.FieldValue.serverTimestamp()
    }, { merge: true });

    return { aiEvaluation };
});

exports.generateWritingWeeklyInsight = onCall(async (request) => {
    const uid = assertAuthedUid(request);
    const weekId = cleanWeekId(request.data?.weekId);
    const db = admin.firestore();
    const userRef = db.collection("users").doc(uid);
    const range = getWeekRange(parseDateId(weekId));
    const previousRange = shiftWeek(range, -1);
    const currentRange = getWeekRange(new Date());

    if (range.weekId >= currentRange.weekId) {
        throw new HttpsError("failed-precondition", "本周结束后才能生成正式 AI 周总结。");
    }

    const reviewRef = userRef.collection("writingWeeklyReviews").doc(range.weekId);
    const reviewSnap = await reviewRef.get();
    const currentReview = reviewSnap.exists ? reviewSnap.data() : {};

    if (currentReview.aiInsight) {
        throw new HttpsError("already-exists", "这一周已经生成过 AI 周总结。");
    }

    const context = await loadUserWritingContext(userRef, range, previousRange);
    const hasWriting = context.stats.totalWords > 0 || context.exercises.length > 0 || context.drafts.length > 0;
    const hasReview = String(currentReview.manualReview || "").trim().length > 0;

    if (!hasWriting && !hasReview) {
        throw new HttpsError("failed-precondition", "这一周还没有可总结的写作内容。");
    }

    const promptInput = {
        task: "请生成小说写作训练周复盘。要比较上周变化，判断本周节奏，并给出下周训练计划。",
        week: {
            weekId: range.weekId,
            label: range.label,
            stats: context.stats,
            manualReview: truncateText(currentReview.manualReview, 1600)
        },
        previousWeek: {
            weekId: previousRange.weekId,
            label: previousRange.label,
            stats: context.previousStats,
            aiSummary: truncateText(
                context.previousReview?.aiInsight?.summaryText || context.previousReview?.aiSummary || "",
                900
            ),
            manualReview: truncateText(context.previousReview?.manualReview || "", 700)
        },
        thisWeekContent: {
            drafts: context.drafts,
            exercises: context.exercises,
            materials: context.materials,
            readings: context.readings
        },
        outputLanguage: "中文"
    };

    const responseMeta = await createOpenAIJsonResponse({
        input: JSON.stringify(promptInput, null, 2),
        schema: WEEKLY_INSIGHT_SCHEMA,
        schemaName: "writing_weekly_insight",
        maxOutputTokens: 1800
    });
    const aiInsight = normalizeWeeklyInsight(responseMeta.data, responseMeta);

    await reviewRef.set({
        weekStart: range.weekStartDate,
        weekEnd: range.weekEndDate,
        manualReview: currentReview.manualReview || "",
        aiInsight,
        aiSummary: aiInsight.summaryText,
        statsSnapshot: {
            draftWords: context.stats.draftWords,
            exerciseWords: context.stats.exerciseWords,
            totalWords: context.stats.totalWords,
            completedExercises: context.stats.completedExercises,
            streak: context.stats.streak
        },
        updatedAt: admin.firestore.FieldValue.serverTimestamp()
    }, { merge: true });

    return { aiInsight, aiSummary: aiInsight.summaryText };
});

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
    console.log("Scheduled report disabled by user request.");
    return; // Feature temporarily turned off

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
