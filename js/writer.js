document.addEventListener('DOMContentLoaded', () => {
    const $ = (selector) => document.querySelector(selector);

    const MATERIAL_LABELS = {
        character: '人物',
        place: '地点',
        worldbuilding: '设定',
        fragment: '片段',
        dialogue: '对话',
        idea: '灵感'
    };

    const PROMPT_BANK = [
        {
            focus: '人物欲望',
            prompt: '写一个角色想得到一件小东西，但另一个人误以为那东西毫不重要。让欲望在行动里显露。'
        },
        {
            focus: '冲突',
            prompt: '写两个人在同一间房里等待一个结果。两人都想保持冷静，但只有一个人知道真相。'
        },
        {
            focus: '对白',
            prompt: '写一段不直接说“我生气了”的争吵。让潜台词比台词更响。'
        },
        {
            focus: '氛围',
            prompt: '写一个深夜回家的场景。不要解释恐惧，只写物件、声音和身体反应。'
        },
        {
            focus: '转折',
            prompt: '写一个角色终于拿到想要的答案，却发现这个答案改变了问题本身。'
        },
        {
            focus: '节奏',
            prompt: '写一个短场景，前半段尽量慢，后半段用三次动作把速度推起来。'
        }
    ];

    const els = {
        bootView: $('#bootView'),
        authView: $('#authView'),
        appView: $('#appView'),
        authForm: $('#authForm'),
        authEmail: $('#authEmail'),
        authPassword: $('#authPassword'),
        authMessage: $('#authMessage'),
        googleSignInBtn: $('#googleSignInBtn'),
        createAccountBtn: $('#createAccountBtn'),
        resetPasswordBtn: $('#resetPasswordBtn'),
        logoutBtn: $('#logoutBtn'),
        userEmail: $('#userEmail'),
        topWeekWords: $('#topWeekWords'),
        writerNav: $('#writerNav'),
        draftCount: $('#draftCount'),
        exerciseCount: $('#exerciseCount'),
        materialCount: $('#materialCount'),
        readingCount: $('#readingCount'),
        newDraftBtn: $('#newDraftBtn'),
        draftList: $('#draftList'),
        draftEmpty: $('#draftEmpty'),
        draftEditor: $('#draftEditor'),
        draftTitle: $('#draftTitle'),
        draftTags: $('#draftTags'),
        draftBody: $('#draftBody'),
        draftWordCount: $('#draftWordCount'),
        draftSaveState: $('#draftSaveState'),
        deleteDraftBtn: $('#deleteDraftBtn'),
        newExerciseBtn: $('#newExerciseBtn'),
        exerciseList: $('#exerciseList'),
        exerciseEmpty: $('#exerciseEmpty'),
        exerciseEditor: $('#exerciseEditor'),
        exerciseFocus: $('#exerciseFocus'),
        exercisePrompt: $('#exercisePrompt'),
        exerciseBody: $('#exerciseBody'),
        exerciseWordCount: $('#exerciseWordCount'),
        exerciseSaveState: $('#exerciseSaveState'),
        exerciseStatusBadge: $('#exerciseStatusBadge'),
        completeExerciseBtn: $('#completeExerciseBtn'),
        deleteExerciseBtn: $('#deleteExerciseBtn'),
        newMaterialBtn: $('#newMaterialBtn'),
        materialTypeFilter: $('#materialTypeFilter'),
        materialSearch: $('#materialSearch'),
        materialList: $('#materialList'),
        materialEmpty: $('#materialEmpty'),
        materialEditor: $('#materialEditor'),
        materialType: $('#materialType'),
        materialTitle: $('#materialTitle'),
        materialTags: $('#materialTags'),
        materialContent: $('#materialContent'),
        materialSaveState: $('#materialSaveState'),
        deleteMaterialBtn: $('#deleteMaterialBtn'),
        newReadingBtn: $('#newReadingBtn'),
        readingList: $('#readingList'),
        readingEmpty: $('#readingEmpty'),
        readingEditor: $('#readingEditor'),
        readingSourceTitle: $('#readingSourceTitle'),
        readingAuthor: $('#readingAuthor'),
        readingLens: $('#readingLens'),
        readingTags: $('#readingTags'),
        readingExcerpt: $('#readingExcerpt'),
        readingNotes: $('#readingNotes'),
        readingSaveState: $('#readingSaveState'),
        deleteReadingBtn: $('#deleteReadingBtn'),
        weekLabel: $('#weekLabel'),
        statsTotalWords: $('#statsTotalWords'),
        statsDraftWords: $('#statsDraftWords'),
        statsExerciseWords: $('#statsExerciseWords'),
        statsCompleted: $('#statsCompleted'),
        statsStreak: $('#statsStreak'),
        dailyBars: $('#dailyBars'),
        weeklyReview: $('#weeklyReview'),
        weeklyReviewSaveState: $('#weeklyReviewSaveState'),
        aiSummary: $('#aiSummary')
    };

    const state = {
        auth: null,
        db: null,
        user: null,
        view: localStorage.getItem('writerView') || 'drafts',
        drafts: [],
        exercises: [],
        materials: [],
        readings: [],
        stats: [],
        weeklyReviews: [],
        activeDraftId: localStorage.getItem('writerActiveDraft') || null,
        activeExerciseId: localStorage.getItem('writerActiveExercise') || null,
        activeMaterialId: localStorage.getItem('writerActiveMaterial') || null,
        activeReadingId: localStorage.getItem('writerActiveReading') || null,
        materialFilter: 'all',
        materialQuery: '',
        promptIndex: Number(localStorage.getItem('writerPromptIndex') || '0'),
        unsubs: [],
        timers: {},
        savedCounts: {
            drafts: new Map(),
            exercises: new Map()
        }
    };

    init();

    function init() {
        bindStaticEvents();

        try {
            if (!window.firebaseConfig || !window.firebaseConfig.apiKey) {
                throw new Error('Firebase 配置缺失。');
            }
            if (!firebase.apps.length) {
                firebase.initializeApp(window.firebaseConfig);
            }
            state.auth = firebase.auth();
            state.db = firebase.firestore();
            state.auth.onAuthStateChanged(handleAuthChange);
        } catch (error) {
            showAuthMessage(error.message || '写作空间初始化失败。');
            showSignedOut();
        }
    }

    function bindStaticEvents() {
        els.authForm.addEventListener('submit', (event) => {
            event.preventDefault();
            signInWithEmail();
        });
        els.googleSignInBtn.addEventListener('click', signInWithGoogle);
        els.createAccountBtn.addEventListener('click', createAccount);
        els.resetPasswordBtn.addEventListener('click', resetPassword);
        els.logoutBtn.addEventListener('click', () => state.auth.signOut());

        els.writerNav.addEventListener('click', (event) => {
            const tab = event.target.closest('[data-view]');
            if (!tab) return;
            setView(tab.getAttribute('data-view'));
        });

        els.newDraftBtn.addEventListener('click', createDraft);
        els.draftList.addEventListener('click', (event) => {
            const item = event.target.closest('[data-draft-id]');
            if (!item) return;
            state.activeDraftId = item.getAttribute('data-draft-id');
            localStorage.setItem('writerActiveDraft', state.activeDraftId);
            renderDrafts(true);
        });
        [els.draftTitle, els.draftTags, els.draftBody].forEach((input) => {
            input.addEventListener('input', scheduleDraftSave);
        });
        els.deleteDraftBtn.addEventListener('click', deleteActiveDraft);

        els.newExerciseBtn.addEventListener('click', createExercise);
        els.exerciseList.addEventListener('click', (event) => {
            const item = event.target.closest('[data-exercise-id]');
            if (!item) return;
            state.activeExerciseId = item.getAttribute('data-exercise-id');
            localStorage.setItem('writerActiveExercise', state.activeExerciseId);
            renderExercises(true);
        });
        [els.exerciseFocus, els.exercisePrompt, els.exerciseBody].forEach((input) => {
            input.addEventListener('input', scheduleExerciseSave);
            input.addEventListener('change', scheduleExerciseSave);
        });
        els.completeExerciseBtn.addEventListener('click', toggleExerciseDone);
        els.deleteExerciseBtn.addEventListener('click', deleteActiveExercise);

        els.materialTypeFilter.addEventListener('change', () => {
            state.materialFilter = els.materialTypeFilter.value;
            renderMaterials();
        });
        els.materialSearch.addEventListener('input', () => {
            state.materialQuery = els.materialSearch.value.trim().toLowerCase();
            renderMaterials();
        });
        els.newMaterialBtn.addEventListener('click', createMaterial);
        els.materialList.addEventListener('click', (event) => {
            const item = event.target.closest('[data-material-id]');
            if (!item) return;
            state.activeMaterialId = item.getAttribute('data-material-id');
            localStorage.setItem('writerActiveMaterial', state.activeMaterialId);
            renderMaterials(true);
        });
        [els.materialType, els.materialTitle, els.materialTags, els.materialContent].forEach((input) => {
            input.addEventListener('input', scheduleMaterialSave);
            input.addEventListener('change', scheduleMaterialSave);
        });
        els.deleteMaterialBtn.addEventListener('click', deleteActiveMaterial);

        els.newReadingBtn.addEventListener('click', createReading);
        els.readingList.addEventListener('click', (event) => {
            const item = event.target.closest('[data-reading-id]');
            if (!item) return;
            state.activeReadingId = item.getAttribute('data-reading-id');
            localStorage.setItem('writerActiveReading', state.activeReadingId);
            renderReading(true);
        });
        [
            els.readingSourceTitle,
            els.readingAuthor,
            els.readingLens,
            els.readingTags,
            els.readingExcerpt,
            els.readingNotes
        ].forEach((input) => {
            input.addEventListener('input', scheduleReadingSave);
            input.addEventListener('change', scheduleReadingSave);
        });
        els.deleteReadingBtn.addEventListener('click', deleteActiveReading);

        els.weeklyReview.addEventListener('input', scheduleWeeklyReviewSave);
    }

    async function handleAuthChange(user) {
        clearTimers();
        cleanupListeners();
        state.user = user;

        if (!user) {
            showSignedOut();
            return;
        }

        if (needsEmailVerification(user)) {
            await state.auth.signOut();
            showSignedOut();
            showAuthMessage('请先完成邮箱验证，再进入写作空间。');
            return;
        }

        showSignedIn(user);
        await state.db.collection('users').doc(user.uid).set({
            email: user.email || ''
        }, { merge: true });
        setupListeners();
    }

    function showSignedOut() {
        els.bootView.classList.add('hidden');
        els.authView.classList.remove('hidden');
        els.appView.classList.add('hidden');
    }

    function showSignedIn(user) {
        els.bootView.classList.add('hidden');
        els.authView.classList.add('hidden');
        els.appView.classList.remove('hidden');
        els.userEmail.textContent = user.email || '已登录';
        setView(state.view);
    }

    async function signInWithEmail() {
        const email = els.authEmail.value.trim();
        const password = els.authPassword.value;
        if (!email || !password) {
            showAuthMessage('请输入邮箱和密码。');
            return;
        }
        try {
            showAuthMessage('');
            await state.auth.signInWithEmailAndPassword(email, password);
        } catch (error) {
            showAuthMessage(normalizeAuthError(error));
        }
    }

    async function signInWithGoogle() {
        try {
            showAuthMessage('');
            const provider = new firebase.auth.GoogleAuthProvider();
            await state.auth.signInWithPopup(provider);
        } catch (error) {
            showAuthMessage(normalizeAuthError(error));
        }
    }

    async function createAccount() {
        const email = els.authEmail.value.trim();
        const password = els.authPassword.value;
        if (!email || !password) {
            showAuthMessage('请输入邮箱和密码。');
            return;
        }
        try {
            showAuthMessage('');
            const credential = await state.auth.createUserWithEmailAndPassword(email, password);
            if (credential.user && credential.user.sendEmailVerification) {
                await credential.user.sendEmailVerification();
            }
            await state.auth.signOut();
            showAuthMessage('验证邮件已发送，请验证后再登录。');
        } catch (error) {
            showAuthMessage(normalizeAuthError(error));
        }
    }

    async function resetPassword() {
        const email = els.authEmail.value.trim();
        if (!email) {
            showAuthMessage('请输入邮箱。');
            return;
        }
        try {
            await state.auth.sendPasswordResetEmail(email);
            showAuthMessage('重置邮件已发送。');
        } catch (error) {
            showAuthMessage(normalizeAuthError(error));
        }
    }

    function setupListeners() {
        listenToCollection('writingDrafts', (docs) => {
            state.drafts = sortByUpdated(docs);
            seedSavedCounts('drafts', state.drafts);
            ensureActive('drafts');
            renderShell();
            renderDrafts();
            renderReview();
        });

        listenToCollection('writingExercises', (docs) => {
            state.exercises = sortByUpdated(docs);
            seedSavedCounts('exercises', state.exercises);
            ensureActive('exercises');
            renderShell();
            renderExercises();
            renderReview();
        });

        listenToCollection('writingMaterials', (docs) => {
            state.materials = sortByUpdated(docs);
            ensureActive('materials');
            renderShell();
            renderMaterials();
        });

        listenToCollection('readingBreakdowns', (docs) => {
            state.readings = sortByUpdated(docs);
            ensureActive('reading');
            renderShell();
            renderReading();
        });

        listenToCollection('writingStats', (docs) => {
            state.stats = docs;
            renderShell();
            renderReview();
        });

        listenToCollection('writingWeeklyReviews', (docs) => {
            state.weeklyReviews = docs;
            renderReview();
        });
    }

    function listenToCollection(collectionName, onData) {
        const unsub = getUserRef(collectionName).onSnapshot((snapshot) => {
            onData(snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() })));
        }, (error) => {
            console.error(`Failed to read ${collectionName}`, error);
        });
        state.unsubs.push(unsub);
    }

    function cleanupListeners() {
        state.unsubs.forEach((unsub) => unsub());
        state.unsubs = [];
    }

    function setView(view) {
        state.view = view || 'drafts';
        localStorage.setItem('writerView', state.view);
        document.querySelectorAll('.writer-view').forEach((panel) => {
            panel.classList.toggle('active', panel.id === `view-${state.view}`);
        });
        els.writerNav.querySelectorAll('.nav-tab').forEach((tab) => {
            tab.classList.toggle('active', tab.getAttribute('data-view') === state.view);
        });
        renderCurrentView(true);
    }

    function renderShell() {
        els.draftCount.textContent = state.drafts.length;
        els.exerciseCount.textContent = state.exercises.length;
        els.materialCount.textContent = state.materials.length;
        els.readingCount.textContent = state.readings.length;
        els.topWeekWords.textContent = `${formatNumber(getWeeklyStats().totalWords)} 字 / 本周`;
    }

    function renderCurrentView(forcePopulate = false) {
        if (state.view === 'drafts') renderDrafts(forcePopulate);
        if (state.view === 'exercises') renderExercises(forcePopulate);
        if (state.view === 'materials') renderMaterials(forcePopulate);
        if (state.view === 'reading') renderReading(forcePopulate);
        if (state.view === 'review') renderReview(forcePopulate);
    }

    function renderDrafts(forcePopulate = false) {
        renderList(els.draftList, state.drafts, state.activeDraftId, 'draft');
        const draft = getActiveDraft();
        toggleEditor(els.draftEmpty, els.draftEditor, Boolean(draft));
        if (!draft) return;
        if (forcePopulate || !isFocusedWithin([els.draftTitle, els.draftTags, els.draftBody])) {
            els.draftTitle.value = draft.title || '';
            els.draftTags.value = tagsToText(draft.tags);
            els.draftBody.value = draft.body || '';
            els.draftSaveState.textContent = '已保存';
        }
        els.draftWordCount.textContent = `${formatNumber(countWords(els.draftBody.value))} 字`;
    }

    async function createDraft() {
        const ref = getUserRef('writingDrafts').doc();
        await ref.set({
            title: '未命名草稿',
            body: '',
            tags: [],
            wordCount: 0,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
        });
        state.activeDraftId = ref.id;
        localStorage.setItem('writerActiveDraft', ref.id);
        setView('drafts');
    }

    function scheduleDraftSave() {
        const wordCount = countWords(els.draftBody.value);
        els.draftWordCount.textContent = `${formatNumber(wordCount)} 字`;
        debounce('draft', saveDraft, els.draftSaveState);
    }

    async function saveDraft() {
        const draft = getActiveDraft();
        if (!draft) return;
        const wordCount = countWords(els.draftBody.value);
        const previousCount = getSavedCount('drafts', draft);
        await getUserRef('writingDrafts').doc(draft.id).set({
            title: els.draftTitle.value.trim() || '未命名草稿',
            body: els.draftBody.value,
            tags: parseTags(els.draftTags.value),
            wordCount,
            updatedAt: serverTimestamp()
        }, { merge: true });
        setSavedCount('drafts', draft.id, wordCount);
        await recordWritingDelta('draftWords', wordCount - previousCount);
        els.draftSaveState.textContent = '已保存';
    }

    async function deleteActiveDraft() {
        const draft = getActiveDraft();
        if (!draft || !confirm('删除这篇草稿？')) return;
        await getUserRef('writingDrafts').doc(draft.id).delete();
        state.activeDraftId = null;
        localStorage.removeItem('writerActiveDraft');
    }

    function renderExercises(forcePopulate = false) {
        renderList(els.exerciseList, state.exercises, state.activeExerciseId, 'exercise');
        const exercise = getActiveExercise();
        toggleEditor(els.exerciseEmpty, els.exerciseEditor, Boolean(exercise));
        if (!exercise) return;
        if (forcePopulate || !isFocusedWithin([els.exerciseFocus, els.exercisePrompt, els.exerciseBody])) {
            els.exerciseFocus.value = exercise.focus || '人物欲望';
            els.exercisePrompt.value = exercise.prompt || '';
            els.exerciseBody.value = exercise.body || '';
            els.exerciseSaveState.textContent = '已保存';
        }
        const done = exercise.status === 'done';
        els.exerciseStatusBadge.textContent = done ? '已完成' : '草稿';
        els.exerciseStatusBadge.classList.toggle('done', done);
        els.completeExerciseBtn.textContent = done ? '重新打开' : '标记完成';
        els.exerciseWordCount.textContent = `${formatNumber(countWords(els.exerciseBody.value))} 字`;
    }

    async function createExercise() {
        const prompt = PROMPT_BANK[state.promptIndex % PROMPT_BANK.length];
        state.promptIndex += 1;
        localStorage.setItem('writerPromptIndex', String(state.promptIndex));
        const ref = getUserRef('writingExercises').doc();
        await ref.set({
            prompt: prompt.prompt,
            focus: prompt.focus,
            body: '',
            status: 'draft',
            wordCount: 0,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
            completedAt: null
        });
        state.activeExerciseId = ref.id;
        localStorage.setItem('writerActiveExercise', ref.id);
        setView('exercises');
    }

    function scheduleExerciseSave() {
        const wordCount = countWords(els.exerciseBody.value);
        els.exerciseWordCount.textContent = `${formatNumber(wordCount)} 字`;
        debounce('exercise', saveExercise, els.exerciseSaveState);
    }

    async function saveExercise(options = {}) {
        const exercise = getActiveExercise();
        if (!exercise) return;
        const wordCount = countWords(els.exerciseBody.value);
        const previousCount = getSavedCount('exercises', exercise);
        const data = {
            prompt: els.exercisePrompt.value.trim(),
            focus: els.exerciseFocus.value,
            body: els.exerciseBody.value,
            wordCount,
            updatedAt: serverTimestamp()
        };
        if (options.status) {
            data.status = options.status;
            data.completedAt = options.status === 'done' ? serverTimestamp() : null;
        }
        await getUserRef('writingExercises').doc(exercise.id).set(data, { merge: true });
        setSavedCount('exercises', exercise.id, wordCount);
        await recordWritingDelta('exerciseWords', wordCount - previousCount);
        els.exerciseSaveState.textContent = '已保存';
    }

    async function toggleExerciseDone() {
        const exercise = getActiveExercise();
        if (!exercise) return;
        const nextStatus = exercise.status === 'done' ? 'draft' : 'done';
        await saveExercise({ status: nextStatus });
    }

    async function deleteActiveExercise() {
        const exercise = getActiveExercise();
        if (!exercise || !confirm('删除这条练习？')) return;
        await getUserRef('writingExercises').doc(exercise.id).delete();
        state.activeExerciseId = null;
        localStorage.removeItem('writerActiveExercise');
    }

    function renderMaterials(forcePopulate = false) {
        const filtered = getFilteredMaterials();
        if (!filtered.some((item) => item.id === state.activeMaterialId)) {
            state.activeMaterialId = filtered[0]?.id || null;
            if (state.activeMaterialId) localStorage.setItem('writerActiveMaterial', state.activeMaterialId);
            else localStorage.removeItem('writerActiveMaterial');
        }
        renderList(els.materialList, filtered, state.activeMaterialId, 'material');
        const material = getActiveMaterial();
        toggleEditor(els.materialEmpty, els.materialEditor, Boolean(material));
        if (!material) return;
        if (forcePopulate || !isFocusedWithin([els.materialType, els.materialTitle, els.materialTags, els.materialContent])) {
            els.materialType.value = material.type || 'idea';
            els.materialTitle.value = material.title || '';
            els.materialTags.value = tagsToText(material.tags);
            els.materialContent.value = material.content || '';
            els.materialSaveState.textContent = '已保存';
        }
    }

    async function createMaterial() {
        const type = state.materialFilter === 'all' ? 'idea' : state.materialFilter;
        const ref = getUserRef('writingMaterials').doc();
        await ref.set({
            type,
            title: `${MATERIAL_LABELS[type] || '素材'}记录`,
            content: '',
            tags: [],
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
        });
        state.activeMaterialId = ref.id;
        localStorage.setItem('writerActiveMaterial', ref.id);
        setView('materials');
    }

    function scheduleMaterialSave() {
        debounce('material', saveMaterial, els.materialSaveState);
    }

    async function saveMaterial() {
        const material = getActiveMaterial();
        if (!material) return;
        await getUserRef('writingMaterials').doc(material.id).set({
            type: els.materialType.value,
            title: els.materialTitle.value.trim() || '未命名素材',
            content: els.materialContent.value,
            tags: parseTags(els.materialTags.value),
            updatedAt: serverTimestamp()
        }, { merge: true });
        els.materialSaveState.textContent = '已保存';
    }

    async function deleteActiveMaterial() {
        const material = getActiveMaterial();
        if (!material || !confirm('删除这条素材？')) return;
        await getUserRef('writingMaterials').doc(material.id).delete();
        state.activeMaterialId = null;
        localStorage.removeItem('writerActiveMaterial');
    }

    function renderReading(forcePopulate = false) {
        renderList(els.readingList, state.readings, state.activeReadingId, 'reading');
        const reading = getActiveReading();
        toggleEditor(els.readingEmpty, els.readingEditor, Boolean(reading));
        if (!reading) return;
        if (forcePopulate || !isFocusedWithin([
            els.readingSourceTitle,
            els.readingAuthor,
            els.readingLens,
            els.readingTags,
            els.readingExcerpt,
            els.readingNotes
        ])) {
            els.readingSourceTitle.value = reading.sourceTitle || '';
            els.readingAuthor.value = reading.author || '';
            els.readingLens.value = reading.lens || '人物';
            els.readingTags.value = tagsToText(reading.tags);
            els.readingExcerpt.value = reading.excerpt || '';
            els.readingNotes.value = reading.notes || '';
            els.readingSaveState.textContent = '已保存';
        }
    }

    async function createReading() {
        const ref = getUserRef('readingBreakdowns').doc();
        await ref.set({
            sourceTitle: '未命名作品',
            author: '',
            excerpt: '',
            lens: '人物',
            notes: '',
            tags: [],
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
        });
        state.activeReadingId = ref.id;
        localStorage.setItem('writerActiveReading', ref.id);
        setView('reading');
    }

    function scheduleReadingSave() {
        debounce('reading', saveReading, els.readingSaveState);
    }

    async function saveReading() {
        const reading = getActiveReading();
        if (!reading) return;
        await getUserRef('readingBreakdowns').doc(reading.id).set({
            sourceTitle: els.readingSourceTitle.value.trim() || '未命名作品',
            author: els.readingAuthor.value.trim(),
            excerpt: els.readingExcerpt.value,
            lens: els.readingLens.value,
            notes: els.readingNotes.value,
            tags: parseTags(els.readingTags.value),
            updatedAt: serverTimestamp()
        }, { merge: true });
        els.readingSaveState.textContent = '已保存';
    }

    async function deleteActiveReading() {
        const reading = getActiveReading();
        if (!reading || !confirm('删除这条拆解？')) return;
        await getUserRef('readingBreakdowns').doc(reading.id).delete();
        state.activeReadingId = null;
        localStorage.removeItem('writerActiveReading');
    }

    function renderReview(forcePopulate = false) {
        const range = getWeekRange();
        const stats = getWeeklyStats();
        const review = getCurrentWeeklyReview();

        els.weekLabel.textContent = `${range.startLabel} - ${range.endLabel}`;
        els.statsTotalWords.textContent = formatNumber(stats.totalWords);
        els.statsDraftWords.textContent = formatNumber(stats.draftWords);
        els.statsExerciseWords.textContent = formatNumber(stats.exerciseWords);
        els.statsCompleted.textContent = formatNumber(stats.completedExercises);
        els.statsStreak.textContent = formatNumber(stats.streak);
        els.topWeekWords.textContent = `${formatNumber(stats.totalWords)} 字 / 本周`;
        renderDailyBars(stats.days);

        if (forcePopulate || !isFocusedWithin([els.weeklyReview])) {
            els.weeklyReview.value = review?.manualReview || '';
            els.aiSummary.value = review?.aiSummary || '';
            els.weeklyReviewSaveState.textContent = '已保存';
        }
    }

    function scheduleWeeklyReviewSave() {
        debounce('weeklyReview', saveWeeklyReview, els.weeklyReviewSaveState);
    }

    async function saveWeeklyReview() {
        const range = getWeekRange();
        const stats = getWeeklyStats();
        const current = getCurrentWeeklyReview();
        await getUserRef('writingWeeklyReviews').doc(range.weekId).set({
            weekStart: range.weekStartDate,
            weekEnd: range.weekEndDate,
            manualReview: els.weeklyReview.value,
            aiSummary: current?.aiSummary || '',
            statsSnapshot: {
                draftWords: stats.draftWords,
                exerciseWords: stats.exerciseWords,
                totalWords: stats.totalWords,
                completedExercises: stats.completedExercises,
                streak: stats.streak
            },
            updatedAt: serverTimestamp()
        }, { merge: true });
        els.weeklyReviewSaveState.textContent = '已保存';
    }

    async function recordWritingDelta(field, delta) {
        const safeDelta = Math.max(0, Number(delta) || 0);
        if (!safeDelta || !state.user) return;
        const today = getDateParts(new Date());
        await getUserRef('writingStats').doc(today.id).set({
            date: today.id,
            draftWords: firebase.firestore.FieldValue.increment(field === 'draftWords' ? safeDelta : 0),
            exerciseWords: firebase.firestore.FieldValue.increment(field === 'exerciseWords' ? safeDelta : 0),
            totalWords: firebase.firestore.FieldValue.increment(safeDelta),
            sessions: firebase.firestore.FieldValue.increment(1),
            updatedAt: serverTimestamp()
        }, { merge: true });
    }

    function renderList(container, items, activeId, type) {
        if (!items.length) {
            container.innerHTML = `<div class="empty-state"><p>暂无记录</p></div>`;
            return;
        }

        container.innerHTML = items.map((item) => {
            const title = getItemTitle(item, type);
            const meta = getItemMeta(item, type);
            const attr = `data-${type}-id`;
            return `
                <button class="list-item ${item.id === activeId ? 'active' : ''}" ${attr}="${escapeAttr(item.id)}" type="button">
                    <span class="list-title">${escapeHtml(title)}</span>
                    <span class="list-meta">${escapeHtml(meta)}</span>
                </button>
            `;
        }).join('');
    }

    function renderDailyBars(days) {
        const maxWords = Math.max(1, ...days.map((day) => day.totalWords));
        els.dailyBars.innerHTML = days.map((day) => {
            const height = Math.max(4, Math.round((day.totalWords / maxWords) * 76));
            return `
                <div class="day-bar">
                    <div class="day-bar-fill" style="height: ${height}px"></div>
                    <div class="day-bar-meta">
                        <span>${escapeHtml(day.label)}</span>
                        <strong>${formatNumber(day.totalWords)}</strong>
                    </div>
                </div>
            `;
        }).join('');
    }

    function getWeeklyStats() {
        const range = getWeekRange();
        const dayMap = new Map();
        for (let i = 0; i < 7; i += 1) {
            const date = new Date(range.weekStart);
            date.setDate(range.weekStart.getDate() + i);
            const parts = getDateParts(date);
            dayMap.set(parts.id, {
                id: parts.id,
                label: parts.shortLabel,
                draftWords: 0,
                exerciseWords: 0,
                totalWords: 0
            });
        }

        state.stats.forEach((entry) => {
            const day = dayMap.get(entry.date);
            if (!day) return;
            day.draftWords += Number(entry.draftWords || 0);
            day.exerciseWords += Number(entry.exerciseWords || 0);
            day.totalWords += Number(entry.totalWords || 0);
        });

        const days = Array.from(dayMap.values());
        const draftWords = days.reduce((sum, day) => sum + day.draftWords, 0);
        const exerciseWords = days.reduce((sum, day) => sum + day.exerciseWords, 0);
        const totalWords = days.reduce((sum, day) => sum + day.totalWords, 0);
        const completedExercises = state.exercises.filter((exercise) => {
            if (exercise.status !== 'done') return false;
            const completed = timestampToDate(exercise.completedAt);
            return completed && completed >= range.weekStart && completed <= range.weekEnd;
        }).length;

        return {
            days,
            draftWords,
            exerciseWords,
            totalWords,
            completedExercises,
            streak: getWritingStreak()
        };
    }

    function getWritingStreak() {
        const writtenDays = new Set(
            state.stats
                .filter((entry) => Number(entry.totalWords || 0) > 0)
                .map((entry) => entry.date)
        );
        let streak = 0;
        const cursor = new Date();
        while (writtenDays.has(getDateParts(cursor).id)) {
            streak += 1;
            cursor.setDate(cursor.getDate() - 1);
        }
        return streak;
    }

    function getCurrentWeeklyReview() {
        const weekId = getWeekRange().weekId;
        return state.weeklyReviews.find((review) => review.id === weekId) || null;
    }

    function getFilteredMaterials() {
        return state.materials.filter((material) => {
            const typeMatch = state.materialFilter === 'all' || material.type === state.materialFilter;
            if (!typeMatch) return false;
            if (!state.materialQuery) return true;
            const haystack = [
                material.title,
                material.content,
                tagsToText(material.tags),
                MATERIAL_LABELS[material.type]
            ].join(' ').toLowerCase();
            return haystack.includes(state.materialQuery);
        });
    }

    function getItemTitle(item, type) {
        if (type === 'draft') return item.title || '未命名草稿';
        if (type === 'exercise') return item.focus || '场景练习';
        if (type === 'material') return item.title || '未命名素材';
        if (type === 'reading') return item.sourceTitle || '未命名作品';
        return '未命名';
    }

    function getItemMeta(item, type) {
        const date = formatDate(item.updatedAt || item.createdAt);
        if (type === 'draft') return `${formatNumber(item.wordCount || 0)} 字 / ${date}`;
        if (type === 'exercise') return `${item.status === 'done' ? '已完成' : '草稿'} / ${formatNumber(item.wordCount || 0)} 字`;
        if (type === 'material') return `${MATERIAL_LABELS[item.type] || '素材'} / ${date}`;
        if (type === 'reading') return `${item.lens || '拆解'} / ${date}`;
        return date;
    }

    function ensureActive(type) {
        const config = {
            drafts: ['activeDraftId', state.drafts, 'writerActiveDraft'],
            exercises: ['activeExerciseId', state.exercises, 'writerActiveExercise'],
            materials: ['activeMaterialId', state.materials, 'writerActiveMaterial'],
            reading: ['activeReadingId', state.readings, 'writerActiveReading']
        }[type];
        if (!config) return;
        const [key, items, storageKey] = config;
        if (items.some((item) => item.id === state[key])) return;
        state[key] = items[0]?.id || null;
        if (state[key]) localStorage.setItem(storageKey, state[key]);
        else localStorage.removeItem(storageKey);
    }

    function seedSavedCounts(type, items) {
        items.forEach((item) => {
            if (!state.savedCounts[type].has(item.id)) {
                state.savedCounts[type].set(item.id, Number(item.wordCount || 0));
            }
        });
    }

    function getSavedCount(type, item) {
        if (!state.savedCounts[type].has(item.id)) {
            state.savedCounts[type].set(item.id, Number(item.wordCount || 0));
        }
        return Number(state.savedCounts[type].get(item.id) || 0);
    }

    function setSavedCount(type, id, count) {
        state.savedCounts[type].set(id, Number(count || 0));
    }

    function getActiveDraft() {
        return state.drafts.find((draft) => draft.id === state.activeDraftId) || null;
    }

    function getActiveExercise() {
        return state.exercises.find((exercise) => exercise.id === state.activeExerciseId) || null;
    }

    function getActiveMaterial() {
        return state.materials.find((material) => material.id === state.activeMaterialId) || null;
    }

    function getActiveReading() {
        return state.readings.find((reading) => reading.id === state.activeReadingId) || null;
    }

    function toggleEditor(emptyEl, editorEl, hasItem) {
        emptyEl.classList.toggle('hidden', hasItem);
        editorEl.classList.toggle('hidden', !hasItem);
    }

    function debounce(key, callback, statusEl) {
        clearTimeout(state.timers[key]);
        statusEl.textContent = '保存中...';
        state.timers[key] = setTimeout(async () => {
            try {
                await callback();
            } catch (error) {
                console.error(error);
                statusEl.textContent = '保存失败';
            }
        }, 650);
    }

    function clearTimers() {
        Object.values(state.timers).forEach((timer) => clearTimeout(timer));
        state.timers = {};
    }

    function getUserRef(collectionName) {
        return state.db.collection('users').doc(state.user.uid).collection(collectionName);
    }

    function serverTimestamp() {
        return firebase.firestore.FieldValue.serverTimestamp();
    }

    function sortByUpdated(items) {
        return [...items].sort((a, b) => toMillis(b.updatedAt || b.createdAt) - toMillis(a.updatedAt || a.createdAt));
    }

    function parseTags(value) {
        return value
            .split(/[，,]/)
            .map((tag) => tag.trim())
            .filter(Boolean)
            .slice(0, 12);
    }

    function tagsToText(tags) {
        return Array.isArray(tags) ? tags.join(', ') : '';
    }

    function countWords(text) {
        const value = (text || '').trim();
        if (!value) return 0;
        const cjkMatches = value.match(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g) || [];
        const latinText = value.replace(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g, ' ');
        const latinMatches = latinText.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g) || [];
        return cjkMatches.length + latinMatches.length;
    }

    function getWeekRange() {
        const now = new Date();
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
            startLabel: startParts.label,
            endLabel: endParts.label
        };
    }

    function getDateParts(date) {
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        return {
            id: `${y}-${m}-${d}`,
            label: `${m}/${d}`,
            shortLabel: `${m}.${d}`
        };
    }

    function formatDate(value) {
        const date = timestampToDate(value);
        if (!date) return '刚刚';
        const parts = getDateParts(date);
        return parts.label;
    }

    function timestampToDate(value) {
        if (!value) return null;
        if (typeof value.toDate === 'function') return value.toDate();
        if (value instanceof Date) return value;
        if (typeof value === 'number') return new Date(value);
        return null;
    }

    function toMillis(value) {
        const date = timestampToDate(value);
        return date ? date.getTime() : 0;
    }

    function formatNumber(value) {
        return new Intl.NumberFormat('zh-CN').format(Number(value || 0));
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function escapeAttr(value) {
        return escapeHtml(value).replace(/`/g, '&#096;');
    }

    function isFocusedWithin(elements) {
        return elements.includes(document.activeElement);
    }

    function showAuthMessage(message) {
        els.authMessage.textContent = message || '';
    }

    function normalizeAuthError(error) {
        const code = error?.code || '';
        if (code.includes('wrong-password') || code.includes('invalid-credential')) return '邮箱或密码不正确。';
        if (code.includes('user-not-found')) return '没有找到这个账号。';
        if (code.includes('email-already-in-use')) return '这个邮箱已经注册。';
        if (code.includes('weak-password')) return '密码至少需要 6 位。';
        if (code.includes('popup-closed-by-user')) return '登录窗口已关闭。';
        return error?.message || '认证失败。';
    }

    function needsEmailVerification(user) {
        const usesPassword = (user.providerData || []).some((provider) => provider.providerId === 'password');
        return usesPassword && user.emailVerified === false;
    }
});
