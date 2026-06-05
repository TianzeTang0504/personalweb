document.addEventListener('DOMContentLoaded', () => {
    const $ = (selector) => document.querySelector(selector);

    const CUSTOM_OPTION = '__custom__';
    const DEFAULT_EXERCISE_FOCUSES = ['人物欲望', '冲突', '对白', '氛围', '转折', '节奏'];
    const DEFAULT_READING_LENSES = ['人物', '冲突', '节奏', '氛围', '句子', '结尾'];
    const DEFAULT_MATERIAL_TYPES = [
        { id: 'character', label: '人物', builtin: true },
        { id: 'place', label: '地点', builtin: true },
        { id: 'worldbuilding', label: '设定', builtin: true },
        { id: 'fragment', label: '片段', builtin: true },
        { id: 'dialogue', label: '对话', builtin: true },
        { id: 'idea', label: '灵感', builtin: true }
    ];

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
        draftTagEditor: $('#draftTagEditor'),
        draftBody: $('#draftBody'),
        draftWordCount: $('#draftWordCount'),
        draftSaveState: $('#draftSaveState'),
        deleteDraftBtn: $('#deleteDraftBtn'),
        newExerciseBtn: $('#newExerciseBtn'),
        exerciseList: $('#exerciseList'),
        exerciseEmpty: $('#exerciseEmpty'),
        exerciseEditor: $('#exerciseEditor'),
        exerciseFocus: $('#exerciseFocus'),
        exerciseCustomFocus: $('#exerciseCustomFocus'),
        inspireExerciseBtn: $('#inspireExerciseBtn'),
        exercisePrompt: $('#exercisePrompt'),
        exerciseBody: $('#exerciseBody'),
        exerciseWordCount: $('#exerciseWordCount'),
        exerciseSaveState: $('#exerciseSaveState'),
        exerciseStatusBadge: $('#exerciseStatusBadge'),
        evaluateExerciseBtn: $('#evaluateExerciseBtn'),
        completeExerciseBtn: $('#completeExerciseBtn'),
        deleteExerciseBtn: $('#deleteExerciseBtn'),
        exerciseAiPanel: $('#exerciseAiPanel'),
        newMaterialBtn: $('#newMaterialBtn'),
        materialTypeFilter: $('#materialTypeFilter'),
        materialSearch: $('#materialSearch'),
        materialList: $('#materialList'),
        materialEmpty: $('#materialEmpty'),
        materialEditor: $('#materialEditor'),
        materialType: $('#materialType'),
        materialCustomType: $('#materialCustomType'),
        materialTitle: $('#materialTitle'),
        materialTagEditor: $('#materialTagEditor'),
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
        readingCustomLens: $('#readingCustomLens'),
        readingTagEditor: $('#readingTagEditor'),
        readingExcerpt: $('#readingExcerpt'),
        readingNotes: $('#readingNotes'),
        readingSaveState: $('#readingSaveState'),
        deleteReadingBtn: $('#deleteReadingBtn'),
        manageExerciseCategoriesBtn: $('#manageExerciseCategoriesBtn'),
        manageMaterialCategoriesBtn: $('#manageMaterialCategoriesBtn'),
        manageReadingCategoriesBtn: $('#manageReadingCategoriesBtn'),
        taxonomyModal: $('#taxonomyModal'),
        taxonomyModalTitle: $('#taxonomyModalTitle'),
        closeTaxonomyModalBtn: $('#closeTaxonomyModalBtn'),
        taxonomyList: $('#taxonomyList'),
        appDialog: $('#appDialog'),
        appDialogKicker: $('#appDialogKicker'),
        appDialogTitle: $('#appDialogTitle'),
        appDialogMessage: $('#appDialogMessage'),
        appDialogCancelBtn: $('#appDialogCancelBtn'),
        appDialogConfirmBtn: $('#appDialogConfirmBtn'),
        weekLabel: $('#weekLabel'),
        statsTotalWords: $('#statsTotalWords'),
        statsDraftWords: $('#statsDraftWords'),
        statsExerciseWords: $('#statsExerciseWords'),
        statsCompleted: $('#statsCompleted'),
        statsStreak: $('#statsStreak'),
        dailyBars: $('#dailyBars'),
        prevWeekBtn: $('#prevWeekBtn'),
        nextWeekBtn: $('#nextWeekBtn'),
        todayWeekBtn: $('#todayWeekBtn'),
        weekPickerToggle: $('#weekPickerToggle'),
        weekPicker: $('#weekPicker'),
        weekContext: $('#weekContext'),
        weekHistoryList: $('#weekHistoryList'),
        weeklyReview: $('#weeklyReview'),
        weeklyReviewSaveState: $('#weeklyReviewSaveState'),
        aiSummaryBtn: $('#aiSummaryBtn'),
        aiSummary: $('#aiSummary'),
        weeklyAiPanel: $('#weeklyAiPanel')
    };

    const state = {
        auth: null,
        db: null,
        functions: null,
        user: null,
        view: localStorage.getItem('writerView') || 'drafts',
        drafts: [],
        exercises: [],
        materials: [],
        readings: [],
        stats: [],
        weeklyReviews: [],
        taxonomy: normalizeTaxonomy(),
        activeDraftId: localStorage.getItem('writerActiveDraft') || null,
        activeExerciseId: localStorage.getItem('writerActiveExercise') || null,
        activeMaterialId: localStorage.getItem('writerActiveMaterial') || null,
        activeReadingId: localStorage.getItem('writerActiveReading') || null,
        materialFilter: 'all',
        materialQuery: '',
        selectedWeekStartId: localStorage.getItem('writerSelectedWeekStart') || getWeekRange().weekId,
        weekPickerOpen: false,
        taxonomyManagerKind: null,
        dialogResolve: null,
        generatingExerciseId: null,
        generatingWeeklyId: null,
        generatedExerciseIds: new Set(),
        generatedWeeklyIds: new Set(),
        promptIndex: Number(localStorage.getItem('writerPromptIndex') || '0'),
        tagValues: {
            draft: [],
            material: [],
            reading: []
        },
        unsubs: [],
        timers: {},
        savedCounts: {
            drafts: new Map(),
            exercises: new Map()
        }
    };

    init();

    function init() {
        selectWeek(state.selectedWeekStartId, { render: false });
        renderTaxonomyControls();
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
            state.functions = typeof firebase.functions === 'function' ? firebase.functions() : null;
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
        setupTagEditor('draft', els.draftTagEditor, scheduleDraftSave);
        [els.draftTitle, els.draftBody].forEach((input) => {
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
        els.exerciseFocus.addEventListener('change', () => toggleCustomCategoryInput('exercise'));
        els.exerciseCustomFocus.addEventListener('input', scheduleExerciseSave);
        els.inspireExerciseBtn.addEventListener('click', inspireExercise);
        els.evaluateExerciseBtn.addEventListener('click', generateExerciseEvaluation);
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
        setupTagEditor('material', els.materialTagEditor, scheduleMaterialSave);
        [els.materialType, els.materialCustomType, els.materialTitle, els.materialContent].forEach((input) => {
            input.addEventListener('input', scheduleMaterialSave);
            input.addEventListener('change', scheduleMaterialSave);
        });
        els.materialType.addEventListener('change', () => toggleCustomCategoryInput('material'));
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
            els.readingCustomLens,
            els.readingExcerpt,
            els.readingNotes
        ].forEach((input) => {
            input.addEventListener('input', scheduleReadingSave);
            input.addEventListener('change', scheduleReadingSave);
        });
        setupTagEditor('reading', els.readingTagEditor, scheduleReadingSave);
        els.readingLens.addEventListener('change', () => toggleCustomCategoryInput('reading'));
        els.deleteReadingBtn.addEventListener('click', deleteActiveReading);

        els.prevWeekBtn.addEventListener('click', () => shiftSelectedWeek(-1));
        els.nextWeekBtn.addEventListener('click', () => shiftSelectedWeek(1));
        els.todayWeekBtn.addEventListener('click', () => selectWeek(getWeekRange().weekId));
        els.weekPickerToggle.addEventListener('click', (event) => {
            event.stopPropagation();
            state.weekPickerOpen = !state.weekPickerOpen;
            renderReview();
        });
        els.weekHistoryList.addEventListener('click', (event) => {
            const item = event.target.closest('[data-week-id]');
            if (!item) return;
            selectWeek(item.getAttribute('data-week-id'));
        });
        document.addEventListener('click', (event) => {
            if (!state.weekPickerOpen || event.target.closest('.week-nav')) return;
            state.weekPickerOpen = false;
            renderReview();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape' || !state.weekPickerOpen) return;
            state.weekPickerOpen = false;
            renderReview();
        });
        els.manageExerciseCategoriesBtn.addEventListener('click', () => openTaxonomyManager('exercise'));
        els.manageMaterialCategoriesBtn.addEventListener('click', () => openTaxonomyManager('material'));
        els.manageReadingCategoriesBtn.addEventListener('click', () => openTaxonomyManager('reading'));
        els.closeTaxonomyModalBtn.addEventListener('click', closeTaxonomyManager);
        els.taxonomyModal.addEventListener('click', (event) => {
            if (event.target === els.taxonomyModal) closeTaxonomyManager();
        });
        els.taxonomyList.addEventListener('click', async (event) => {
            const renameBtn = event.target.closest('[data-taxonomy-rename]');
            const deleteBtn = event.target.closest('[data-taxonomy-delete]');
            if (renameBtn) {
                await renameTaxonomyItem(renameBtn.getAttribute('data-taxonomy-rename'));
            }
            if (deleteBtn) {
                await deleteTaxonomyItem(deleteBtn.getAttribute('data-taxonomy-delete'));
            }
        });
        els.taxonomyList.addEventListener('keydown', async (event) => {
            if (event.key !== 'Enter') return;
            const input = event.target.closest('[data-taxonomy-input]');
            if (!input) return;
            event.preventDefault();
            await renameTaxonomyItem(input.getAttribute('data-taxonomy-input'));
        });
        els.appDialogConfirmBtn.addEventListener('click', () => closeAppDialog(true));
        els.appDialogCancelBtn.addEventListener('click', () => closeAppDialog(false));
        els.appDialog.addEventListener('click', (event) => {
            if (event.target === els.appDialog) closeAppDialog(false);
        });
        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape' || els.appDialog.classList.contains('hidden')) return;
            closeAppDialog(false);
        });
        els.weeklyReview.addEventListener('input', () => {
            scheduleWeeklyReviewSave();
            updateWeeklyAiButton(getCurrentWeeklyReview(), getWeeklyStats());
        });
        els.aiSummaryBtn.addEventListener('click', generateWeeklyInsight);
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
        const taxonomyUnsub = state.db.collection('users').doc(state.user.uid)
            .collection('writingTaxonomy').doc('config')
            .onSnapshot((doc) => {
                state.taxonomy = normalizeTaxonomy(doc.exists ? doc.data() : null);
                renderTaxonomyControls();
                renderCurrentView();
            }, (error) => {
                console.error('Failed to read writing taxonomy', error);
            });
        state.unsubs.push(taxonomyUnsub);

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

    async function selectWeek(weekId, options = {}) {
        if (state.timers.weeklyReview && state.user) {
            clearTimeout(state.timers.weeklyReview);
            state.timers.weeklyReview = null;
            try {
                await saveWeeklyReview();
            } catch (error) {
                console.error('Failed to save review before switching weeks', error);
            }
        }
        const currentWeekId = getWeekRange().weekId;
        const normalized = normalizeWeekStartId(weekId);
        state.selectedWeekStartId = compareWeekIds(normalized, currentWeekId) > 0 ? currentWeekId : normalized;
        state.weekPickerOpen = false;
        localStorage.setItem('writerSelectedWeekStart', state.selectedWeekStartId);
        if (options.render !== false) {
            renderReview(true);
        }
    }

    async function shiftSelectedWeek(offset) {
        const start = parseDateId(state.selectedWeekStartId);
        start.setDate(start.getDate() + (offset * 7));
        await selectWeek(getWeekRange(start).weekId);
    }

    function renderTaxonomyControls() {
        const exerciseValue = els.exerciseFocus.value;
        const materialValue = els.materialType.value;
        const materialFilterValue = state.materialFilter;
        const readingValue = els.readingLens.value;

        setSelectOptions(
            els.exerciseFocus,
            state.taxonomy.exerciseFocuses.map((focus) => ({ value: focus, label: focus })),
            exerciseValue,
            { custom: true, blankLabel: '未分类' }
        );
        setSelectOptions(
            els.materialType,
            getVisibleMaterialTypes().map((type) => ({ value: type.id, label: type.label })),
            materialValue,
            { custom: true }
        );
        setSelectOptions(
            els.materialTypeFilter,
            [
                { value: 'all', label: '全部类型' },
                ...getVisibleMaterialTypes().map((type) => ({ value: type.id, label: type.label }))
            ],
            materialFilterValue,
            { custom: false }
        );
        setSelectOptions(
            els.readingLens,
            state.taxonomy.readingLenses.map((lens) => ({ value: lens, label: lens })),
            readingValue,
            { custom: true }
        );

        if (state.materialFilter !== 'all' && !getVisibleMaterialTypes().some((type) => type.id === state.materialFilter)) {
            state.materialFilter = 'all';
            els.materialTypeFilter.value = 'all';
        }
    }

    function setSelectOptions(select, options, preferredValue, config = {}) {
        const allOptions = config.blankLabel ? [{ value: '', label: config.blankLabel }, ...options] : options;
        const validPreferred = allOptions.some((option) => option.value === preferredValue);
        const nextValue = validPreferred ? preferredValue : allOptions[0]?.value || '';
        select.innerHTML = allOptions.map((option) => (
            `<option value="${escapeAttr(option.value)}">${escapeHtml(option.label)}</option>`
        )).join('');
        if (config.custom) {
            select.insertAdjacentHTML('beforeend', `<option value="${CUSTOM_OPTION}">自定义...</option>`);
        }
        select.value = nextValue;
    }

    function openTaxonomyManager(kind) {
        state.taxonomyManagerKind = kind;
        renderTaxonomyManager();
        els.taxonomyModal.classList.remove('hidden');
    }

    function closeTaxonomyManager() {
        state.taxonomyManagerKind = null;
        els.taxonomyModal.classList.add('hidden');
    }

    function showAppDialog(options = {}) {
        if (state.dialogResolve) {
            closeAppDialog(false);
        }

        const {
            kicker = 'Notice',
            title = '提示',
            message = '',
            confirmText = '确定',
            cancelText = '',
            danger = false
        } = options;

        els.appDialogKicker.textContent = kicker;
        els.appDialogTitle.textContent = title;
        els.appDialogMessage.textContent = message;
        els.appDialogConfirmBtn.textContent = confirmText;
        els.appDialogCancelBtn.textContent = cancelText || '取消';
        els.appDialogCancelBtn.classList.toggle('hidden', !cancelText);
        els.appDialog.classList.toggle('danger', Boolean(danger));
        els.appDialog.classList.remove('hidden');

        setTimeout(() => els.appDialogConfirmBtn.focus(), 0);

        return new Promise((resolve) => {
            state.dialogResolve = resolve;
        });
    }

    function closeAppDialog(result) {
        if (!state.dialogResolve) return;
        const resolve = state.dialogResolve;
        state.dialogResolve = null;
        els.appDialog.classList.add('hidden');
        els.appDialog.classList.remove('danger');
        resolve(Boolean(result));
    }

    function showNotice(title, message) {
        return showAppDialog({
            kicker: 'Notice',
            title,
            message,
            confirmText: '知道了'
        });
    }

    function confirmDanger(title, message, confirmText = '删除') {
        return showAppDialog({
            kicker: 'Confirm',
            title,
            message,
            confirmText,
            cancelText: '取消',
            danger: true
        });
    }

    function renderTaxonomyManager() {
        const config = getTaxonomyManagerConfig(state.taxonomyManagerKind);
        if (!config) return;
        els.taxonomyModalTitle.textContent = config.title;
        const rows = config.items.map((item) => {
            if (item.builtin) {
                return `
                    <div class="taxonomy-row locked">
                        <span class="taxonomy-row-label">${escapeHtml(item.label)}</span>
                        <span class="taxonomy-row-badge">预设</span>
                    </div>
                `;
            }
            return `
                <div class="taxonomy-row">
                    <input data-taxonomy-input="${escapeAttr(item.id)}" value="${escapeAttr(item.label)}" autocomplete="off">
                    <button class="secondary-btn compact" data-taxonomy-rename="${escapeAttr(item.id)}" type="button">改名</button>
                    <button class="danger-btn" data-taxonomy-delete="${escapeAttr(item.id)}" type="button">删除</button>
                </div>
            `;
        }).join('');
        els.taxonomyList.innerHTML = rows || '<div class="taxonomy-empty">暂无分类</div>';
    }

    function getTaxonomyManagerConfig(kind) {
        if (kind === 'exercise') {
            return {
                title: '管理训练重点',
                field: 'exerciseFocuses',
                collection: 'writingExercises',
                docField: 'focus',
                defaultLabels: DEFAULT_EXERCISE_FOCUSES,
                items: state.taxonomy.exerciseFocuses.map((label) => ({
                    id: label,
                    label,
                    builtin: DEFAULT_EXERCISE_FOCUSES.includes(label)
                }))
            };
        }
        if (kind === 'material') {
            return {
                title: '管理素材类型',
                field: 'materialTypes',
                items: getVisibleMaterialTypes().map((type) => ({
                    id: type.id,
                    label: type.label,
                    builtin: Boolean(type.builtin),
                    hidden: Boolean(type.hidden)
                }))
            };
        }
        if (kind === 'reading') {
            return {
                title: '管理拆解视角',
                field: 'readingLenses',
                collection: 'readingBreakdowns',
                docField: 'lens',
                defaultLabels: DEFAULT_READING_LENSES,
                items: state.taxonomy.readingLenses.map((label) => ({
                    id: label,
                    label,
                    builtin: DEFAULT_READING_LENSES.includes(label)
                }))
            };
        }
        return null;
    }

    async function renameTaxonomyItem(id) {
        const config = getTaxonomyManagerConfig(state.taxonomyManagerKind);
        if (!config) return;
        const input = Array.from(els.taxonomyList.querySelectorAll('[data-taxonomy-input]'))
            .find((node) => node.getAttribute('data-taxonomy-input') === id);
        const nextLabel = input?.value.trim();
        if (!nextLabel) return;

        if (state.taxonomyManagerKind === 'material') {
            const item = state.taxonomy.materialTypes.find((type) => type.id === id);
            if (!item || item.builtin) return;
            const exists = state.taxonomy.materialTypes.some((type) => type.id !== id && !type.hidden && type.label.toLowerCase() === nextLabel.toLowerCase());
            if (exists) return showNotice('分类已存在', '这个分类已经存在，请换一个名字。');
            item.label = nextLabel;
            item.hidden = false;
        } else {
            if (config.defaultLabels.includes(id)) return;
            const exists = config.defaultLabels.includes(nextLabel) || state.taxonomy[config.field].some((label) => label !== id && label.toLowerCase() === nextLabel.toLowerCase());
            if (exists) return showNotice('分类已存在', '这个分类已经存在，请换一个名字。');
            state.taxonomy[config.field] = state.taxonomy[config.field].map((label) => label === id ? nextLabel : label);
            await updateTextCategoryReferences(config.collection, config.docField, id, nextLabel);
        }

        await persistTaxonomy();
        renderTaxonomyControls();
        renderTaxonomyManager();
        renderCurrentView(true);
    }

    async function deleteTaxonomyItem(id) {
        const config = getTaxonomyManagerConfig(state.taxonomyManagerKind);
        if (!config) return;
        const confirmed = await confirmDanger(
            '移除自定义分类',
            '从下拉中移除这个自定义分类？已有记录会保留原分类。',
            '移除'
        );
        if (!confirmed) return;

        if (state.taxonomyManagerKind === 'material') {
            const item = state.taxonomy.materialTypes.find((type) => type.id === id);
            if (!item || item.builtin) return;
            item.hidden = true;
        } else {
            if (config.defaultLabels.includes(id)) return;
            state.taxonomy[config.field] = state.taxonomy[config.field].filter((label) => label !== id);
        }

        await persistTaxonomy();
        renderTaxonomyControls();
        renderTaxonomyManager();
        renderCurrentView(true);
    }

    async function updateTextCategoryReferences(collectionName, field, oldValue, newValue) {
        if (!state.user || oldValue === newValue) return;
        const localItems = collectionName === 'writingExercises' ? state.exercises : state.readings;
        const matches = localItems.filter((item) => item[field] === oldValue);
        if (!matches.length) return;
        const batch = state.db.batch();
        matches.slice(0, 450).forEach((item) => {
            batch.update(getUserRef(collectionName).doc(item.id), {
                [field]: newValue,
                updatedAt: serverTimestamp()
            });
        });
        await batch.commit();
    }

    function renderShell() {
        const currentWeekStats = getWeeklyStats(getWeekRange());
        els.draftCount.textContent = state.drafts.length;
        els.exerciseCount.textContent = state.exercises.length;
        els.materialCount.textContent = state.materials.length;
        els.readingCount.textContent = state.readings.length;
        els.topWeekWords.textContent = `${formatNumber(currentWeekStats.totalWords)} 字 / 本周`;
    }

    function renderCurrentView(forcePopulate = false) {
        if (state.view === 'drafts') renderDrafts(forcePopulate);
        if (state.view === 'exercises') renderExercises(forcePopulate);
        if (state.view === 'materials') renderMaterials(forcePopulate);
        if (state.view === 'reading') renderReading(forcePopulate);
        if (state.view === 'review') renderReview(forcePopulate);
    }

    function setupTagEditor(key, editor, onChange) {
        editor.addEventListener('click', (event) => {
            const removeBtn = event.target.closest('[data-remove-tag]');
            if (removeBtn) {
                const tag = removeBtn.getAttribute('data-remove-tag');
                state.tagValues[key] = state.tagValues[key].filter((item) => item !== tag);
                renderTagEditor(key);
                onChange();
                return;
            }
            const input = editor.querySelector('.tag-chip-input');
            if (input) input.focus();
        });

        editor.addEventListener('keydown', (event) => {
            if (!event.target.classList.contains('tag-chip-input')) return;
            if (event.key === 'Enter' || event.key === ',' || event.key === '，') {
                event.preventDefault();
                commitTagInput(key, onChange);
            }
            if (event.key === 'Backspace' && !event.target.value && state.tagValues[key].length) {
                state.tagValues[key].pop();
                renderTagEditor(key);
                onChange();
            }
        });

        editor.addEventListener('paste', (event) => {
            if (!event.target.classList.contains('tag-chip-input')) return;
            const text = event.clipboardData?.getData('text') || '';
            if (!/[，,\n]/.test(text)) return;
            event.preventDefault();
            addTags(key, parseTags(text));
            renderTagEditor(key);
            onChange();
        });

        editor.addEventListener('focusout', (event) => {
            if (!event.target.classList.contains('tag-chip-input')) return;
            commitTagInput(key, onChange, { keepFocus: false });
        });

        renderTagEditor(key);
    }

    function renderTagEditor(key) {
        const editor = getTagEditor(key);
        if (!editor) return;
        const placeholder = editor.getAttribute('data-placeholder') || '标签';
        const chips = getTagValues(key).map((tag) => `
            <span class="tag-chip">
                ${escapeHtml(tag)}
                <button type="button" data-remove-tag="${escapeAttr(tag)}" aria-label="删除 ${escapeAttr(tag)}">x</button>
            </span>
        `).join('');

        editor.innerHTML = `
            <div class="tag-chip-row">
                ${chips}
                <input class="tag-chip-input" type="text" autocomplete="off" placeholder="${escapeAttr(placeholder)}">
            </div>
        `;
    }

    function commitTagInput(key, onChange, options = {}) {
        const editor = getTagEditor(key);
        const input = editor?.querySelector('.tag-chip-input');
        if (!input || !input.value.trim()) return;
        const changed = addTags(key, parseTags(input.value));
        input.value = '';
        if (!changed) return;
        renderTagEditor(key);
        if (options.keepFocus !== false) {
            getTagEditor(key)?.querySelector('.tag-chip-input')?.focus();
        }
        onChange();
    }

    function addTags(key, tags) {
        const before = getTagValues(key).join('\u0000');
        const existing = new Set(getTagValues(key).map((tag) => tag.toLowerCase()));
        tags.forEach((tag) => {
            const normalized = tag.trim();
            if (!normalized || existing.has(normalized.toLowerCase())) return;
            if (state.tagValues[key].length >= 12) return;
            state.tagValues[key].push(normalized);
            existing.add(normalized.toLowerCase());
        });
        return before !== getTagValues(key).join('\u0000');
    }

    function setTagValues(key, tags) {
        state.tagValues[key] = normalizeTags(tags);
        renderTagEditor(key);
    }

    function getTagValues(key) {
        return normalizeTags(state.tagValues[key] || []);
    }

    function getTagEditor(key) {
        return {
            draft: els.draftTagEditor,
            material: els.materialTagEditor,
            reading: els.readingTagEditor
        }[key];
    }

    function setTextCategoryValue(kind, value) {
        const config = getTextCategoryConfig(kind);
        if (!config) return;
        if (!value) {
            config.select.value = '';
            config.input.value = '';
        } else if (config.options.includes(value)) {
            config.select.value = value;
            config.input.value = '';
        } else {
            config.select.value = CUSTOM_OPTION;
            config.input.value = value || '';
        }
        toggleCustomCategoryInput(kind);
    }

    function getPendingExerciseFocus() {
        if (els.exerciseFocus.value === CUSTOM_OPTION) {
            return els.exerciseCustomFocus.value.trim();
        }
        return els.exerciseFocus.value.trim();
    }

    function setMaterialTypeValue(typeId) {
        if (getVisibleMaterialTypes().some((type) => type.id === typeId)) {
            els.materialType.value = typeId;
            els.materialCustomType.value = '';
        } else {
            els.materialType.value = CUSTOM_OPTION;
            els.materialCustomType.value = getMaterialLabel(typeId);
        }
        toggleCustomCategoryInput('material');
    }

    function toggleCustomCategoryInput(kind) {
        const config = {
            exercise: { select: els.exerciseFocus, input: els.exerciseCustomFocus },
            material: { select: els.materialType, input: els.materialCustomType },
            reading: { select: els.readingLens, input: els.readingCustomLens }
        }[kind];
        if (!config) return;
        const isCustom = config.select.value === CUSTOM_OPTION;
        config.input.classList.toggle('hidden', !isCustom);
        if (isCustom) config.input.focus();
    }

    async function resolveTextCategory(field, select, input, fallback) {
        if (select.value !== CUSTOM_OPTION) return select.value || fallback;
        const label = input.value.trim();
        if (!label) return fallback;
        await addTextTaxonomy(field, label);
        select.value = label;
        input.value = '';
        input.classList.add('hidden');
        return label;
    }

    async function resolveMaterialType(fallback) {
        if (els.materialType.value !== CUSTOM_OPTION) return els.materialType.value || fallback;
        const label = els.materialCustomType.value.trim();
        if (!label) return fallback;
        const id = await addMaterialType(label);
        els.materialType.value = id;
        els.materialCustomType.value = '';
        els.materialCustomType.classList.add('hidden');
        return id;
    }

    async function addTextTaxonomy(field, label) {
        const clean = label.trim();
        if (!clean) return;
        const values = uniqueTextValues([...(state.taxonomy[field] || []), clean]);
        state.taxonomy[field] = values;
        renderTaxonomyControls();
        await persistTaxonomy();
    }

    async function addMaterialType(label) {
        const clean = label.trim();
        const existing = state.taxonomy.materialTypes.find((type) => type.label.toLowerCase() === clean.toLowerCase());
        if (existing) {
            existing.hidden = false;
            await persistTaxonomy();
            renderTaxonomyControls();
            return existing.id;
        }
        const id = makeCustomTypeId(clean);
        state.taxonomy.materialTypes = [
            ...state.taxonomy.materialTypes,
            { id, label: clean, builtin: false }
        ];
        renderTaxonomyControls();
        await persistTaxonomy();
        return id;
    }

    async function persistTaxonomy() {
        if (!state.user) return;
        await state.db.collection('users').doc(state.user.uid)
            .collection('writingTaxonomy').doc('config')
            .set({
                exerciseFocuses: state.taxonomy.exerciseFocuses,
                materialTypes: state.taxonomy.materialTypes,
                readingLenses: state.taxonomy.readingLenses,
                updatedAt: serverTimestamp()
            }, { merge: true });
    }

    function renderDrafts(forcePopulate = false) {
        renderList(els.draftList, state.drafts, state.activeDraftId, 'draft');
        const draft = getActiveDraft();
        toggleEditor(els.draftEmpty, els.draftEditor, Boolean(draft));
        if (!draft) return;
        if (forcePopulate || !isFocusedWithin([els.draftTitle, els.draftBody]) && !els.draftTagEditor.contains(document.activeElement)) {
            els.draftTitle.value = draft.title || '';
            setTagValues('draft', draft.tags);
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
            tags: getTagValues('draft'),
            wordCount,
            updatedAt: serverTimestamp()
        }, { merge: true });
        setSavedCount('drafts', draft.id, wordCount);
        await recordWritingDelta('draftWords', wordCount - previousCount);
        els.draftSaveState.textContent = '已保存';
    }

    async function deleteActiveDraft() {
        const draft = getActiveDraft();
        if (!draft) return;
        const confirmed = await confirmDanger('删除草稿', '删除这篇草稿？这个操作不能撤销。');
        if (!confirmed) return;
        setActiveId('activeDraftId', 'writerActiveDraft', getNextItemId(state.drafts, draft.id));
        await getUserRef('writingDrafts').doc(draft.id).delete();
    }

    function renderExercises(forcePopulate = false) {
        renderList(els.exerciseList, state.exercises, state.activeExerciseId, 'exercise');
        const exercise = getActiveExercise();
        toggleEditor(els.exerciseEmpty, els.exerciseEditor, Boolean(exercise));
        if (!exercise) return;
        if (forcePopulate || !isFocusedWithin([els.exerciseFocus, els.exerciseCustomFocus, els.exercisePrompt, els.exerciseBody])) {
            setTextCategoryValue('exercise', exercise.focus || '');
            els.exercisePrompt.value = exercise.prompt || '';
            els.exerciseBody.value = exercise.body || '';
            els.exerciseSaveState.textContent = '已保存';
        }
        const done = exercise.status === 'done';
        els.exerciseStatusBadge.textContent = done ? '已完成' : '草稿';
        els.exerciseStatusBadge.classList.toggle('done', done);
        els.completeExerciseBtn.textContent = done ? '重新打开' : '标记完成';
        els.exerciseWordCount.textContent = `${formatNumber(countWords(els.exerciseBody.value))} 字`;
        renderExerciseAi(exercise.aiEvaluation);
        updateExerciseAiButton(exercise);
    }

    async function createExercise() {
        const ref = getUserRef('writingExercises').doc();
        await ref.set({
            prompt: '',
            focus: '',
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

    async function inspireExercise() {
        const exercise = getActiveExercise();
        if (!exercise) return;
        const prompt = PROMPT_BANK[state.promptIndex % PROMPT_BANK.length];
        state.promptIndex += 1;
        localStorage.setItem('writerPromptIndex', String(state.promptIndex));
        await addTextTaxonomy('exerciseFocuses', prompt.focus);
        setTextCategoryValue('exercise', prompt.focus);
        els.exercisePrompt.value = prompt.prompt;
        await saveExercise();
    }

    function scheduleExerciseSave() {
        const wordCount = countWords(els.exerciseBody.value);
        els.exerciseWordCount.textContent = `${formatNumber(wordCount)} 字`;
        updateExerciseAiButton(getActiveExercise(), wordCount);
        debounce('exercise', saveExercise, els.exerciseSaveState);
    }

    async function saveExercise(options = {}) {
        const exercise = getActiveExercise();
        if (!exercise) return;
        const wordCount = countWords(els.exerciseBody.value);
        const previousCount = getSavedCount('exercises', exercise);
        const focus = await resolveTextCategory('exerciseFocuses', els.exerciseFocus, els.exerciseCustomFocus, exercise.focus || '');
        const data = {
            prompt: els.exercisePrompt.value.trim(),
            focus,
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
        if (nextStatus === 'done' && !getPendingExerciseFocus()) {
            await showNotice('需要分类', '请先选择训练重点，或用“自定义...”填写一个分类，再标记完成。');
            els.exerciseFocus.focus();
            return;
        }
        await saveExercise({ status: nextStatus });
    }

    async function deleteActiveExercise() {
        const exercise = getActiveExercise();
        if (!exercise) return;
        const confirmed = await confirmDanger('删除练习', '删除这条场景练习？这个操作不能撤销。');
        if (!confirmed) return;
        setActiveId('activeExerciseId', 'writerActiveExercise', getNextItemId(state.exercises, exercise.id));
        await getUserRef('writingExercises').doc(exercise.id).delete();
    }

    function updateExerciseAiButton(exercise, wordCount = countWords(els.exerciseBody.value)) {
        const generating = Boolean(exercise && state.generatingExerciseId === exercise.id);
        const hasEvaluation = Boolean(exercise?.aiEvaluation || (exercise && state.generatedExerciseIds.has(exercise.id)));
        els.evaluateExerciseBtn.disabled = !exercise || !state.functions || hasEvaluation || generating || wordCount < 30;
        if (generating) {
            els.evaluateExerciseBtn.textContent = '生成中';
        } else if (hasEvaluation) {
            els.evaluateExerciseBtn.textContent = '已评估';
        } else {
            els.evaluateExerciseBtn.textContent = 'AI评估';
        }
    }

    function renderExerciseAi(evaluation) {
        if (!evaluation) {
            els.exerciseAiPanel.classList.add('hidden');
            els.exerciseAiPanel.innerHTML = '';
            return;
        }

        const score = evaluation.score || {};
        const scores = [
            ['欲望', score.characterDesire],
            ['冲突', score.conflictClarity],
            ['推进', score.sceneProgression],
            ['文字', score.proseControl],
            ['整体', score.overall]
        ].map(([label, value]) => (
            `<span class="ai-score-chip">${escapeHtml(label)} ${escapeHtml(formatScore(value))}</span>`
        )).join('');

        els.exerciseAiPanel.classList.remove('hidden');
        els.exerciseAiPanel.innerHTML = `
            <div class="ai-result-head">
                <span>AI评估</span>
                <small>${escapeHtml(evaluation.model || 'AI')}</small>
            </div>
            <p class="ai-result-summary">${escapeHtml(evaluation.summary || '')}</p>
            <div class="ai-score-row">${scores}</div>
            ${renderAiTextList('亮点', evaluation.strengths)}
            ${renderAiTextList('卡点', evaluation.weaknesses)}
            ${renderAiEvidence(evaluation.evidence)}
            ${renderAiTargets(evaluation.revisionTargets)}
            ${evaluation.nextPrompt ? `<div class="ai-next-prompt"><span>下一题</span><p>${escapeHtml(evaluation.nextPrompt)}</p></div>` : ''}
        `;
    }

    async function generateExerciseEvaluation() {
        const exercise = getActiveExercise();
        if (!exercise) return;
        if (exercise.aiEvaluation) {
            await showNotice('已经评估过', '每条练习第一版只保留一次 AI 评估。');
            return;
        }
        const wordCount = countWords(els.exerciseBody.value);
        if (wordCount < 30) {
            await showNotice('正文太短', '先写到 30 字以上，再让 AI 做训练反馈。');
            return;
        }
        if (!state.functions) {
            await showNotice('AI 暂不可用', 'Firebase Functions SDK 没有加载成功。');
            return;
        }

        try {
            await saveExercise();
            state.generatingExerciseId = exercise.id;
            updateExerciseAiButton(exercise, wordCount);
            const generate = state.functions.httpsCallable('generateWritingExerciseEvaluation');
            const result = await generate({ exerciseId: exercise.id });
            if (result.data?.aiEvaluation) {
                state.generatedExerciseIds.add(exercise.id);
                renderExerciseAi(result.data.aiEvaluation);
            }
            await showNotice('AI评估已生成', '结果已经保存到这条练习里。');
        } catch (error) {
            console.error('Failed to generate exercise evaluation', error);
            await showNotice('AI评估失败', normalizeFunctionError(error));
        } finally {
            state.generatingExerciseId = null;
            updateExerciseAiButton(getActiveExercise());
        }
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
        if (forcePopulate || !isFocusedWithin([els.materialType, els.materialCustomType, els.materialTitle, els.materialContent]) && !els.materialTagEditor.contains(document.activeElement)) {
            setMaterialTypeValue(material.type || 'idea');
            els.materialTitle.value = material.title || '';
            setTagValues('material', material.tags);
            els.materialContent.value = material.content || '';
            els.materialSaveState.textContent = '已保存';
        }
    }

    async function createMaterial() {
        const type = state.materialFilter === 'all' ? 'idea' : state.materialFilter;
        const ref = getUserRef('writingMaterials').doc();
        await ref.set({
            type,
            title: `${getMaterialLabel(type)}记录`,
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
        const type = await resolveMaterialType(material.type || 'idea');
        await getUserRef('writingMaterials').doc(material.id).set({
            type,
            title: els.materialTitle.value.trim() || '未命名素材',
            content: els.materialContent.value,
            tags: getTagValues('material'),
            updatedAt: serverTimestamp()
        }, { merge: true });
        els.materialSaveState.textContent = '已保存';
    }

    async function deleteActiveMaterial() {
        const material = getActiveMaterial();
        if (!material) return;
        const confirmed = await confirmDanger('删除素材', '删除这条素材？这个操作不能撤销。');
        if (!confirmed) return;
        setActiveId('activeMaterialId', 'writerActiveMaterial', getNextItemId(getFilteredMaterials(), material.id));
        await getUserRef('writingMaterials').doc(material.id).delete();
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
            els.readingCustomLens,
            els.readingExcerpt,
            els.readingNotes
        ]) && !els.readingTagEditor.contains(document.activeElement)) {
            els.readingSourceTitle.value = reading.sourceTitle || '';
            els.readingAuthor.value = reading.author || '';
            setTextCategoryValue('reading', reading.lens || DEFAULT_READING_LENSES[0]);
            setTagValues('reading', reading.tags);
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
        const lens = await resolveTextCategory('readingLenses', els.readingLens, els.readingCustomLens, reading.lens || DEFAULT_READING_LENSES[0]);
        await getUserRef('readingBreakdowns').doc(reading.id).set({
            sourceTitle: els.readingSourceTitle.value.trim() || '未命名作品',
            author: els.readingAuthor.value.trim(),
            excerpt: els.readingExcerpt.value,
            lens,
            notes: els.readingNotes.value,
            tags: getTagValues('reading'),
            updatedAt: serverTimestamp()
        }, { merge: true });
        els.readingSaveState.textContent = '已保存';
    }

    async function deleteActiveReading() {
        const reading = getActiveReading();
        if (!reading) return;
        const confirmed = await confirmDanger('删除拆解', '删除这条阅读拆解？这个操作不能撤销。');
        if (!confirmed) return;
        setActiveId('activeReadingId', 'writerActiveReading', getNextItemId(state.readings, reading.id));
        await getUserRef('readingBreakdowns').doc(reading.id).delete();
    }

    function renderReview(forcePopulate = false) {
        const range = getSelectedWeekRange();
        const stats = getWeeklyStats(range);
        const review = getCurrentWeeklyReview(range.weekId);
        const currentWeekId = getWeekRange().weekId;
        const isCurrentWeek = range.weekId === currentWeekId;

        els.weekLabel.textContent = `${range.startLabel} - ${range.endLabel}`;
        els.weekContext.textContent = isCurrentWeek ? '本周' : '历史';
        els.statsTotalWords.textContent = formatNumber(stats.totalWords);
        els.statsDraftWords.textContent = formatNumber(stats.draftWords);
        els.statsExerciseWords.textContent = formatNumber(stats.exerciseWords);
        els.statsCompleted.textContent = formatNumber(stats.completedExercises);
        els.statsStreak.textContent = formatNumber(stats.streak);
        els.nextWeekBtn.disabled = isCurrentWeek;
        els.todayWeekBtn.disabled = isCurrentWeek;
        els.weekPickerToggle.setAttribute('aria-expanded', String(state.weekPickerOpen));
        els.weekPicker.classList.toggle('hidden', !state.weekPickerOpen);
        renderWeekHistory(range.weekId);
        renderDailyBars(stats.days);
        renderWeeklyAi(review?.aiInsight || null);

        if (forcePopulate || !isFocusedWithin([els.weeklyReview])) {
            els.weeklyReview.value = review?.manualReview || '';
            els.aiSummary.value = review?.aiSummary || review?.aiInsight?.summaryText || '';
            els.weeklyReviewSaveState.textContent = '已保存';
        }
        updateWeeklyAiButton(review, stats);
    }

    function scheduleWeeklyReviewSave() {
        debounce('weeklyReview', saveWeeklyReview, els.weeklyReviewSaveState);
    }

    async function saveWeeklyReview() {
        const range = getSelectedWeekRange();
        const stats = getWeeklyStats(range);
        const current = getCurrentWeeklyReview(range.weekId);
        await getUserRef('writingWeeklyReviews').doc(range.weekId).set({
            weekStart: range.weekStartDate,
            weekEnd: range.weekEndDate,
            manualReview: els.weeklyReview.value,
            aiSummary: current?.aiSummary || current?.aiInsight?.summaryText || '',
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

    function updateWeeklyAiButton(review, stats = getWeeklyStats()) {
        const range = getSelectedWeekRange();
        const currentWeekId = getWeekRange().weekId;
        const isCurrentOrFutureWeek = compareWeekIds(range.weekId, currentWeekId) >= 0;
        const generating = state.generatingWeeklyId === range.weekId;
        const hasInsight = Boolean(review?.aiInsight || state.generatedWeeklyIds.has(range.weekId));
        const hasContent = Number(stats.totalWords || 0) > 0 || Boolean((review?.manualReview || els.weeklyReview.value || '').trim());
        els.aiSummaryBtn.disabled = !state.functions || isCurrentOrFutureWeek || hasInsight || generating || !hasContent;
        if (generating) {
            els.aiSummaryBtn.textContent = '生成中';
        } else if (hasInsight) {
            els.aiSummaryBtn.textContent = '已总结';
        } else if (isCurrentOrFutureWeek) {
            els.aiSummaryBtn.textContent = '本周结束后可用';
        } else {
            els.aiSummaryBtn.textContent = 'AI 周总结';
        }
    }

    function renderWeeklyAi(insight) {
        if (!insight) {
            els.weeklyAiPanel.classList.add('hidden');
            els.weeklyAiPanel.innerHTML = '';
            return;
        }

        els.weeklyAiPanel.classList.remove('hidden');
        els.weeklyAiPanel.innerHTML = `
            <div class="ai-result-head">
                <span>${escapeHtml(insight.headline || 'AI周总结')}</span>
                <small>${escapeHtml(insight.model || 'AI')}</small>
            </div>
            <p class="ai-result-summary">${escapeHtml(insight.summaryText || '')}</p>
            <div class="ai-grid-notes">
                ${renderAiNote('数据', insight.statsRead)}
                ${renderAiNote('对比', insight.comparedToLastWeek)}
                ${renderAiNote('节奏', insight.rhythm)}
                ${renderAiNote('进步', insight.strongestProgress)}
                ${renderAiNote('卡点', insight.mainBlocker)}
            </div>
            ${renderAiEvidence(insight.evidence)}
            ${renderAiPlan(insight.nextWeekPlan)}
        `;
    }

    async function generateWeeklyInsight() {
        const range = getSelectedWeekRange();
        const review = getCurrentWeeklyReview(range.weekId);
        const stats = getWeeklyStats(range);
        if (compareWeekIds(range.weekId, getWeekRange().weekId) >= 0) {
            await showNotice('本周还没结束', '进入下一周后，再回到这一周生成正式 AI 周总结。');
            return;
        }
        if (review?.aiInsight) {
            await showNotice('已经总结过', '每周第一版只保留一次 AI 周总结。');
            return;
        }
        if (!state.functions) {
            await showNotice('AI 暂不可用', 'Firebase Functions SDK 没有加载成功。');
            return;
        }
        if (!stats.totalWords && !els.weeklyReview.value.trim()) {
            await showNotice('暂无内容', '这一周还没有可总结的写作内容。');
            return;
        }

        try {
            await saveWeeklyReview();
            state.generatingWeeklyId = range.weekId;
            updateWeeklyAiButton(review, stats);
            const generate = state.functions.httpsCallable('generateWritingWeeklyInsight');
            const result = await generate({ weekId: range.weekId });
            const insight = result.data?.aiInsight || null;
            if (insight) {
                state.generatedWeeklyIds.add(range.weekId);
                els.aiSummary.value = result.data?.aiSummary || insight.summaryText || '';
                renderWeeklyAi(insight);
            }
            await showNotice('AI周总结已生成', '结果已经保存到这一周的复盘里。');
        } catch (error) {
            console.error('Failed to generate weekly insight', error);
            await showNotice('AI周总结失败', normalizeFunctionError(error));
        } finally {
            state.generatingWeeklyId = null;
            updateWeeklyAiButton(getCurrentWeeklyReview(range.weekId), getWeeklyStats(range));
        }
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

    function renderAiTextList(title, values) {
        const items = Array.isArray(values) ? values.filter(Boolean) : [];
        if (!items.length) return '';
        return `
            <div class="ai-result-section">
                <h4>${escapeHtml(title)}</h4>
                <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
            </div>
        `;
    }

    function renderAiEvidence(values) {
        const items = Array.isArray(values) ? values.filter(Boolean) : [];
        if (!items.length) return '';
        return `
            <div class="ai-result-section">
                <h4>证据</h4>
                <ul>${items.map((item) => {
                    const quote = item.quote || item.source || '';
                    const point = item.point || '';
                    return `<li>${quote ? `<em>${escapeHtml(quote)}</em>` : ''}${point ? `<span>${escapeHtml(point)}</span>` : ''}</li>`;
                }).join('')}</ul>
            </div>
        `;
    }

    function renderAiTargets(values) {
        const items = Array.isArray(values) ? values.filter(Boolean) : [];
        if (!items.length) return '';
        return `
            <div class="ai-result-section">
                <h4>下次练</h4>
                <ul>${items.map((item) => `
                    <li>
                        <strong>${escapeHtml(item.target || '')}</strong>
                        ${item.why ? `<span>${escapeHtml(item.why)}</span>` : ''}
                        ${item.exercise ? `<small>${escapeHtml(item.exercise)}</small>` : ''}
                    </li>
                `).join('')}</ul>
            </div>
        `;
    }

    function renderAiPlan(values) {
        const items = Array.isArray(values) ? values.filter(Boolean) : [];
        if (!items.length) return '';
        return `
            <div class="ai-result-section">
                <h4>下周计划</h4>
                <ul>${items.map((item) => `
                    <li>
                        <strong>${escapeHtml(item.focus || '')}</strong>
                        <span>${escapeHtml(item.task || '')}</span>
                    </li>
                `).join('')}</ul>
            </div>
        `;
    }

    function renderAiNote(title, value) {
        if (!value) return '';
        return `
            <div class="ai-note">
                <span>${escapeHtml(title)}</span>
                <p>${escapeHtml(value)}</p>
            </div>
        `;
    }

    function renderWeekHistory(activeWeekId) {
        const weeks = getHistoryWeeks();
        let currentMonth = '';
        const html = weeks.map((week) => {
            const monthLabel = getWeekMonthLabel(week.weekId);
            const monthHead = monthLabel === currentMonth ? '' : `<div class="week-month-label">${escapeHtml(monthLabel)}</div>`;
            currentMonth = monthLabel;
            const meta = getWeekHistoryMeta(week);
            return `
                ${monthHead}
                <button class="week-history-item ${week.weekId === activeWeekId ? 'active' : ''}" data-week-id="${escapeAttr(week.weekId)}" type="button">
                    <span>${escapeHtml(week.startLabel)} - ${escapeHtml(week.endLabel)}</span>
                    <small>${escapeHtml(meta)}</small>
                </button>
            `;
        }).join('');

        els.weekHistoryList.innerHTML = html || '<div class="week-empty">暂无历史</div>';
    }

    function getWeeklyStats(range = getSelectedWeekRange()) {
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
            streak: getWritingStreak(range.weekEnd > new Date() ? new Date() : range.weekEnd)
        };
    }

    function getWritingStreak(anchorDate = new Date()) {
        const writtenDays = new Set(
            state.stats
                .filter((entry) => Number(entry.totalWords || 0) > 0)
                .map((entry) => entry.date)
        );
        let streak = 0;
        const cursor = new Date(anchorDate.getFullYear(), anchorDate.getMonth(), anchorDate.getDate());
        while (writtenDays.has(getDateParts(cursor).id)) {
            streak += 1;
            cursor.setDate(cursor.getDate() - 1);
        }
        return streak;
    }

    function getCurrentWeeklyReview(weekId = getSelectedWeekRange().weekId) {
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
                getMaterialLabel(material.type)
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
        if (type === 'material') return `${getMaterialLabel(item.type)} / ${date}`;
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

    function getNextItemId(items, currentId) {
        const index = items.findIndex((item) => item.id === currentId);
        if (index === -1) return items[0]?.id || null;
        return items[index + 1]?.id || items[index - 1]?.id || null;
    }

    function setActiveId(key, storageKey, id) {
        state[key] = id || null;
        if (id) localStorage.setItem(storageKey, id);
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

    function normalizeTags(tags) {
        return uniqueTextValues(Array.isArray(tags) ? tags : [])
            .filter(Boolean)
            .slice(0, 12);
    }

    function normalizeTaxonomy(data = null) {
        return {
            exerciseFocuses: uniqueTextValues([
                ...DEFAULT_EXERCISE_FOCUSES,
                ...(Array.isArray(data?.exerciseFocuses) ? data.exerciseFocuses : [])
            ]),
            materialTypes: normalizeMaterialTypes(data?.materialTypes),
            readingLenses: uniqueTextValues([
                ...DEFAULT_READING_LENSES,
                ...(Array.isArray(data?.readingLenses) ? data.readingLenses : [])
            ])
        };
    }

    function normalizeMaterialTypes(types) {
        const map = new Map(DEFAULT_MATERIAL_TYPES.map((type) => [type.id, { ...type }]));
        if (Array.isArray(types)) {
            types.forEach((type) => {
                if (!type || !type.id || !type.label) return;
                const existing = map.get(type.id);
                map.set(type.id, {
                    id: String(type.id),
                    label: String(type.label).trim(),
                    builtin: Boolean(existing?.builtin || type.builtin),
                    hidden: Boolean(type.hidden)
                });
            });
        }
        return Array.from(map.values()).filter((type) => type.label);
    }

    function getVisibleMaterialTypes() {
        return state.taxonomy.materialTypes.filter((type) => !type.hidden);
    }

    function uniqueTextValues(values) {
        const seen = new Set();
        const result = [];
        values.forEach((value) => {
            const clean = String(value || '').trim();
            const key = clean.toLowerCase();
            if (!clean || seen.has(key)) return;
            seen.add(key);
            result.push(clean);
        });
        return result;
    }

    function getTextCategoryConfig(kind) {
        return {
            exercise: {
                select: els.exerciseFocus,
                input: els.exerciseCustomFocus,
                options: state.taxonomy.exerciseFocuses
            },
            reading: {
                select: els.readingLens,
                input: els.readingCustomLens,
                options: state.taxonomy.readingLenses
            }
        }[kind];
    }

    function getMaterialLabel(typeId) {
        const match = state.taxonomy.materialTypes.find((type) => type.id === typeId);
        return match?.label || DEFAULT_MATERIAL_TYPES.find((type) => type.id === typeId)?.label || typeId || '素材';
    }

    function makeCustomTypeId(label) {
        const ascii = label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 24);
        const suffix = `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
        return `custom_${ascii || 'type'}_${suffix}`;
    }

    function countWords(text) {
        const value = (text || '').trim();
        if (!value) return 0;
        const cjkMatches = value.match(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g) || [];
        const latinText = value.replace(/[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g, ' ');
        const latinMatches = latinText.match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g) || [];
        return cjkMatches.length + latinMatches.length;
    }

    function getHistoryWeeks() {
        const currentWeekId = getWeekRange().weekId;
        const weekMap = new Map();
        const ensureWeek = (weekId) => {
            const normalized = normalizeWeekStartId(weekId);
            if (compareWeekIds(normalized, currentWeekId) > 0) return null;
            if (!weekMap.has(normalized)) {
                const range = getWeekRange(parseDateId(normalized));
                weekMap.set(normalized, {
                    ...range,
                    draftWords: 0,
                    exerciseWords: 0,
                    totalWords: 0,
                    hasReview: false
                });
            }
            return weekMap.get(normalized);
        };

        ensureWeek(currentWeekId);
        ensureWeek(state.selectedWeekStartId);

        state.stats.forEach((entry) => {
            if (!entry.date) return;
            const week = ensureWeek(getWeekRange(parseDateId(entry.date)).weekId);
            if (!week) return;
            week.draftWords += Number(entry.draftWords || 0);
            week.exerciseWords += Number(entry.exerciseWords || 0);
            week.totalWords += Number(entry.totalWords || 0);
        });

        state.weeklyReviews.forEach((review) => {
            const weekId = review.weekStart || review.id;
            const week = ensureWeek(weekId);
            if (!week) return;
            week.hasReview = Boolean((review.manualReview || '').trim() || (review.aiSummary || '').trim());
            if (!week.totalWords && review.statsSnapshot?.totalWords) {
                week.totalWords = Number(review.statsSnapshot.totalWords || 0);
            }
        });

        return Array.from(weekMap.values())
            .filter((week) => week.weekId === currentWeekId || week.weekId === state.selectedWeekStartId || week.totalWords > 0 || week.hasReview)
            .sort((a, b) => compareWeekIds(b.weekId, a.weekId))
            .slice(0, 78);
    }

    function getWeekHistoryMeta(week) {
        const parts = [];
        if (week.weekId === getWeekRange().weekId) parts.push('本周');
        if (week.totalWords > 0) parts.push(`${formatNumber(week.totalWords)} 字`);
        if (week.hasReview) parts.push('复盘');
        return parts.join(' / ') || '空白';
    }

    function getWeekMonthLabel(weekId) {
        const date = parseDateId(weekId);
        return `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}`;
    }

    function getSelectedWeekRange() {
        return getWeekRange(parseDateId(state.selectedWeekStartId));
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
            startLabel: startParts.label,
            endLabel: endParts.label
        };
    }

    function normalizeWeekStartId(value) {
        const parsed = parseDateId(value);
        return getWeekRange(parsed).weekId;
    }

    function compareWeekIds(a, b) {
        return parseDateId(a).getTime() - parseDateId(b).getTime();
    }

    function parseDateId(value) {
        if (value instanceof Date) {
            return new Date(value.getFullYear(), value.getMonth(), value.getDate());
        }
        if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
            return new Date();
        }
        const [year, month, day] = value.split('-').map(Number);
        const parsed = new Date(year, month - 1, day);
        if (Number.isNaN(parsed.getTime())) return new Date();
        return parsed;
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

    function formatScore(value) {
        const score = Number(value || 0);
        if (!score) return '-';
        return `${Math.max(1, Math.min(5, score)).toFixed(1)}`;
    }

    function normalizeFunctionError(error) {
        const message = String(error?.message || '').trim();
        if (message) return message;
        if (error?.code === 'functions/already-exists') return '已经生成过，第一版不会覆盖旧结果。';
        if (error?.code === 'functions/failed-precondition') return '当前内容还不满足生成条件。';
        if (error?.code === 'functions/unauthenticated') return '请先登录后再使用 AI 功能。';
        return '生成时遇到问题，请稍后再试。';
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
