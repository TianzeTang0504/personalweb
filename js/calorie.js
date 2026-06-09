document.addEventListener('DOMContentLoaded', () => {
    const $ = (selector) => document.querySelector(selector);

    const DEFAULT_SETTINGS = {
        dailyCalorieTarget: 3000,
        proteinTarget: 125,
        carbsTarget: 425,
        fatTarget: 90,
        units: 'metric'
    };
    const DEFAULT_MEAL_DEFS = [
        { id: 'default_breakfast', name: '早餐' },
        { id: 'default_lunch', name: '午餐' },
        { id: 'default_afternoon', name: '下午茶' },
        { id: 'default_dinner', name: '晚餐' },
        { id: 'default_late', name: '夜宵' }
    ];
    const DEFAULT_MEALS = DEFAULT_MEAL_DEFS.map((meal) => meal.name);
    const UNIT_OPTIONS = ['g', 'ml', '份', '个', '片', '勺', '碗', '杯', '袋', '盒'];
    const FOOD_STATES = [
        { value: 'raw', label: '生重' },
        { value: 'cooked', label: '熟重' },
        { value: 'packaged', label: '包装' },
        { value: 'serving', label: '份量' },
        { value: 'unknown', label: '未注明' }
    ];
    const NUTRIENTS = [
        { key: 'kcal', label: '热量', unit: 'kcal' },
        { key: 'protein', label: '蛋白质', unit: 'g' },
        { key: 'carbs', label: '碳水', unit: 'g' },
        { key: 'fat', label: '脂肪', unit: 'g' }
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
        topDateLabel: $('#topDateLabel'),
        topCalorieStatus: $('#topCalorieStatus'),
        calorieNav: $('#calorieNav'),
        todayNavBadge: $('#todayNavBadge'),
        timelineNavBadge: $('#timelineNavBadge'),
        bodyNavBadge: $('#bodyNavBadge'),
        settingsNavBadge: $('#settingsNavBadge'),
        prevDateBtn: $('#prevDateBtn'),
        nextDateBtn: $('#nextDateBtn'),
        todayBtn: $('#todayBtn'),
        dateInput: $('#dateInput'),
        metricCalories: $('#metricCalories'),
        metricCalorieDelta: $('#metricCalorieDelta'),
        metricProtein: $('#metricProtein'),
        metricCarbs: $('#metricCarbs'),
        metricFat: $('#metricFat'),
        proteinTargetLabel: $('#proteinTargetLabel'),
        carbsTargetLabel: $('#carbsTargetLabel'),
        fatTargetLabel: $('#fatTargetLabel'),
        estimateStatusTitle: $('#estimateStatusTitle'),
        estimateFreshness: $('#estimateFreshness'),
        estimateBtn: $('#estimateBtn'),
        estimateResult: $('#estimateResult'),
        addMealBtn: $('#addMealBtn'),
        daySaveState: $('#daySaveState'),
        mealList: $('#mealList'),
        dayNote: $('#dayNote'),
        rangeControl: $('#rangeControl'),
        nutritionChart: $('#nutritionChart'),
        historyList: $('#historyList'),
        bodyDateInput: $('#bodyDateInput'),
        weightInput: $('#weightInput'),
        bodyFatInput: $('#bodyFatInput'),
        bodyNote: $('#bodyNote'),
        bodySaveState: $('#bodySaveState'),
        bodyChart: $('#bodyChart'),
        bodyHistoryList: $('#bodyHistoryList'),
        dailyCalorieTarget: $('#dailyCalorieTarget'),
        proteinTarget: $('#proteinTarget'),
        carbsTarget: $('#carbsTarget'),
        fatTarget: $('#fatTarget'),
        settingsSaveState: $('#settingsSaveState'),
        appDialog: $('#appDialog'),
        appDialogKicker: $('#appDialogKicker'),
        appDialogTitle: $('#appDialogTitle'),
        appDialogMessage: $('#appDialogMessage'),
        appDialogCancelBtn: $('#appDialogCancelBtn'),
        appDialogConfirmBtn: $('#appDialogConfirmBtn')
    };

    const state = {
        auth: null,
        db: null,
        functions: null,
        user: null,
        view: 'today',
        selectedDateId: getDateParts(new Date()).id,
        timelineRange: 14,
        settings: { ...DEFAULT_SETTINGS },
        days: [],
        bodyLogs: [],
        unsubs: [],
        timers: {
            day: null,
            body: null,
            settings: null
        },
        isRendering: false,
        estimatingDateId: null,
        dialogResolve: null
    };

    init();

    function init() {
        bindStaticEvents();
        els.dateInput.value = state.selectedDateId;
        els.bodyDateInput.value = state.selectedDateId;

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
            showAuthMessage(error.message || '热量记录初始化失败。');
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

        els.calorieNav.addEventListener('click', (event) => {
            const tab = event.target.closest('[data-view]');
            if (!tab) return;
            setView(tab.getAttribute('data-view'));
        });

        els.prevDateBtn.addEventListener('click', () => shiftSelectedDate(-1));
        els.nextDateBtn.addEventListener('click', () => shiftSelectedDate(1));
        els.todayBtn.addEventListener('click', () => selectDate(getDateParts(new Date()).id));
        els.dateInput.addEventListener('change', () => selectDate(els.dateInput.value));
        els.bodyDateInput.addEventListener('change', () => selectDate(els.bodyDateInput.value));

        els.addMealBtn.addEventListener('click', addMeal);
        els.estimateBtn.addEventListener('click', estimateCurrentDay);
        els.dayNote.addEventListener('input', handleDayInput);

        els.mealList.addEventListener('input', (event) => {
            if (event.target.closest('[data-day-field]')) handleDayInput();
        });
        els.mealList.addEventListener('change', (event) => {
            if (event.target.closest('[data-day-field]')) handleDayInput();
        });
        els.mealList.addEventListener('click', async (event) => {
            const toggleMealBtn = event.target.closest('[data-toggle-meal]');
            const addBtn = event.target.closest('[data-add-ingredient]');
            const deleteMealBtn = event.target.closest('[data-delete-meal]');
            const deleteIngredientBtn = event.target.closest('[data-delete-ingredient]');

            if (toggleMealBtn) {
                toggleMeal(toggleMealBtn.getAttribute('data-toggle-meal'), toggleMealBtn.closest('[data-meal-id]'));
                return;
            }
            if (addBtn) {
                addIngredient(addBtn.getAttribute('data-add-ingredient'));
                return;
            }
            if (deleteMealBtn) {
                await deleteMeal(deleteMealBtn.getAttribute('data-delete-meal'));
                return;
            }
            if (deleteIngredientBtn) {
                await deleteIngredient(deleteIngredientBtn.getAttribute('data-delete-ingredient'));
            }
        });

        els.rangeControl.addEventListener('click', (event) => {
            const btn = event.target.closest('[data-range]');
            if (!btn) return;
            state.timelineRange = Number(btn.getAttribute('data-range')) || 14;
            setLocalValue('calorieTimelineRange', String(state.timelineRange));
            renderTimeline();
        });

        [els.weightInput, els.bodyFatInput, els.bodyNote].forEach((input) => {
            input.addEventListener('input', scheduleBodySave);
        });

        [els.dailyCalorieTarget, els.proteinTarget, els.carbsTarget, els.fatTarget].forEach((input) => {
            input.addEventListener('input', scheduleSettingsSave);
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
    }

    async function handleAuthChange(user) {
        clearTimers();
        cleanupListeners();
        state.user = user;

        if (!user) {
            resetLoadedUserData();
            showSignedOut();
            return;
        }

        if (needsEmailVerification(user)) {
            await state.auth.signOut();
            showSignedOut();
            showAuthMessage('请先完成邮箱验证，再进入热量管理。');
            return;
        }

        loadUserLocalState(user);
        showSignedIn(user);
        await state.db.collection('users').doc(user.uid).set({
            email: user.email || '',
            displayName: user.displayName || '',
            photoURL: user.photoURL || '',
            authProviders: user.providerData.map((provider) => provider.providerId).filter(Boolean),
            lastLoginAt: serverTimestamp()
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
        els.userEmail.textContent = user.displayName || user.email || '已登录';
        setView(state.view);
    }

    function resetLoadedUserData() {
        state.settings = { ...DEFAULT_SETTINGS };
        state.days = [];
        state.bodyLogs = [];
        state.selectedDateId = getDateParts(new Date()).id;
        state.timelineRange = 14;
        state.estimatingDateId = null;
    }

    function loadUserLocalState(user) {
        resetLoadedUserData();
        state.view = getLocalValue('calorieView', 'today', user.uid);
        state.selectedDateId = normalizeDateId(getLocalValue('calorieSelectedDate', getDateParts(new Date()).id, user.uid));
        state.timelineRange = Number(getLocalValue('calorieTimelineRange', '14', user.uid)) || 14;
        els.dateInput.value = state.selectedDateId;
        els.bodyDateInput.value = state.selectedDateId;
    }

    function setupListeners() {
        const userRef = state.db.collection('users').doc(state.user.uid);

        state.unsubs.push(userRef.collection('calorieSettings').doc('main').onSnapshot((doc) => {
            state.settings = normalizeSettings(doc.exists ? doc.data() : null);
            renderSettings();
            renderAll();
        }, (error) => {
            console.error('Failed to read calorie settings', error);
        }));

        state.unsubs.push(userRef.collection('calorieDays').onSnapshot((snapshot) => {
            state.days = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
            renderAll();
        }, (error) => {
            console.error('Failed to read calorie days', error);
        }));

        state.unsubs.push(userRef.collection('bodyLogs').onSnapshot((snapshot) => {
            state.bodyLogs = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
            renderAll();
        }, (error) => {
            console.error('Failed to read body logs', error);
        }));
    }

    function cleanupListeners() {
        state.unsubs.forEach((unsub) => unsub());
        state.unsubs = [];
    }

    function setView(view) {
        state.view = view || 'today';
        setLocalValue('calorieView', state.view);
        document.querySelectorAll('.calorie-view').forEach((panel) => {
            panel.classList.toggle('active', panel.id === `view-${state.view}`);
        });
        els.calorieNav.querySelectorAll('.nav-tab').forEach((tab) => {
            tab.classList.toggle('active', tab.getAttribute('data-view') === state.view);
        });
        renderAll();
    }

    async function selectDate(dateId) {
        const normalized = normalizeDateId(dateId);
        if (state.user && normalized !== state.selectedDateId) {
            await saveCurrentDay();
            await saveBodyLog(state.selectedDateId);
        }
        state.selectedDateId = normalized;
        els.dateInput.value = state.selectedDateId;
        els.bodyDateInput.value = state.selectedDateId;
        setLocalValue('calorieSelectedDate', state.selectedDateId);
        renderAll(true);
    }

    function shiftSelectedDate(offset) {
        const date = parseDateId(state.selectedDateId);
        date.setDate(date.getDate() + offset);
        selectDate(getDateParts(date).id);
    }

    function renderAll(force = false) {
        if (!state.user) return;
        renderCurrentDay(force);
        renderBody(force);
        renderTimeline();
        renderNavAndTopbar();
    }

    function renderCurrentDay(force = false) {
        const day = getCurrentDay();
        const focusedInDay = isFocusedWithin([els.mealList, els.dayNote]);

        if (force || !focusedInDay) {
            state.isRendering = true;
            els.dateInput.value = state.selectedDateId;
            els.dayNote.value = day.note || '';
            renderMeals(day);
            state.isRendering = false;
        }

        renderEstimate(day);
        renderTodayMetrics(day);
    }

    function renderMeals(day) {
        const meals = getMealsForRender(day);
        const ingredients = normalizeIngredients(day.ingredients);

        if (!meals.length) {
            els.mealList.innerHTML = '<div class="empty-state">还没有餐次</div>';
            return;
        }

        els.mealList.innerHTML = meals.map((meal) => {
            const mealIngredients = ingredients.filter((item) => item.mealId === meal.id);
            const mealTotals = sumIngredientTotals(day, mealIngredients);
            const expanded = isMealExpanded(meal.id);
            const mealBodyId = `meal-body-${meal.id}`;
            return `
                <article class="meal-card ${expanded ? '' : 'is-collapsed'}" data-meal-id="${escapeAttr(meal.id)}">
                    <div class="meal-head">
                        <div class="meal-title-row">
                            <button class="meal-toggle" data-toggle-meal="${escapeAttr(meal.id)}" type="button"
                                aria-expanded="${expanded ? 'true' : 'false'}" aria-controls="${escapeAttr(mealBodyId)}"
                                aria-label="${escapeAttr(`${expanded ? '收起' : '展开'}${meal.name}`)}">
                                <span class="meal-toggle-mark" aria-hidden="true"></span>
                            </button>
                            <input class="text-input meal-name-input" data-day-field data-meal-name="${escapeAttr(meal.id)}"
                                type="text" autocomplete="off" value="${escapeAttr(meal.name)}" placeholder="餐次">
                            <span class="meal-total">${escapeHtml(formatMealTotal(mealTotals, mealIngredients.length))}</span>
                        </div>
                        <div class="meal-actions">
                            <button class="secondary-btn compact" data-add-ingredient="${escapeAttr(meal.id)}" type="button">加食物</button>
                            <button class="danger-btn compact" data-delete-meal="${escapeAttr(meal.id)}" type="button">删除餐次</button>
                        </div>
                    </div>
                    <div class="meal-body" id="${escapeAttr(mealBodyId)}">
                        ${renderIngredientTable(mealIngredients)}
                    </div>
                </article>
            `;
        }).join('');
    }

    function renderIngredientTable(ingredients) {
        if (!ingredients.length) {
            return '<div class="empty-state">这一餐还没有食物</div>';
        }

        return `
            <div class="ingredient-table-wrap">
                <table class="ingredient-table">
                    <thead>
                        <tr>
                            <th>食物名称</th>
                            <th>量</th>
                            <th>单位</th>
                            <th>状态</th>
                            <th>备注</th>
                            <th>kcal/100g</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${ingredients.map(renderIngredientRow).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    function renderIngredientRow(item) {
        const label = item.labelPer100g || {};
        return `
            <tr data-ingredient-id="${escapeAttr(item.id)}">
                <td class="ingredient-name-cell">
                    <input class="text-input" data-day-field data-ingredient-field="name" type="text"
                        autocomplete="off" value="${escapeAttr(item.name)}" placeholder="鸡蛋 / 牛奶 / 米饭 / 牛肉饭">
                </td>
                <td class="ingredient-number-cell">
                    <input class="number-input mini-input" data-day-field data-ingredient-field="amount" type="number"
                        inputmode="decimal" min="0" step="0.1" value="${escapeAttr(numberToInput(item.amount))}">
                </td>
                <td class="ingredient-unit-cell">
                    <select class="select-input" data-day-field data-ingredient-field="unit">
                        ${UNIT_OPTIONS.map((unit) => (
                            `<option value="${escapeAttr(unit)}" ${unit === item.unit ? 'selected' : ''}>${escapeHtml(unit)}</option>`
                        )).join('')}
                    </select>
                </td>
                <td class="ingredient-state-cell">
                    <select class="select-input" data-day-field data-ingredient-field="state">
                        ${FOOD_STATES.map((option) => (
                            `<option value="${escapeAttr(option.value)}" ${option.value === item.state ? 'selected' : ''}>${escapeHtml(option.label)}</option>`
                        )).join('')}
                    </select>
                </td>
                <td class="ingredient-note-cell">
                    <input class="text-input" data-day-field data-ingredient-field="note" type="text"
                        autocomplete="off" value="${escapeAttr(item.note)}" placeholder="品牌 / 外食 / 熟重">
                </td>
                <td class="ingredient-label-cell">
                    <input class="number-input mini-input" data-day-field data-label-field="kcal" type="number"
                        inputmode="decimal" min="0" step="0.1" value="${escapeAttr(numberToInput(label.kcal))}">
                </td>
                <td class="ingredient-actions-cell">
                    <button class="danger-btn compact" data-delete-ingredient="${escapeAttr(item.id)}" type="button">删除</button>
                </td>
            </tr>
        `;
    }

    function toggleMeal(mealId, card) {
        if (!mealId) return;
        const expanded = !isMealExpanded(mealId);
        setMealExpanded(mealId, expanded);
        syncMealCardCollapsedState(card, expanded);
    }

    function syncMealCardCollapsedState(card, expanded) {
        if (!card) return;
        card.classList.toggle('is-collapsed', !expanded);
        const button = card.querySelector('[data-toggle-meal]');
        const mealName = card.querySelector('[data-meal-name]')?.value.trim() || '餐次';
        if (button) {
            button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
            button.setAttribute('aria-label', `${expanded ? '收起' : '展开'}${mealName}`);
        }
    }

    function renderTodayMetrics(day) {
        const totals = getDayTotals(day);
        const target = Number(state.settings.dailyCalorieTarget || DEFAULT_SETTINGS.dailyCalorieTarget);
        const calorieDelta = Math.round(target - totals.kcal.mid);

        els.metricCalories.textContent = formatNumber(totals.kcal.mid, 0);
        els.metricCalorieDelta.textContent = calorieDelta > 0
            ? `差 ${formatNumber(calorieDelta, 0)} kcal`
            : `超 ${formatNumber(Math.abs(calorieDelta), 0)} kcal`;
        els.metricProtein.textContent = `${formatNumber(totals.protein.mid, 1)}g`;
        els.metricCarbs.textContent = `${formatNumber(totals.carbs.mid, 1)}g`;
        els.metricFat.textContent = `${formatNumber(totals.fat.mid, 1)}g`;
        els.proteinTargetLabel.textContent = formatMacroTarget('protein', totals.protein.mid);
        els.carbsTargetLabel.textContent = formatMacroTarget('carbs', totals.carbs.mid);
        els.fatTargetLabel.textContent = formatMacroTarget('fat', totals.fat.mid);
    }

    function renderEstimate(day) {
        const estimate = day.aiEstimate || null;
        const fresh = isEstimateFresh(day);
        const hasEstimate = Boolean(estimate);

        els.estimateBtn.disabled = state.estimatingDateId === state.selectedDateId;
        els.estimateBtn.textContent = state.estimatingDateId === state.selectedDateId ? '估算中...' : 'AI 估算';
        els.estimateStatusTitle.textContent = hasEstimate
            ? fresh ? '估算已更新' : '估算已过期'
            : '尚未估算';
        els.estimateFreshness.textContent = hasEstimate
            ? fresh ? '当前输入' : '需要重算'
            : '未生成';
        els.estimateFreshness.className = `status-pill ${hasEstimate ? fresh ? 'good' : 'warn' : 'muted'}`;

        if (!hasEstimate) {
            els.estimateResult.classList.add('hidden');
            els.estimateResult.innerHTML = '';
            return;
        }

        const totals = normalizeTotals(estimate.totals);
        const ingredientEstimates = Array.isArray(estimate.ingredientEstimates) ? estimate.ingredientEstimates : [];
        els.estimateResult.classList.remove('hidden');
        els.estimateResult.innerHTML = `
            <div class="estimate-summary-grid">
                ${NUTRIENTS.map((nutrient) => `
                    <div class="estimate-summary">
                        <span>${escapeHtml(nutrient.label)}</span>
                        <strong>${escapeHtml(formatRange(totals[nutrient.key], nutrient.unit))}</strong>
                    </div>
                `).join('')}
            </div>
            ${renderDailyAssessment(estimate.dailyAssessment)}
            <div class="estimate-columns">
                <div class="estimate-table-wrap">
                    <table class="estimate-table">
                        <thead>
                            <tr>
                                <th>食物名称</th>
                                <th>kcal</th>
                                <th>P</th>
                                <th>C</th>
                                <th>F</th>
                                <th>依据</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${ingredientEstimates.map(renderEstimateRow).join('')}
                        </tbody>
                    </table>
                </div>
                <div class="estimate-notes">
                    ${renderNoteBlock('假设', estimate.assumptions)}
                    ${renderNoteBlock('提醒', estimate.warnings)}
                    ${renderNoteBlock('缺失信息', estimate.missingInfo)}
                </div>
            </div>
        `;
    }

    function renderDailyAssessment(assessment) {
        const text = String(assessment || '').trim();
        if (!text) return '';
        return `
            <div class="daily-assessment-box">
                <label for="dailyAssessmentText">今日饮食评估</label>
                <textarea id="dailyAssessmentText" class="assessment-textarea" readonly>${escapeHtml(text)}</textarea>
            </div>
        `;
    }

    function renderEstimateRow(item) {
        return `
            <tr>
                <td>${escapeHtml(item.name || '未命名食物')}</td>
                <td>${escapeHtml(formatRange(item.kcal, ''))}</td>
                <td>${escapeHtml(formatRange(item.protein, ''))}</td>
                <td>${escapeHtml(formatRange(item.carbs, ''))}</td>
                <td>${escapeHtml(formatRange(item.fat, ''))}</td>
                <td>${escapeHtml(item.basis || item.source || '')}</td>
            </tr>
        `;
    }

    function renderNoteBlock(title, values) {
        const list = (Array.isArray(values) ? values : []).filter(Boolean);
        if (!list.length) return '';
        return `
            <div class="note-block">
                <h4>${escapeHtml(title)}</h4>
                <ul>${list.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
            </div>
        `;
    }

    async function addMeal() {
        const day = getCurrentFormOrStateDay();
        const nextMeal = {
            id: makeId('meal'),
            name: `餐次 ${getMealsForRender(day).length + 1}`,
            order: getMealsForRender(day).length
        };
        day.meals = [...getMealsForRender(day), nextMeal];
        setMealExpanded(nextMeal.id, true);
        await saveDayObject(day);
        renderCurrentDay(true);
    }

    async function addIngredient(mealId = null) {
        let day = getCurrentFormOrStateDay();
        let meals = getMealsForRender(day);
        if (!meals.length) {
            meals = [{ id: makeId('meal'), name: '早餐', order: 0 }];
            day = { ...day, meals };
        }
        const targetMealId = mealId || meals[0].id;
        const nextIngredient = {
            id: makeId('ing'),
            mealId: targetMealId,
            name: '',
            amount: null,
            unit: 'g',
            state: 'raw',
            note: '',
            labelPer100g: {},
            order: normalizeIngredients(day.ingredients).filter((item) => item.mealId === targetMealId).length
        };
        day.ingredients = [...normalizeIngredients(day.ingredients), nextIngredient];
        setMealExpanded(targetMealId, true);
        await saveDayObject(day);
        renderCurrentDay(true);
    }

    async function deleteMeal(mealId) {
        const confirmed = await confirmDanger('删除餐次', '删除这个餐次和其中的食物？');
        if (!confirmed) return;
        const day = getCurrentFormOrStateDay();
        day.meals = getMealsForRender(day).filter((meal) => meal.id !== mealId);
        day.ingredients = normalizeIngredients(day.ingredients).filter((item) => item.mealId !== mealId);
        setMealExpanded(mealId, false);
        await saveDayObject(day);
        renderCurrentDay(true);
    }

    async function deleteIngredient(ingredientId) {
        const day = getCurrentFormOrStateDay();
        day.ingredients = normalizeIngredients(day.ingredients).filter((item) => item.id !== ingredientId);
        await saveDayObject(day);
        renderCurrentDay(true);
    }

    function handleDayInput() {
        if (state.isRendering) return;
        renderTodayMetrics(getCurrentFormOrStateDay());
        renderEstimate(getCurrentFormOrStateDay());
        scheduleDaySave();
    }

    function scheduleDaySave() {
        debounce('day', saveCurrentDay, els.daySaveState);
    }

    async function saveCurrentDay() {
        if (!state.user || !document.body.contains(els.mealList)) return;
        await saveDayObject(getCurrentFormOrStateDay());
    }

    async function saveDayObject(day) {
        if (!state.user) return;
        const cleanDay = normalizeDayForSave(day);
        if (!getDayByDate(cleanDay.dateId) && !isMeaningfulDay(cleanDay)) {
            els.daySaveState.textContent = '已保存';
            return;
        }
        await getUserRef('calorieDays').doc(cleanDay.dateId).set({
            dateId: cleanDay.dateId,
            meals: cleanDay.meals,
            ingredients: cleanDay.ingredients,
            note: cleanDay.note,
            inputHash: cleanDay.inputHash,
            updatedAt: serverTimestamp()
        }, { merge: true });
        upsertLocalDay(cleanDay);
        els.daySaveState.textContent = '已保存';
    }

    async function estimateCurrentDay() {
        if (!state.functions) {
            await showNotice('AI 暂不可用', 'Firebase Functions SDK 没有加载成功。');
            return;
        }

        const day = getCurrentFormOrStateDay();
        const namedIngredients = normalizeIngredients(day.ingredients).filter((item) => item.name.trim());
        const missingAmount = namedIngredients.find((item) => !Number(item.amount));
        if (!namedIngredients.length) {
            await showNotice('没有可估算的食物', '先填至少一条食物名称和数量。');
            return;
        }
        if (missingAmount) {
            await showNotice('数量缺失', `“${missingAmount.name}” 需要填写数量。`);
            return;
        }

        try {
            await saveDayObject(day);
            state.estimatingDateId = state.selectedDateId;
            renderEstimate(getCurrentDay());
            const estimate = state.functions.httpsCallable('estimateCalorieDay');
            const result = await estimate({ dateId: state.selectedDateId });
            if (result.data?.aiEstimate) {
                const localDay = {
                    ...getCurrentDay(),
                    inputHash: result.data.inputHash || getCurrentDay().inputHash,
                    aiEstimate: result.data.aiEstimate
                };
                upsertLocalDay(localDay);
                renderEstimate(localDay);
                renderTodayMetrics(localDay);
            }
            await showNotice('估算已生成', '热量和三大营养素已经写回当天记录。');
        } catch (error) {
            console.error('Failed to estimate calorie day', error);
            await showNotice('估算失败', normalizeFunctionError(error));
        } finally {
            state.estimatingDateId = null;
            renderEstimate(getCurrentDay());
        }
    }

    function renderTimeline() {
        const rangeDays = getRangeDateIds(state.selectedDateId, state.timelineRange);
        const rows = rangeDays.map((dateId) => {
            const day = getDayByDate(dateId);
            return {
                dateId,
                day,
                totals: getDayTotals(day || getEmptyDay(dateId))
            };
        });

        els.timelineNavBadge.textContent = `${state.timelineRange} 天`;
        els.rangeControl.querySelectorAll('[data-range]').forEach((btn) => {
            btn.classList.toggle('active', Number(btn.getAttribute('data-range')) === state.timelineRange);
        });
        renderNutritionChart(rows);
        renderHistoryList(rows);
    }

    function renderNutritionChart(rows) {
        const hasData = rows.some((row) => row.day && (row.day.aiEstimate || normalizeIngredients(row.day.ingredients).length));
        if (!hasData) {
            els.nutritionChart.innerHTML = '<div class="chart-empty">暂无历史记录</div>';
            return;
        }

        els.nutritionChart.innerHTML = `
            ${renderLegend([
                ['line-calorie', '热量'],
                ['line-protein', '蛋白质'],
                ['line-carbs', '碳水'],
                ['line-fat', '脂肪']
            ])}
            ${renderLineChart(rows, [
                { key: 'kcal', className: 'line-calorie' },
                { key: 'protein', className: 'line-protein' },
                { key: 'carbs', className: 'line-carbs' },
                { key: 'fat', className: 'line-fat' }
            ])}
        `;
    }

    function renderHistoryList(rows) {
        const items = [...rows].reverse().filter((row) => row.day);
        if (!items.length) {
            els.historyList.innerHTML = '<div class="empty-state">暂无历史记录</div>';
            return;
        }

        els.historyList.innerHTML = items.map((row) => {
            const totals = row.totals;
            const delta = Math.round(totals.kcal.mid - Number(state.settings.dailyCalorieTarget || 3000));
            const estimateText = row.day.aiEstimate
                ? isEstimateFresh(row.day) ? '当前估算' : '估算过期'
                : '未估算';
            return `
                <button class="history-item" type="button" data-history-date="${escapeAttr(row.dateId)}">
                    <span class="history-date">${escapeHtml(formatDateLabel(row.dateId))}</span>
                    <span class="history-detail">
                        ${escapeHtml(formatNumber(totals.kcal.mid, 0))} kcal ·
                        P ${escapeHtml(formatNumber(totals.protein.mid, 1))}g ·
                        C ${escapeHtml(formatNumber(totals.carbs.mid, 1))}g ·
                        F ${escapeHtml(formatNumber(totals.fat.mid, 1))}g
                    </span>
                    <span class="history-meta">${delta >= 0 ? '+' : ''}${escapeHtml(String(delta))} · ${escapeHtml(estimateText)}</span>
                </button>
            `;
        }).join('');

        els.historyList.querySelectorAll('[data-history-date]').forEach((item) => {
            item.addEventListener('click', () => {
                selectDate(item.getAttribute('data-history-date'));
                setView('today');
            });
        });
    }

    function renderBody(force = false) {
        const log = getBodyLogByDate(state.selectedDateId) || getEmptyBodyLog(state.selectedDateId);
        const focusedInBody = isFocusedWithin([els.weightInput, els.bodyFatInput, els.bodyNote, els.bodyDateInput]);

        if (force || !focusedInBody) {
            state.isRendering = true;
            els.bodyDateInput.value = state.selectedDateId;
            els.weightInput.value = numberToInput(log.weightKg);
            els.bodyFatInput.value = numberToInput(log.bodyFatPct);
            els.bodyNote.value = log.note || '';
            state.isRendering = false;
        }

        const latest = getLatestBodyLog();
        els.bodyNavBadge.textContent = latest?.weightKg ? `${formatNumber(latest.weightKg, 1)} kg` : '-- kg';
        renderBodyChart();
        renderBodyHistory();
    }

    function scheduleBodySave() {
        if (state.isRendering) return;
        debounce('body', saveBodyLog, els.bodySaveState);
    }

    async function saveBodyLog(dateOverride = null) {
        if (!state.user) return;
        const dateId = normalizeDateId(dateOverride || els.bodyDateInput.value || state.selectedDateId);
        const bodyLog = {
            dateId,
            weightKg: nullableNumber(els.weightInput.value),
            bodyFatPct: nullableNumber(els.bodyFatInput.value),
            note: els.bodyNote.value.trim(),
            updatedAt: serverTimestamp()
        };
        if (!getBodyLogByDate(dateId) && !bodyLog.weightKg && !bodyLog.bodyFatPct && !bodyLog.note) {
            els.bodySaveState.textContent = '已保存';
            return;
        }
        await getUserRef('bodyLogs').doc(dateId).set(bodyLog, { merge: true });
        upsertLocalBodyLog(bodyLog);
        els.bodySaveState.textContent = '已保存';
    }

    function renderBodyChart() {
        const rows = getRangeDateIds(state.selectedDateId, state.timelineRange).map((dateId) => {
            const log = getBodyLogByDate(dateId);
            return {
                dateId,
                day: log,
                totals: {
                    weight: makeExactRange(log?.weightKg || 0),
                    bodyfat: makeExactRange(log?.bodyFatPct || 0)
                }
            };
        });
        const hasData = rows.some((row) => row.day && (row.day.weightKg || row.day.bodyFatPct));
        if (!hasData) {
            els.bodyChart.innerHTML = '<div class="chart-empty">暂无身体指标</div>';
            return;
        }
        els.bodyChart.innerHTML = `
            ${renderLegend([
                ['line-weight', '体重'],
                ['line-bodyfat', '体脂率']
            ])}
            ${renderLineChart(rows, [
                { key: 'weight', className: 'line-weight' },
                { key: 'bodyfat', className: 'line-bodyfat' }
            ])}
        `;
    }

    function renderBodyHistory() {
        const items = sortByDateDesc(state.bodyLogs.filter((log) => log.weightKg || log.bodyFatPct || log.note));
        if (!items.length) {
            els.bodyHistoryList.innerHTML = '<div class="empty-state">暂无身体记录</div>';
            return;
        }

        els.bodyHistoryList.innerHTML = items.slice(0, 30).map((log) => `
            <button class="history-item" type="button" data-body-date="${escapeAttr(log.dateId || log.id)}">
                <span class="history-date">${escapeHtml(formatDateLabel(log.dateId || log.id))}</span>
                <span class="history-detail">${escapeHtml(log.note || '身体指标')}</span>
                <span class="history-meta">
                    ${log.weightKg ? `${escapeHtml(formatNumber(log.weightKg, 1))} kg` : '-- kg'} ·
                    ${log.bodyFatPct ? `${escapeHtml(formatNumber(log.bodyFatPct, 1))}%` : '-- %'}
                </span>
            </button>
        `).join('');

        els.bodyHistoryList.querySelectorAll('[data-body-date]').forEach((item) => {
            item.addEventListener('click', () => selectDate(item.getAttribute('data-body-date')));
        });
    }

    function renderSettings() {
        if (isFocusedWithin([els.dailyCalorieTarget, els.proteinTarget, els.carbsTarget, els.fatTarget])) return;
        els.dailyCalorieTarget.value = numberToInput(state.settings.dailyCalorieTarget || DEFAULT_SETTINGS.dailyCalorieTarget);
        els.proteinTarget.value = numberToInput(state.settings.proteinTarget);
        els.carbsTarget.value = numberToInput(state.settings.carbsTarget);
        els.fatTarget.value = numberToInput(state.settings.fatTarget);
        els.settingsNavBadge.textContent = formatNumber(state.settings.dailyCalorieTarget || 3000, 0);
    }

    function scheduleSettingsSave() {
        if (state.isRendering) return;
        state.settings = normalizeSettings({
            dailyCalorieTarget: nullableNumber(els.dailyCalorieTarget.value) || DEFAULT_SETTINGS.dailyCalorieTarget,
            proteinTarget: nullableNumber(els.proteinTarget.value),
            carbsTarget: nullableNumber(els.carbsTarget.value),
            fatTarget: nullableNumber(els.fatTarget.value),
            units: 'metric'
        });
        renderTodayMetrics(getCurrentFormOrStateDay());
        debounce('settings', saveSettings, els.settingsSaveState);
    }

    async function saveSettings() {
        if (!state.user) return;
        const clean = normalizeSettings({
            dailyCalorieTarget: nullableNumber(els.dailyCalorieTarget.value) || DEFAULT_SETTINGS.dailyCalorieTarget,
            proteinTarget: nullableNumber(els.proteinTarget.value),
            carbsTarget: nullableNumber(els.carbsTarget.value),
            fatTarget: nullableNumber(els.fatTarget.value),
            units: 'metric'
        });
        await getUserRef('calorieSettings').doc('main').set({
            ...clean,
            updatedAt: serverTimestamp()
        }, { merge: true });
        state.settings = clean;
        els.settingsSaveState.textContent = '已保存';
        renderNavAndTopbar();
    }

    function renderNavAndTopbar() {
        const day = getCurrentFormOrStateDay();
        const totals = getDayTotals(day);
        const target = Number(state.settings.dailyCalorieTarget || DEFAULT_SETTINGS.dailyCalorieTarget);
        els.topDateLabel.textContent = formatDateLabel(state.selectedDateId);
        els.topCalorieStatus.textContent = `${formatNumber(totals.kcal.mid, 0)} / ${formatNumber(target, 0)} kcal`;
        els.todayNavBadge.textContent = `${formatNumber(totals.kcal.mid, 0)} kcal`;
        els.settingsNavBadge.textContent = formatNumber(target, 0);
    }

    function renderLegend(items) {
        return `
            <div class="chart-legend">
                ${items.map(([className, label]) => `
                    <span class="legend-item">
                        <span class="legend-swatch ${escapeAttr(className)}"></span>
                        ${escapeHtml(label)}
                    </span>
                `).join('')}
            </div>
        `;
    }

    function renderLineChart(rows, series) {
        const width = 900;
        const height = 260;
        const pad = { left: 42, right: 18, top: 16, bottom: 34 };
        const innerW = width - pad.left - pad.right;
        const innerH = height - pad.top - pad.bottom;
        const xFor = (index) => pad.left + (rows.length <= 1 ? innerW / 2 : (index / (rows.length - 1)) * innerW);
        const yFor = (value, max) => pad.top + innerH - ((max ? value / max : 0) * innerH);
        const maxByKey = {};

        series.forEach((item) => {
            const maxValue = Math.max(...rows.map((row) => Number(row.totals?.[item.key]?.mid || 0)));
            maxByKey[item.key] = maxValue > 0 ? maxValue * 1.12 : 1;
        });

        const lines = series.map((item) => {
            const points = rows
                .map((row, index) => {
                    const value = Number(row.totals?.[item.key]?.mid || 0);
                    if (!value) return null;
                    return `${xFor(index).toFixed(1)},${yFor(value, maxByKey[item.key]).toFixed(1)}`;
                })
                .filter(Boolean)
                .join(' ');
            if (!points) return '';
            return `<polyline class="${escapeAttr(item.className)}" points="${points}" fill="none" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></polyline>`;
        }).join('');

        const dots = series.map((item) => rows.map((row, index) => {
            const value = Number(row.totals?.[item.key]?.mid || 0);
            if (!value) return '';
            return `<circle class="${escapeAttr(item.className)}" cx="${xFor(index).toFixed(1)}" cy="${yFor(value, maxByKey[item.key]).toFixed(1)}" r="3.2"></circle>`;
        }).join('')).join('');

        const labelStep = rows.length > 30 ? 7 : rows.length > 14 ? 4 : 2;
        const dateLabels = rows.map((row, index) => {
            if (index !== 0 && index !== rows.length - 1 && index % labelStep !== 0) return '';
            return `<text class="chart-label" x="${xFor(index).toFixed(1)}" y="${height - 8}" text-anchor="middle">${escapeHtml(shortDate(row.dateId))}</text>`;
        }).join('');

        return `
            <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img">
                <line class="chart-axis" x1="${pad.left}" y1="${pad.top + innerH}" x2="${width - pad.right}" y2="${pad.top + innerH}"></line>
                ${[0.25, 0.5, 0.75, 1].map((ratio) => {
                    const y = pad.top + innerH - innerH * ratio;
                    return `<line class="chart-grid-line" x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(1)}"></line>`;
                }).join('')}
                ${lines}
                ${dots}
                ${dateLabels}
            </svg>
        `;
    }

    function getCurrentFormOrStateDay() {
        if (!els.mealList.children.length || state.isRendering) return getCurrentDay();
        const meals = Array.from(els.mealList.querySelectorAll('[data-meal-id]')).map((card, index) => ({
            id: card.getAttribute('data-meal-id'),
            name: card.querySelector('[data-meal-name]')?.value.trim() || `餐次 ${index + 1}`,
            order: index
        }));
        const ingredients = [];
        Array.from(els.mealList.querySelectorAll('[data-ingredient-id]')).forEach((row, index) => {
            const meal = row.closest('[data-meal-id]');
            const labelPer100g = {};
            row.querySelectorAll('[data-label-field]').forEach((input) => {
                const value = nullableNumber(input.value);
                if (value !== null) labelPer100g[input.getAttribute('data-label-field')] = value;
            });
            ingredients.push({
                id: row.getAttribute('data-ingredient-id'),
                mealId: meal?.getAttribute('data-meal-id') || meals[0]?.id || makeId('meal'),
                name: row.querySelector('[data-ingredient-field="name"]')?.value.trim() || '',
                amount: nullableNumber(row.querySelector('[data-ingredient-field="amount"]')?.value),
                unit: row.querySelector('[data-ingredient-field="unit"]')?.value || 'g',
                state: row.querySelector('[data-ingredient-field="state"]')?.value || 'unknown',
                note: row.querySelector('[data-ingredient-field="note"]')?.value.trim() || '',
                labelPer100g,
                order: index
            });
        });

        return normalizeDayForSave({
            ...getCurrentDay(),
            dateId: state.selectedDateId,
            meals,
            ingredients,
            note: els.dayNote.value
        });
    }

    function getCurrentDay() {
        const stored = getDayByDate(state.selectedDateId);
        return stored ? normalizeDayForSave(stored) : getDefaultDay(state.selectedDateId);
    }

    function getDayByDate(dateId) {
        return state.days.find((day) => (day.dateId || day.id) === dateId) || null;
    }

    function upsertLocalDay(day) {
        const existing = getDayByDate(day.dateId);
        const hasEstimate = Object.prototype.hasOwnProperty.call(day, 'aiEstimate');
        const nextDay = {
            ...(existing || {}),
            ...day,
            id: day.dateId,
            aiEstimate: hasEstimate ? day.aiEstimate || null : existing?.aiEstimate || null
        };
        state.days = [
            ...state.days.filter((item) => (item.dateId || item.id) !== day.dateId),
            nextDay
        ];
    }

    function getDefaultDay(dateId) {
        const meals = DEFAULT_MEAL_DEFS.map((meal, index) => ({
            id: meal.id,
            name: meal.name,
            order: index
        }));
        return normalizeDayForSave({
            id: dateId,
            dateId,
            meals,
            ingredients: [],
            note: '',
            aiEstimate: null
        });
    }

    function getEmptyDay(dateId) {
        return {
            id: dateId,
            dateId,
            meals: [],
            ingredients: [],
            note: '',
            aiEstimate: null,
            inputHash: makeDayInputHash({ meals: [], ingredients: [], note: '' })
        };
    }

    function normalizeDayForSave(day) {
        const meals = ensureDefaultMeals(normalizeMeals(day.meals));
        const ingredients = normalizeIngredients(day.ingredients).map((item, index) => ({
            ...item,
            mealId: meals.some((meal) => meal.id === item.mealId) ? item.mealId : meals[0]?.id || '',
            order: Number.isFinite(Number(item.order)) ? Number(item.order) : index
        }));
        const clean = {
            ...day,
            dateId: normalizeDateId(day.dateId || day.id || state.selectedDateId),
            meals,
            ingredients,
            note: String(day.note || '')
        };
        clean.inputHash = makeDayInputHash(clean);
        return clean;
    }

    function normalizeMeals(meals) {
        return (Array.isArray(meals) ? meals : [])
            .map((meal, index) => ({
                id: cleanId(meal?.id || makeId('meal')),
                name: limitString(meal?.name || `餐次 ${index + 1}`, 40),
                order: Number.isFinite(Number(meal?.order)) ? Number(meal.order) : index
            }))
            .sort((a, b) => a.order - b.order);
    }

    function ensureDefaultMeals(meals) {
        if (!meals.length) return [];
        const onlyDefaultNames = meals.every((meal) => DEFAULT_MEALS.includes(meal.name));
        if (!onlyDefaultNames) return meals;

        const existingByName = new Map();
        meals.forEach((meal) => {
            if (!existingByName.has(meal.name)) {
                existingByName.set(meal.name, meal);
            }
        });

        return DEFAULT_MEAL_DEFS.map((defaultMeal, index) => {
            const existing = existingByName.get(defaultMeal.name);
            return {
                id: existing?.id || defaultMeal.id,
                name: defaultMeal.name,
                order: index
            };
        });
    }

    function normalizeIngredients(ingredients) {
        return (Array.isArray(ingredients) ? ingredients : [])
            .map((item, index) => ({
                id: cleanId(item?.id || makeId('ing')),
                mealId: cleanId(item?.mealId || ''),
                name: limitString(item?.name || '', 80),
                amount: nullableNumber(item?.amount),
                unit: UNIT_OPTIONS.includes(item?.unit) ? item.unit : 'g',
                state: FOOD_STATES.some((option) => option.value === item?.state) ? item.state : 'unknown',
                note: limitString(item?.note || '', 160),
                labelPer100g: normalizeLabel(item?.labelPer100g),
                order: Number.isFinite(Number(item?.order)) ? Number(item.order) : index
            }))
            .sort((a, b) => a.order - b.order);
    }

    function normalizeLabel(label = {}) {
        const result = {};
        ['kcal', 'protein', 'carbs', 'fat'].forEach((key) => {
            const value = nullableNumber(label?.[key]);
            if (value !== null) result[key] = value;
        });
        return result;
    }

    function getMealsForRender(day) {
        const meals = ensureDefaultMeals(normalizeMeals(day.meals));
        if (meals.length) return meals;
        if (normalizeIngredients(day.ingredients).length) {
            return [{ id: 'meal_main', name: '记录', order: 0 }];
        }
        return [];
    }

    function getDayTotals(day) {
        const estimate = day?.aiEstimate;
        if (estimate?.totals && isEstimateFresh(day)) {
            return normalizeTotals(estimate.totals);
        }
        const totals = makeEmptyTotals();
        normalizeIngredients(day?.ingredients).forEach((item) => {
            addTotals(totals, getDeterministicIngredientTotals(item));
        });
        return totals;
    }

    function sumIngredientTotals(day, ingredients) {
        const estimateMap = isEstimateFresh(day) ? getIngredientEstimateMap(day) : new Map();
        const totals = makeEmptyTotals();
        ingredients.forEach((item) => {
            const estimate = estimateMap.get(item.id);
            if (estimate) {
                addTotals(totals, normalizeTotals(estimate));
                return;
            }
            addTotals(totals, getDeterministicIngredientTotals(item));
        });
        return totals;
    }

    function getIngredientEstimateMap(day) {
        const map = new Map();
        const estimates = Array.isArray(day?.aiEstimate?.ingredientEstimates) ? day.aiEstimate.ingredientEstimates : [];
        estimates.forEach((item) => {
            if (!item?.ingredientId) return;
            map.set(item.ingredientId, {
                kcal: item.kcal,
                protein: item.protein,
                carbs: item.carbs,
                fat: item.fat
            });
        });
        return map;
    }

    function getDeterministicIngredientTotals(item) {
        const label = item.labelPer100g || {};
        const amount = Number(item.amount || 0);
        const canScale = amount > 0 && ['g', 'ml'].includes(item.unit);
        const hasAllLabel = ['kcal', 'protein', 'carbs', 'fat'].every((key) => Number.isFinite(Number(label[key])));
        if (!canScale || !hasAllLabel) return makeEmptyTotals();
        const scale = amount / 100;
        return {
            kcal: makeExactRange(Number(label.kcal) * scale),
            protein: makeExactRange(Number(label.protein) * scale),
            carbs: makeExactRange(Number(label.carbs) * scale),
            fat: makeExactRange(Number(label.fat) * scale)
        };
    }

    function normalizeTotals(raw = {}) {
        return {
            kcal: normalizeRange(raw.kcal),
            protein: normalizeRange(raw.protein),
            carbs: normalizeRange(raw.carbs),
            fat: normalizeRange(raw.fat),
            weight: normalizeRange(raw.weight),
            bodyfat: normalizeRange(raw.bodyfat)
        };
    }

    function normalizeRange(raw) {
        if (typeof raw === 'number') return makeExactRange(raw);
        const low = Math.max(0, Number(raw?.low || 0));
        const mid = Math.max(0, Number(raw?.mid || 0));
        const high = Math.max(0, Number(raw?.high || 0));
        const values = [low, mid, high].sort((a, b) => a - b);
        return { low: values[0], mid: values[1], high: values[2] };
    }

    function makeExactRange(value) {
        const clean = Math.max(0, Number(value || 0));
        return { low: clean, mid: clean, high: clean };
    }

    function makeEmptyTotals() {
        return {
            kcal: makeExactRange(0),
            protein: makeExactRange(0),
            carbs: makeExactRange(0),
            fat: makeExactRange(0)
        };
    }

    function addTotals(target, addition) {
        ['kcal', 'protein', 'carbs', 'fat'].forEach((key) => {
            const range = normalizeRange(addition?.[key]);
            target[key].low += range.low;
            target[key].mid += range.mid;
            target[key].high += range.high;
        });
    }

    function isEstimateFresh(day) {
        if (!day?.aiEstimate?.inputHash || !day?.inputHash) return false;
        return day.aiEstimate.inputHash === day.inputHash;
    }

    function isMeaningfulDay(day) {
        if (String(day.note || '').trim()) return true;
        const meals = normalizeMeals(day.meals);
        const mealNames = meals.map((meal) => meal.name);
        const hasCustomMeal = meals.length !== DEFAULT_MEALS.length || mealNames.some((name, index) => name !== DEFAULT_MEALS[index]);
        if (hasCustomMeal) return true;
        return normalizeIngredients(day.ingredients).some((item) => (
            item.name || item.amount || item.note || Object.keys(item.labelPer100g || {}).length
        ));
    }

    function makeDayInputHash(day) {
        const meals = normalizeMeals(day.meals).map((meal) => ({
            id: meal.id,
            name: meal.name,
            order: meal.order
        }));
        const ingredients = normalizeIngredients(day.ingredients).map((item) => ({
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
            note: String(day.note || '')
        }));
    }

    function simpleHash(value) {
        let hash = 5381;
        for (let i = 0; i < value.length; i += 1) {
            hash = ((hash << 5) + hash) ^ value.charCodeAt(i);
        }
        return `h${(hash >>> 0).toString(36)}`;
    }

    function normalizeSettings(data) {
        return {
            dailyCalorieTarget: clampNumber(data?.dailyCalorieTarget, 1000, 7000, DEFAULT_SETTINGS.dailyCalorieTarget),
            proteinTarget: nullableNumber(data?.proteinTarget),
            carbsTarget: nullableNumber(data?.carbsTarget),
            fatTarget: nullableNumber(data?.fatTarget),
            units: 'metric'
        };
    }

    function getBodyLogByDate(dateId) {
        return state.bodyLogs.find((log) => (log.dateId || log.id) === dateId) || null;
    }

    function upsertLocalBodyLog(log) {
        state.bodyLogs = [
            ...state.bodyLogs.filter((item) => (item.dateId || item.id) !== log.dateId),
            { ...log, id: log.dateId }
        ];
    }

    function getEmptyBodyLog(dateId) {
        return { id: dateId, dateId, weightKg: null, bodyFatPct: null, note: '' };
    }

    function getLatestBodyLog() {
        return sortByDateDesc(state.bodyLogs.filter((log) => log.weightKg || log.bodyFatPct))[0] || null;
    }

    function getRangeDateIds(endDateId, count) {
        const end = parseDateId(endDateId);
        const result = [];
        for (let i = count - 1; i >= 0; i -= 1) {
            const next = new Date(end);
            next.setDate(end.getDate() - i);
            result.push(getDateParts(next).id);
        }
        return result;
    }

    function sortByDateDesc(items) {
        return [...items].sort((a, b) => String(b.dateId || b.id || '').localeCompare(String(a.dateId || a.id || '')));
    }

    function debounce(key, fn, saveEl) {
        clearTimeout(state.timers[key]);
        if (saveEl) saveEl.textContent = '保存中...';
        state.timers[key] = setTimeout(async () => {
            try {
                await fn();
            } catch (error) {
                console.error(`Failed to save ${key}`, error);
                if (saveEl) saveEl.textContent = '保存失败';
            }
        }, 520);
    }

    function clearTimers() {
        Object.values(state.timers).forEach((timer) => clearTimeout(timer));
        state.timers.day = null;
        state.timers.body = null;
        state.timers.settings = null;
    }

    function getUserRef(collectionName) {
        return state.db.collection('users').doc(state.user.uid).collection(collectionName);
    }

    function serverTimestamp() {
        return firebase.firestore.FieldValue.serverTimestamp();
    }

    async function signInWithEmail() {
        const email = els.authEmail.value.trim();
        const password = els.authPassword.value;
        if (!email || !password) {
            showAuthMessage('请填写邮箱和密码。');
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
        if (!email || password.length < 6) {
            showAuthMessage('邮箱账号需要填写邮箱和至少 6 位密码。');
            return;
        }
        try {
            const credential = await state.auth.createUserWithEmailAndPassword(email, password);
            await credential.user.sendEmailVerification();
            await state.auth.signOut();
            showAuthMessage('验证邮件已发送，请验证后再登录。');
        } catch (error) {
            showAuthMessage(normalizeAuthError(error));
        }
    }

    async function resetPassword() {
        const email = els.authEmail.value.trim();
        if (!email) {
            showAuthMessage('先填写邮箱。');
            return;
        }
        try {
            await state.auth.sendPasswordResetEmail(email);
            showAuthMessage('重置邮件已发送。');
        } catch (error) {
            showAuthMessage(normalizeAuthError(error));
        }
    }

    function needsEmailVerification(user) {
        const providers = user.providerData.map((provider) => provider.providerId);
        return providers.includes('password') && !user.emailVerified;
    }

    function showAuthMessage(message) {
        els.authMessage.textContent = message || '';
    }

    function showAppDialog(options = {}) {
        if (state.dialogResolve) closeAppDialog(false);
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

    function confirmDanger(title, message) {
        return showAppDialog({
            kicker: 'Confirm',
            title,
            message,
            confirmText: '删除',
            cancelText: '取消',
            danger: true
        });
    }

    function normalizeAuthError(error) {
        const code = error?.code || '';
        if (code.includes('user-not-found') || code.includes('wrong-password') || code.includes('invalid-credential')) {
            return '邮箱或密码不正确。';
        }
        if (code.includes('email-already-in-use')) return '这个邮箱已经注册。';
        if (code.includes('weak-password')) return '密码至少需要 6 位。';
        if (code.includes('popup-closed-by-user')) return '登录窗口已关闭。';
        return error?.message || '登录失败。';
    }

    function normalizeFunctionError(error) {
        const code = String(error?.code || '').replace(/^functions\//, '');
        if (code === 'unauthenticated') return '请先登录后再使用 AI 功能。';
        if (code === 'not-found') return error.message || '找不到这一天的记录，请先保存后再估算。';
        if (code === 'failed-precondition') return error.message || '当前记录还不能估算。';
        if (code === 'invalid-argument') return error.message || '输入内容不完整。';
        if (code === 'resource-exhausted') return error.message || 'AI 输出被截断，请减少当天食物条目后重试。';
        if (code === 'unavailable') return 'AI 服务暂时不可用，请稍后再试。';
        if (code === 'internal') {
            return error.message && error.message !== 'internal'
                ? error.message
                : '热量估算后端执行失败，请稍后重试或查看函数日志。';
        }
        return error?.message || '请求失败。';
    }

    function getDateParts(date) {
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        return {
            id: `${y}-${m}-${d}`,
            label: `${m}/${d}`
        };
    }

    function normalizeDateId(value) {
        const raw = String(value || '').trim();
        if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
        return getDateParts(new Date()).id;
    }

    function parseDateId(value) {
        const clean = normalizeDateId(value);
        const [year, month, day] = clean.split('-').map(Number);
        return new Date(year, month - 1, day);
    }

    function formatDateLabel(dateId) {
        const date = parseDateId(dateId);
        const today = getDateParts(new Date()).id;
        if (dateId === today) return '今天';
        return `${date.getMonth() + 1}/${date.getDate()}`;
    }

    function shortDate(dateId) {
        const date = parseDateId(dateId);
        return `${date.getMonth() + 1}/${date.getDate()}`;
    }

    function formatNumber(value, digits = 0) {
        const number = Number(value || 0);
        return number.toLocaleString('zh-CN', {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        });
    }

    function formatRange(range, unit) {
        const clean = normalizeRange(range);
        const digits = unit === 'kcal' || unit === '' ? 0 : 1;
        const suffix = unit ? ` ${unit}` : '';
        if (Math.abs(clean.low - clean.high) < 0.05) return `${formatNumber(clean.mid, digits)}${suffix}`;
        return `${formatNumber(clean.low, digits)}-${formatNumber(clean.high, digits)}${suffix}`;
    }

    function formatMealTotal(totals, itemCount = 0) {
        const kcal = Math.round(totals.kcal.mid || 0);
        return `${formatNumber(kcal, 0)} kcal · ${formatNumber(itemCount, 0)} 项`;
    }

    function formatMacroTarget(key, current) {
        const targetKey = `${key}Target`;
        const target = Number(state.settings[targetKey] || 0);
        if (!target) return '目标未设';
        const delta = target - Number(current || 0);
        return delta > 0 ? `差 ${formatNumber(delta, 1)}g` : `超 ${formatNumber(Math.abs(delta), 1)}g`;
    }

    function nullableNumber(value) {
        if (value === null || value === undefined || value === '') return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function clampNumber(value, min, max, fallback) {
        const number = nullableNumber(value);
        if (number === null) return fallback;
        return Math.min(max, Math.max(min, number));
    }

    function numberToInput(value) {
        if (value === null || value === undefined || value === '') return '';
        const number = Number(value);
        return Number.isFinite(number) ? String(number) : '';
    }

    function cleanId(value) {
        return String(value || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 80) || makeId('id');
    }

    function makeId(prefix) {
        return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    }

    function limitString(value, maxLength) {
        return String(value || '').trim().slice(0, maxLength);
    }

    function escapeHtml(value) {
        return String(value || '').replace(/[&<>"']/g, (char) => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[char]));
    }

    function escapeAttr(value) {
        return escapeHtml(value);
    }

    function isFocusedWithin(elements) {
        return elements.some((element) => element && (element === document.activeElement || element.contains(document.activeElement)));
    }

    function isMealExpanded(mealId) {
        return getExpandedMealIds().has(mealId);
    }

    function setMealExpanded(mealId, expanded) {
        if (!mealId) return;
        const expandedMealIds = getExpandedMealIds();
        if (expanded) {
            expandedMealIds.add(mealId);
        } else {
            expandedMealIds.delete(mealId);
        }
        setLocalValue(getMealExpansionStorageKey(), JSON.stringify([...expandedMealIds]));
    }

    function getExpandedMealIds() {
        const raw = getLocalValue(getMealExpansionStorageKey(), '[]');
        try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                return new Set(parsed
                    .map((value) => String(value || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 80))
                    .filter(Boolean));
            }
        } catch (error) {
            // Fall through to the default collapsed state.
        }
        return new Set();
    }

    function getMealExpansionStorageKey(dateId = state.selectedDateId) {
        return `calorieExpandedMeals:${normalizeDateId(dateId)}`;
    }

    function getLocalKey(key, uid = state.user?.uid) {
        return uid ? `${key}:${uid}` : key;
    }

    function getLocalValue(key, fallback, uid = state.user?.uid) {
        try {
            const value = localStorage.getItem(getLocalKey(key, uid));
            return value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    function setLocalValue(key, value, uid = state.user?.uid) {
        try {
            localStorage.setItem(getLocalKey(key, uid), String(value));
        } catch (error) {
            // Ignore storage failures in private browsing.
        }
    }
});
