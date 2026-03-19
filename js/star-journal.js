document.addEventListener('DOMContentLoaded', () => {
    const workspaceSwitch = document.getElementById('workspaceSwitch');
    const workspaceViews = {
        'command-center': document.getElementById('workspace-command-center'),
        'star-journal': document.getElementById('workspace-star-journal')
    };
    const starJournalStatus = document.getElementById('starJournalStatus');
    const sessionListEl = document.getElementById('starSessionList');
    const sessionDetailEl = document.getElementById('starSessionDetail');
    const membersEl = document.getElementById('journalMembers');
    const sameLocationHistoryEl = document.getElementById('sameLocationHistory');
    const tagFilterEl = document.getElementById('starTagFilter');
    const locationFilterEl = document.getElementById('starLocationFilter');
    const searchInputEl = document.getElementById('starSessionSearch');
    const inviteEmailEl = document.getElementById('partnerInviteEmail');
    const invitePartnerBtn = document.getElementById('invitePartnerBtn');
    const newSessionBtn = document.getElementById('newSessionBtn');
    const starJournalModal = document.getElementById('starJournalModal');
    const photoEditorModal = document.getElementById('photoEditorModal');

    const SOFT_SIZE_BYTES = 20 * 1024 * 1024;
    const HARD_SIZE_BYTES = 120 * 1024 * 1024;
    const HARD_MAX_DIMENSION = 14000;
    const HARD_MAX_PIXELS = 90000000;
    const MAX_PHOTOS_PER_SESSION = 10;
    const DISPLAY_EDGE = 2560;
    const THUMB_EDGE = 480;
    const DISPLAY_QUALITY = 0.82;
    const THUMB_QUALITY = 0.72;

    const state = {
        auth: null,
        db: null,
        storage: null,
        currentUser: null,
        currentJournalId: null,
        currentJournal: null,
        pendingInvites: [],
        sessions: [],
        activeSessionId: null,
        activePhotos: [],
        boundPhotoSessionId: null,
        syncingPhotoMeta: false,
        filters: {
            search: '',
            location: 'all',
            tag: 'all'
        },
        workspace: localStorage.getItem('adminWorkspace') || 'command-center',
        unsubs: {
            journal: null,
            sessions: null,
            invites: null,
            photos: null
        },
        editor: {
            photo: null,
            image: null,
            annotations: [],
            selectedTool: 'move',
            selectedId: null,
            color: '#00ff9d',
            lineWidth: 4,
            fontSize: 26,
            draft: null,
            isPointerDown: false,
            dragOrigin: null,
            history: [],
            historyIndex: -1
        }
    };

    initWorkspaceSwitch();
    initStaticActions();
    initModalShells();
    setWorkspace(state.workspace);
    initFirebase();

    function initWorkspaceSwitch() {
        if (!workspaceSwitch) return;
        workspaceSwitch.addEventListener('click', (event) => {
            const tab = event.target.closest('.workspace-tab');
            if (!tab) return;
            const workspace = tab.getAttribute('data-workspace');
            setWorkspace(workspace);
        });
    }

    function initStaticActions() {
        if (newSessionBtn) {
            newSessionBtn.addEventListener('click', () => {
                if (!state.currentUser || !state.currentJournalId) return;
                openSessionModal();
            });
        }

        if (invitePartnerBtn) {
            invitePartnerBtn.addEventListener('click', () => invitePartner());
        }

        if (searchInputEl) {
            searchInputEl.addEventListener('input', (event) => {
                state.filters.search = event.target.value.trim().toLowerCase();
                applyFiltersAndRender();
            });
        }

        if (locationFilterEl) {
            locationFilterEl.addEventListener('change', (event) => {
                state.filters.location = event.target.value;
                applyFiltersAndRender();
            });
        }

        if (tagFilterEl) {
            tagFilterEl.addEventListener('click', (event) => {
                const btn = event.target.closest('[data-tag-filter]');
                if (!btn) return;
                state.filters.tag = btn.getAttribute('data-tag-filter');
                renderTagFilter();
                applyFiltersAndRender();
            });
        }

        if (sessionListEl) {
            sessionListEl.addEventListener('click', (event) => {
                const card = event.target.closest('[data-session-id]');
                if (!card) return;
                state.activeSessionId = card.getAttribute('data-session-id');
                renderSessionList();
                bindActivePhotos();
                renderSessionDetail();
            });
        }

        if (sameLocationHistoryEl) {
            sameLocationHistoryEl.addEventListener('click', (event) => {
                const item = event.target.closest('[data-history-session-id]');
                if (!item) return;
                state.activeSessionId = item.getAttribute('data-history-session-id');
                renderSessionList();
                bindActivePhotos();
                renderSessionDetail();
            });
        }

        if (sessionDetailEl) {
            sessionDetailEl.addEventListener('click', (event) => {
                const actionNode = event.target.closest('[data-star-action]');
                if (!actionNode) return;

                const action = actionNode.getAttribute('data-star-action');
                const session = getActiveSession();
                const photoId = actionNode.getAttribute('data-photo-id');

                if (action === 'edit-session' && session) openSessionModal(session);
                if (action === 'delete-session' && session) deleteSession(session);
                if (action === 'upload-photos' && session) openPhotoUploadPicker(session);
                if (action === 'edit-photo' && photoId) openPhotoEditor(findPhoto(photoId));
                if (action === 'delete-photo' && photoId) deletePhoto(findPhoto(photoId));
                if (action === 'set-cover' && photoId && session) setCoverPhoto(session.id, photoId);
            });
        }
    }

    function initModalShells() {
        [starJournalModal, photoEditorModal].forEach((modal) => {
            if (!modal) return;
            modal.addEventListener('click', (event) => {
                if (event.target === modal || event.target.hasAttribute('data-close-modal')) {
                    closeModal(modal);
                }
            });
        });
    }

    function setWorkspace(workspace) {
        state.workspace = workspace in workspaceViews ? workspace : 'command-center';
        localStorage.setItem('adminWorkspace', state.workspace);
        Object.entries(workspaceViews).forEach(([key, view]) => {
            if (!view) return;
            view.classList.toggle('hidden', key !== state.workspace);
        });

        if (workspaceSwitch) {
            workspaceSwitch.querySelectorAll('.workspace-tab').forEach((tab) => {
                tab.classList.toggle('active', tab.getAttribute('data-workspace') === state.workspace);
            });
        }
    }

    async function initFirebase() {
        try {
            await waitForFirebaseReady();
            state.auth = firebase.auth();
            state.db = firebase.firestore();
            state.storage = firebase.storage();

            state.auth.onAuthStateChanged(async (user) => {
                try {
                    await handleAuthChange(user);
                } catch (error) {
                    console.error('Star journal auth flow failed:', error);
                    showStatus(error.message || '观星日记本初始化失败。', 'error');
                }
            });
        } catch (error) {
            console.error('Failed to initialize Firebase for Star Journal:', error);
            showStatus('观星日记本的 Firebase 初始化失败。', 'error');
        }
    }

    async function handleAuthChange(user) {
        state.currentUser = user;

        if (!user) {
            cleanupListeners();
            state.currentJournalId = null;
            state.currentJournal = null;
            state.pendingInvites = [];
            state.sessions = [];
            state.activeSessionId = null;
            state.activePhotos = [];
            state.boundPhotoSessionId = null;
            closeModal(starJournalModal);
            closeModal(photoEditorModal);
            renderMembers();
            renderFilters();
            renderSessionList();
            renderSessionDetail();
            renderSameLocationHistory();
            showStatus('登录后即可进入私密观星日记本。', 'info');
            return;
        }

        await state.db.collection('users').doc(user.uid).set({
            email: normalizeEmail(user.email || '')
        }, { merge: true });

        await acceptPendingInvite(user);
        await ensureActiveJournal(user);
        bindJournalListeners();
        showStatus('观星日记本已连接，图片会在浏览器中压缩后再上传。', 'info');
    }

    async function acceptPendingInvite(user) {
        const emailKey = normalizeEmail(user.email || '');
        if (!emailKey) return;

        const invitesSnap = await state.db.collection('starJournalInvites')
            .where('emailKey', '==', emailKey)
            .get();

        if (invitesSnap.empty) return;

        const pendingInvite = invitesSnap.docs
            .map((doc) => ({ id: doc.id, ...doc.data() }))
            .filter((invite) => invite.status === 'pending')
            .sort((a, b) => toMillis(b.createdAt) - toMillis(a.createdAt))[0];

        if (!pendingInvite) return;

        const journalRef = state.db.collection('starJournals').doc(pendingInvite.journalId);
        const journalSnap = await journalRef.get();
        if (!journalSnap.exists) return;

        const journalData = journalSnap.data() || {};
        const memberEmails = Array.isArray(journalData.memberEmails) ? journalData.memberEmails : [];
        const memberUids = Array.isArray(journalData.memberUids) ? journalData.memberUids : [];

        if (!memberEmails.includes(emailKey) && memberEmails.length >= 2 && !memberUids.includes(user.uid)) {
            showStatus('这本共享日记目前只支持两位成员。', 'warning');
            return;
        }

        const batch = state.db.batch();
        batch.set(journalRef, {
            memberEmails: firebase.firestore.FieldValue.arrayUnion(emailKey),
            memberUids: firebase.firestore.FieldValue.arrayUnion(user.uid),
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        }, { merge: true });
        batch.set(state.db.collection('starJournalInvites').doc(pendingInvite.id), {
            status: 'accepted',
            acceptedAt: firebase.firestore.FieldValue.serverTimestamp()
        }, { merge: true });
        batch.set(state.db.collection('users').doc(user.uid), {
            activeStarJournalId: pendingInvite.journalId
        }, { merge: true });
        await batch.commit();
    }

    async function ensureActiveJournal(user) {
        const userRef = state.db.collection('users').doc(user.uid);
        const userSnap = await userRef.get();
        let journalId = userSnap.exists ? userSnap.data().activeStarJournalId : null;

        if (journalId) {
            const activeSnap = await state.db.collection('starJournals').doc(journalId).get();
            if (activeSnap.exists) {
                state.currentJournalId = journalId;
                return;
            }
        }

        const journalSnap = await state.db.collection('starJournals')
            .where('memberUids', 'array-contains', user.uid)
            .get();

        if (!journalSnap.empty) {
            journalId = journalSnap.docs[0].id;
            state.currentJournalId = journalId;
            await userRef.set({ activeStarJournalId: journalId }, { merge: true });
            return;
        }

        const createdRef = state.db.collection('starJournals').doc();
        const displayName = deriveJournalName(user.email || 'pair');
        const batch = state.db.batch();
        batch.set(createdRef, {
            name: `${displayName} Sky Log`,
            ownerUid: user.uid,
            ownerEmail: normalizeEmail(user.email || ''),
            memberUids: [user.uid],
            memberEmails: [normalizeEmail(user.email || '')],
            createdAt: firebase.firestore.FieldValue.serverTimestamp(),
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        });
        batch.set(userRef, {
            activeStarJournalId: createdRef.id
        }, { merge: true });
        await batch.commit();
        state.currentJournalId = createdRef.id;
    }

    function bindJournalListeners() {
        cleanupListeners();
        if (!state.currentJournalId) return;

        const journalRef = getJournalRef();
        state.unsubs.journal = journalRef.onSnapshot((snap) => {
            if (!snap.exists) {
                showStatus('共享日记不存在或当前账号无权访问。', 'error');
                return;
            }
            state.currentJournal = { id: snap.id, ...snap.data() };
            renderMembers();
        });

        state.unsubs.invites = state.db.collection('starJournalInvites')
            .where('journalId', '==', state.currentJournalId)
            .onSnapshot((snap) => {
                state.pendingInvites = snap.docs
                    .map((doc) => ({ id: doc.id, ...doc.data() }))
                    .filter((invite) => invite.status === 'pending');
                renderMembers();
            });

        state.unsubs.sessions = journalRef.collection('sessions').onSnapshot((snap) => {
            state.sessions = snap.docs
                .map((doc) => ({ id: doc.id, ...doc.data() }))
                .sort(sortSessions);

            if (state.activeSessionId && !state.sessions.some((session) => session.id === state.activeSessionId)) {
                state.activeSessionId = state.sessions[0]?.id || null;
            }

            if (!state.activeSessionId && state.sessions.length) {
                state.activeSessionId = state.sessions[0].id;
            }

            renderFilters();
            applyFiltersAndRender();
        });
    }

    function applyFiltersAndRender() {
        const sessions = getFilteredSessions();
        if (!sessions.length) {
            state.activeSessionId = null;
            bindActivePhotos();
            renderSessionList();
            renderSessionDetail();
            return;
        }

        if (!sessions.some((session) => session.id === state.activeSessionId)) {
            state.activeSessionId = sessions[0].id;
            bindActivePhotos();
        }

        renderSessionList();
        renderSessionDetail();
    }

    function bindActivePhotos() {
        const activeSession = getActiveSession();
        if (!activeSession) {
            state.activePhotos = [];
            state.boundPhotoSessionId = null;
            if (state.unsubs.photos) {
                state.unsubs.photos();
                state.unsubs.photos = null;
            }
            renderSessionDetail();
            return;
        }

        if (state.boundPhotoSessionId === activeSession.id) return;

        if (state.unsubs.photos) {
            state.unsubs.photos();
            state.unsubs.photos = null;
        }

        state.boundPhotoSessionId = activeSession.id;
        state.activePhotos = [];
        state.unsubs.photos = getJournalRef().collection('sessions').doc(activeSession.id)
            .collection('photos')
            .onSnapshot(async (snap) => {
                state.activePhotos = snap.docs
                    .map((doc) => ({ id: doc.id, ...doc.data() }))
                    .sort((a, b) => (a.sortOrder || 0) - (b.sortOrder || 0));
                renderSessionDetail();
                await syncActiveSessionPhotoMeta();
            });
    }

    async function syncActiveSessionPhotoMeta() {
        const session = getActiveSession();
        if (!session || state.syncingPhotoMeta) return;

        const fallbackCoverId = state.activePhotos.some((photo) => photo.id === session.coverPhotoId)
            ? session.coverPhotoId
            : (state.activePhotos[0]?.id || null);
        const needsSync = session.photoCount !== state.activePhotos.length || session.coverPhotoId !== fallbackCoverId;

        if (!needsSync) return;

        state.syncingPhotoMeta = true;
        try {
            await getJournalRef().collection('sessions').doc(session.id).set({
                photoCount: state.activePhotos.length,
                coverPhotoId: fallbackCoverId,
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            }, { merge: true });
        } finally {
            state.syncingPhotoMeta = false;
        }
    }

    function renderSessionList() {
        if (!sessionListEl) return;
        const sessions = getFilteredSessions();

        if (!state.currentUser) {
            sessionListEl.innerHTML = `
                <div class="star-empty">
                    <p>需要先登录，观星日记本才会加载。</p>
                </div>
            `;
            return;
        }

        if (!sessions.length) {
            sessionListEl.innerHTML = `
                <div class="star-empty">
                    <p>还没有观星记录。先创建第一条，把你们的夜空记下来。</p>
                </div>
            `;
            return;
        }

        sessionListEl.innerHTML = sessions.map((session) => `
            <article class="star-session-card ${session.id === state.activeSessionId ? 'active' : ''}" data-session-id="${session.id}">
                <div class="star-session-date">${escapeHtml(session.sessionDate || '日期未知')}</div>
                <div class="star-session-title">${escapeHtml(session.title || '未命名记录')}</div>
                <div class="star-session-location">${escapeHtml(session.locationName || '未知地点')}</div>
                <div class="star-session-note">${escapeHtml(truncateText(session.note || session.skySummary || '这次观星还没有写备注。', 110))}</div>
                <div class="star-tag-row">
                    ${renderTagHtml(session.tags)}
                </div>
            </article>
        `).join('');
    }

    function renderSessionDetail() {
        if (!sessionDetailEl) return;
        const session = getActiveSession();

        if (!session) {
            sessionDetailEl.innerHTML = `
                <div class="star-empty star-empty-large">
                    <p>当前没有选中的记录。可以新建一条，或调整筛选条件。</p>
                </div>
            `;
            renderSameLocationHistory();
            return;
        }

        const coverPhoto = state.activePhotos.find((photo) => photo.id === session.coverPhotoId) || state.activePhotos[0] || null;
        const coverUrl = coverPhoto ? getPhotoPreviewUrl(coverPhoto) : '';

        sessionDetailEl.innerHTML = `
            <div class="star-detail-header">
                <div>
                    <div class="star-section-kicker">${escapeHtml(session.sessionDate || '日期未知')} // ${escapeHtml(session.locationName || '未知地点')}</div>
                    <h2 class="star-detail-title">${escapeHtml(session.title || '未命名记录')}</h2>
                    <div class="star-tag-row">${renderTagHtml(session.tags)}</div>
                </div>
                <div class="star-detail-actions">
                    <button class="star-action-btn" data-star-action="upload-photos">上传照片</button>
                    <button class="star-action-btn star-action-btn-muted" data-star-action="edit-session">编辑记录</button>
                    <button class="star-action-btn star-action-btn-danger" data-star-action="delete-session">删除记录</button>
                </div>
            </div>

            ${coverPhoto ? `
                <div class="star-cover-card">
                    <img src="${escapeAttribute(coverUrl)}" alt="Session cover" class="star-cover-image">
                </div>
            ` : ''}

            <div class="star-detail-grid">
                <div class="star-meta-card">
                    <div class="star-meta-label">日期 / 时间</div>
                    <div class="star-meta-value">${escapeHtml(buildTimeLabel(session))}</div>
                </div>
                <div class="star-meta-card">
                    <div class="star-meta-label">地点</div>
                    <div class="star-meta-value">${escapeHtml(session.locationName || '未知')}<br>${escapeHtml(session.locationNote || '还没有写地点备注。')}</div>
                </div>
                <div class="star-meta-card">
                    <div class="star-meta-label">天空概况</div>
                    <div class="star-meta-value">${escapeHtml(session.skySummary || '还没有记录当天的天空情况。')}</div>
                </div>
            </div>

            <div class="star-copy-block">${formatMultiline(session.note || '这一晚的详细观星感受还没有写下来。')}</div>

            <div class="star-gallery-header">
                <div class="star-gallery-title">照片区 // ${state.activePhotos.length} 张</div>
                <div class="star-meta-label">点击“编辑 / 查看”即可圈出星星、星座和路径。</div>
            </div>

            <div class="star-gallery-grid">
                ${renderPhotoCards(session)}
            </div>
        `;

        renderSameLocationHistory();
    }

    function renderPhotoCards(session) {
        if (!state.activePhotos.length) {
            return `
                <div class="star-empty">
                    <p>还没有上传照片。点击“上传照片”把这次的夜空放进来。</p>
                </div>
            `;
        }

        return state.activePhotos.map((photo) => `
            <article class="star-photo-card">
                <img src="${escapeAttribute(getPhotoPreviewUrl(photo))}" alt="Sky photo" class="star-photo-thumb" data-star-action="edit-photo" data-photo-id="${photo.id}">
                <div class="star-photo-body">
                    <div class="star-photo-caption">${escapeHtml(photo.caption || '这张照片还没有说明，打开编辑器后可以补充。')}</div>
                    <div class="star-photo-actions">
                        <button class="star-inline-btn" data-star-action="edit-photo" data-photo-id="${photo.id}">编辑 / 查看</button>
                        <button class="star-inline-btn ${session.coverPhotoId === photo.id ? 'is-cover' : ''}" data-star-action="set-cover" data-photo-id="${photo.id}">
                            ${session.coverPhotoId === photo.id ? '当前封面' : '设为封面'}
                        </button>
                        <button class="star-inline-btn" data-star-action="delete-photo" data-photo-id="${photo.id}">删除</button>
                    </div>
                </div>
            </article>
        `).join('');
    }

    function renderMembers() {
        if (!membersEl) return;
        if (!state.currentJournal) {
            membersEl.innerHTML = `
                <div class="star-empty compact">
                    <p>登录后这里会显示共享成员。</p>
                </div>
            `;
            return;
        }

        const ownerUid = state.currentJournal.ownerUid;
        const ownerEmail = normalizeEmail(state.currentJournal.ownerEmail || '');
        const memberEmails = Array.isArray(state.currentJournal.memberEmails) ? state.currentJournal.memberEmails : [];
        const memberRows = memberEmails.map((email) => {
            const isOwner = email === ownerEmail || (ownerUid && email === normalizeEmail(state.currentUser?.email || '') && state.currentUser?.uid === ownerUid);
            return `
                <div class="star-member-pill ${isOwner ? 'owner' : ''}">
                    <span>${escapeHtml(email)}</span>
                    <span>${isOwner ? '拥有者' : '成员'}</span>
                </div>
            `;
        }).join('');

        const inviteRows = state.pendingInvites.map((invite) => `
            <div class="star-member-pill">
                <span>${escapeHtml(invite.email)}</span>
                <span>待加入</span>
            </div>
        `).join('');

        membersEl.innerHTML = memberRows + inviteRows || `
            <div class="star-empty compact">
                <p>暂时没有成员信息。</p>
            </div>
        `;
    }

    function renderFilters() {
        renderLocationFilter();
        renderTagFilter();
    }

    function renderLocationFilter() {
        if (!locationFilterEl) return;
        const locations = [];
        const seen = new Set();

        state.sessions.forEach((session) => {
            if (!session.locationKey || seen.has(session.locationKey)) return;
            seen.add(session.locationKey);
            locations.push({
                key: session.locationKey,
                label: session.locationName || session.locationKey
            });
        });

        if (state.filters.location !== 'all' && !locations.some((location) => location.key === state.filters.location)) {
            state.filters.location = 'all';
        }

        locationFilterEl.innerHTML = `
            <option value="all">全部地点</option>
            ${locations.sort((a, b) => a.label.localeCompare(b.label)).map((location) => `
                <option value="${escapeAttribute(location.key)}" ${location.key === state.filters.location ? 'selected' : ''}>${escapeHtml(location.label)}</option>
            `).join('')}
        `;
    }

    function renderTagFilter() {
        if (!tagFilterEl) return;
        const tags = Array.from(new Set(state.sessions.flatMap((session) => Array.isArray(session.tags) ? session.tags : [])))
            .sort((a, b) => a.localeCompare(b));

        if (state.filters.tag !== 'all' && !tags.includes(state.filters.tag)) {
            state.filters.tag = 'all';
        }

        tagFilterEl.innerHTML = `
            <button class="star-tag-filter-btn ${state.filters.tag === 'all' ? 'active' : ''}" data-tag-filter="all">全部标签</button>
            ${tags.map((tag) => `
                <button class="star-tag-filter-btn ${state.filters.tag === tag ? 'active' : ''}" data-tag-filter="${escapeAttribute(tag)}">${escapeHtml(tag)}</button>
            `).join('')}
        `;
    }

    function renderSameLocationHistory() {
        if (!sameLocationHistoryEl) return;
        const session = getActiveSession();

        if (!session || !session.locationKey) {
            sameLocationHistoryEl.innerHTML = `
                <div class="star-empty compact">
                    <p>选择一条记录后，可按地点查看历史。</p>
                </div>
            `;
            return;
        }

        const relatedSessions = state.sessions
            .filter((item) => item.locationKey === session.locationKey)
            .sort(sortSessions);

        sameLocationHistoryEl.innerHTML = relatedSessions.map((item) => `
            <div class="star-history-item ${item.id === session.id ? 'active' : ''}" data-history-session-id="${item.id}">
                <div class="star-session-date">${escapeHtml(item.sessionDate || '日期未知')}</div>
                <div class="star-session-title">${escapeHtml(item.title || '未命名记录')}</div>
                <div class="star-session-note">${escapeHtml(truncateText(item.note || item.skySummary || '还没有备注。', 70))}</div>
            </div>
        `).join('');
    }

    function getFilteredSessions() {
        return state.sessions.filter((session) => {
            const searchBlob = [
                session.title || '',
                session.locationName || '',
                session.locationNote || '',
                session.note || '',
                session.skySummary || '',
                ...(Array.isArray(session.tags) ? session.tags : [])
            ].join(' ').toLowerCase();

            if (state.filters.search && !searchBlob.includes(state.filters.search)) return false;
            if (state.filters.location !== 'all' && session.locationKey !== state.filters.location) return false;
            if (state.filters.tag !== 'all' && !(Array.isArray(session.tags) && session.tags.includes(state.filters.tag))) return false;
            return true;
        }).sort(sortSessions);
    }

    function openSessionModal(session = null) {
        const isEdit = Boolean(session);
        starJournalModal.innerHTML = `
            <div class="star-modal">
                <div class="star-modal-header">
                    <div class="star-modal-title">${isEdit ? '编辑观星记录' : '新建观星记录'}</div>
                    <button class="star-modal-close" data-close-modal>&times;</button>
                </div>
                <div class="star-modal-body">
                    <div class="star-form-grid">
                        <label class="star-form-field">
                            <span class="star-form-label">标题 *</span>
                            <input type="text" id="sj-title" class="star-form-input" value="${escapeAttribute(session?.title || '')}" autocomplete="off">
                        </label>
                        <label class="star-form-field">
                            <span class="star-form-label">日期 *</span>
                            <input type="date" id="sj-date" class="star-form-input" value="${escapeAttribute(session?.sessionDate || '')}">
                        </label>
                        <label class="star-form-field">
                            <span class="star-form-label">开始时间</span>
                            <input type="time" id="sj-start-time" class="star-form-input" value="${escapeAttribute(session?.startTime || '')}">
                        </label>
                        <label class="star-form-field">
                            <span class="star-form-label">结束时间</span>
                            <input type="time" id="sj-end-time" class="star-form-input" value="${escapeAttribute(session?.endTime || '')}">
                        </label>
                        <label class="star-form-field star-form-grid-full">
                            <span class="star-form-label">地点 *</span>
                            <input type="text" id="sj-location" class="star-form-input" value="${escapeAttribute(session?.locationName || '')}" autocomplete="off">
                        </label>
                        <label class="star-form-field star-form-grid-full">
                            <span class="star-form-label">地点备注</span>
                            <input type="text" id="sj-location-note" class="star-form-input" value="${escapeAttribute(session?.locationNote || '')}" autocomplete="off">
                        </label>
                        <label class="star-form-field star-form-grid-full">
                            <span class="star-form-label">天空概况</span>
                            <input type="text" id="sj-sky-summary" class="star-form-input" value="${escapeAttribute(session?.skySummary || '')}" autocomplete="off">
                        </label>
                        <label class="star-form-field star-form-grid-full">
                            <span class="star-form-label">标签（用逗号分隔）</span>
                            <input type="text" id="sj-tags" class="star-form-input" value="${escapeAttribute((session?.tags || []).join(', '))}" autocomplete="off">
                        </label>
                        <label class="star-form-field star-form-grid-full">
                            <span class="star-form-label">详细备注</span>
                            <textarea id="sj-note" class="star-form-textarea">${escapeHtml(session?.note || '')}</textarea>
                        </label>
                    </div>
                    <div class="star-form-actions">
                        <div>
                            ${isEdit ? '<button id="sj-delete" class="star-action-btn star-action-btn-danger">删除记录</button>' : ''}
                        </div>
                        <div class="star-detail-actions">
                            <button class="star-action-btn star-action-btn-muted" data-close-modal>取消</button>
                            <button id="sj-save" class="star-action-btn">${isEdit ? '保存修改' : '创建记录'}</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        starJournalModal.classList.add('active');

        const saveBtn = document.getElementById('sj-save');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                try {
                    await saveSession(session);
                    closeModal(starJournalModal);
                } catch (error) {
                    console.error('Failed to save session:', error);
                    alert(error.message || '保存观星记录失败。');
                }
            });
        }

        const deleteBtn = document.getElementById('sj-delete');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', async () => {
                closeModal(starJournalModal);
                await deleteSession(session);
            });
        }
    }

    async function saveSession(existingSession = null) {
        const title = document.getElementById('sj-title')?.value.trim();
        const sessionDate = document.getElementById('sj-date')?.value.trim();
        const startTime = document.getElementById('sj-start-time')?.value.trim();
        const endTime = document.getElementById('sj-end-time')?.value.trim();
        const locationName = document.getElementById('sj-location')?.value.trim();
        const locationNote = document.getElementById('sj-location-note')?.value.trim();
        const skySummary = document.getElementById('sj-sky-summary')?.value.trim();
        const note = document.getElementById('sj-note')?.value.trim();
        const tags = parseTags(document.getElementById('sj-tags')?.value || '');

        if (!title) throw new Error('请填写记录标题。');
        if (!sessionDate) throw new Error('请选择日期。');
        if (!locationName) throw new Error('请填写地点。');
        if (startTime && endTime && startTime > endTime) throw new Error('结束时间不能早于开始时间。');

        const payload = {
            title,
            sessionDate,
            startTime,
            endTime,
            locationName,
            locationKey: normalizeLocation(locationName),
            locationNote,
            note,
            skySummary,
            tags,
            season: getSeason(sessionDate),
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        };

        if (existingSession) {
            await getJournalRef().collection('sessions').doc(existingSession.id).set(payload, { merge: true });
            showStatus(`已更新记录：${title}`, 'info');
            return;
        }

        const sessionRef = getJournalRef().collection('sessions').doc();
        await sessionRef.set({
            ...payload,
            photoCount: 0,
            coverPhotoId: null,
            createdByUid: state.currentUser.uid,
            createdAt: firebase.firestore.FieldValue.serverTimestamp()
        });
        state.activeSessionId = sessionRef.id;
        bindActivePhotos();
        showStatus(`已创建记录：${title}`, 'info');
    }

    async function deleteSession(session) {
        if (!session) return;
        if (!confirm(`确定删除“${session.title}”以及其中所有照片吗？`)) return;

        const photosSnap = await getJournalRef().collection('sessions').doc(session.id).collection('photos').get();
        const batch = state.db.batch();

        for (const photoDoc of photosSnap.docs) {
            const photo = photoDoc.data();
            await deleteStoragePath(photo.thumbPath);
            await deleteStoragePath(photo.displayPath);
            await deleteStoragePath(photo.annotatedPreviewPath);
            batch.delete(photoDoc.ref);
        }

        batch.delete(getJournalRef().collection('sessions').doc(session.id));
        await batch.commit();

        if (state.activeSessionId === session.id) {
            state.activeSessionId = null;
            state.activePhotos = [];
        }

        showStatus(`已删除记录：${session.title}`, 'warning');
    }

    async function invitePartner() {
        const email = normalizeEmail(inviteEmailEl?.value || '');
        if (!email) {
            alert('请先输入对方的邮箱。');
            return;
        }
        if (!state.currentJournal) {
            alert('共享日记还没准备好。');
            return;
        }
        if (normalizeEmail(state.currentUser?.email || '') === email) {
            alert('你已经在这本日记里了。');
            return;
        }

        const memberEmails = Array.isArray(state.currentJournal.memberEmails) ? state.currentJournal.memberEmails : [];
        const pendingEmails = state.pendingInvites.map((invite) => normalizeEmail(invite.email || ''));
        if (memberEmails.includes(email) || pendingEmails.includes(email)) {
            alert('这个邮箱已经有权限，或邀请仍在待处理。');
            return;
        }
        if (memberEmails.length >= 2) {
            alert('这个版本的共享日记只支持两位成员。');
            return;
        }

        const userSnap = await state.db.collection('users')
            .where('email', '==', email)
            .get();

        if (!userSnap.empty) {
            const partnerDoc = userSnap.docs[0];
            await state.db.runTransaction(async (transaction) => {
                const journalDoc = await transaction.get(getJournalRef());
                const journalData = journalDoc.data() || {};
                const liveMemberEmails = Array.isArray(journalData.memberEmails) ? journalData.memberEmails : [];
                if (liveMemberEmails.length >= 2 && !liveMemberEmails.includes(email)) {
                    throw new Error('这本日记已经有两位成员。');
                }
                transaction.set(getJournalRef(), {
                    memberEmails: firebase.firestore.FieldValue.arrayUnion(email),
                    memberUids: firebase.firestore.FieldValue.arrayUnion(partnerDoc.id),
                    updatedAt: firebase.firestore.FieldValue.serverTimestamp()
                }, { merge: true });
                transaction.set(partnerDoc.ref, {
                    activeStarJournalId: state.currentJournalId
                }, { merge: true });
            });
            if (inviteEmailEl) inviteEmailEl.value = '';
            showStatus(`已将 ${email} 直接加入共享日记。`, 'info');
            return;
        }

        await state.db.collection('starJournalInvites').add({
            journalId: state.currentJournalId,
            email,
            emailKey: email,
            invitedByUid: state.currentUser.uid,
            status: 'pending',
            createdAt: firebase.firestore.FieldValue.serverTimestamp(),
            acceptedAt: null
        });
        if (inviteEmailEl) inviteEmailEl.value = '';
        showStatus(`已向 ${email} 发出邀请，对方下次登录后会自动加入。`, 'info');
    }

    function openPhotoUploadPicker(session) {
        if (!session) return;

        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.multiple = true;
        input.onchange = async () => {
            if (!input.files || !input.files.length) return;

            const files = Array.from(input.files);
            if (state.activePhotos.length + files.length > MAX_PHOTOS_PER_SESSION) {
                alert(`每条观星记录最多支持 ${MAX_PHOTOS_PER_SESSION} 张照片。`);
                return;
            }

            const oversizeCount = files.filter((file) => file.size > SOFT_SIZE_BYTES).length;
            if (oversizeCount > 0) {
                const proceed = confirm(`有 ${oversizeCount} 张照片超过 20 MB。它们仍然可以上传，但会先在浏览器里压缩，过程可能更慢。要继续吗？`);
                if (!proceed) return;
            }

            try {
                showStatus('正在压缩并上传照片...', 'info');
                let sortBase = state.activePhotos.length;
                for (const file of files) {
                    await uploadPhotoFile(session, file, sortBase);
                    sortBase += 1;
                }
                showStatus('照片上传完成。', 'info');
            } catch (error) {
                console.error('Photo upload failed:', error);
                showStatus(error.message || '照片上传失败。', 'error');
            }
        };
        input.click();
    }

    async function uploadPhotoFile(session, file, sortOrder) {
        if (!file.type.startsWith('image/')) {
            throw new Error(`${file.name} 不是支持的图片格式。`);
        }
        if (file.size > HARD_SIZE_BYTES) {
            throw new Error(`${file.name} 太大，浏览器端无法安全处理。`);
        }

        const image = await loadImageFromFile(file);
        if (image.width > HARD_MAX_DIMENSION || image.height > HARD_MAX_DIMENSION || image.width * image.height > HARD_MAX_PIXELS) {
            throw new Error(`${file.name} 的像素尺寸过大，浏览器编辑不安全。`);
        }

        const displayBlob = await makeDerivedBlob(image, DISPLAY_EDGE, DISPLAY_QUALITY);
        const thumbBlob = await makeDerivedBlob(image, THUMB_EDGE, THUMB_QUALITY);
        const photoRef = getJournalRef().collection('sessions').doc(session.id).collection('photos').doc();
        const basePath = `star-journals/${state.currentJournalId}/sessions/${session.id}/photos/${photoRef.id}`;

        const [displayUpload, thumbUpload] = await Promise.all([
            putBlob(`${basePath}/display.jpg`, displayBlob),
            putBlob(`${basePath}/thumb.jpg`, thumbBlob)
        ]);

        await photoRef.set({
            caption: '',
            sortOrder,
            thumbPath: `${basePath}/thumb.jpg`,
            thumbUrl: thumbUpload.downloadURL,
            displayPath: `${basePath}/display.jpg`,
            displayUrl: displayUpload.downloadURL,
            annotatedPreviewPath: '',
            annotatedPreviewUrl: '',
            width: displayUpload.width,
            height: displayUpload.height,
            sizeBytes: file.size,
            hasAnnotations: false,
            annotations: [],
            createdAt: firebase.firestore.FieldValue.serverTimestamp(),
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        });

        const sessionRef = getJournalRef().collection('sessions').doc(session.id);
        const sessionSnap = await sessionRef.get();
        if (sessionSnap.exists && !sessionSnap.data().coverPhotoId) {
            await sessionRef.set({
                coverPhotoId: photoRef.id,
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            }, { merge: true });
        }
    }

    async function deletePhoto(photo) {
        const session = getActiveSession();
        if (!photo || !session) return;
        if (!confirm('确定删除这张照片以及它的全部标注吗？')) return;

        await deleteStoragePath(photo.thumbPath);
        await deleteStoragePath(photo.displayPath);
        await deleteStoragePath(photo.annotatedPreviewPath);
        await getJournalRef().collection('sessions').doc(session.id).collection('photos').doc(photo.id).delete();
        showStatus('照片已删除。', 'warning');
    }

    async function setCoverPhoto(sessionId, photoId) {
        await getJournalRef().collection('sessions').doc(sessionId).set({
            coverPhotoId: photoId,
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        }, { merge: true });
    }

    function openPhotoEditor(photo) {
        if (!photo) return;
        const editor = state.editor;
        editor.photo = photo;
        editor.image = null;
        editor.annotations = deepClone(photo.annotations || []);
        editor.selectedTool = 'move';
        editor.selectedId = null;
        editor.color = '#00ff9d';
        editor.lineWidth = 4;
        editor.fontSize = 26;
        editor.draft = null;
        editor.isPointerDown = false;
        editor.dragOrigin = null;
        editor.history = [];
        editor.historyIndex = -1;

        photoEditorModal.innerHTML = `
            <div class="star-modal star-photo-editor-modal">
                <div class="star-modal-header">
                    <div class="star-modal-title">照片标注编辑器</div>
                    <button class="star-modal-close" data-close-modal>&times;</button>
                </div>
                <div class="star-modal-body">
                    <div class="star-photo-editor-shell">
                        <div class="star-editor-stage">
                            <div class="star-editor-toolbar" id="starEditorToolbar">
                                ${['move', 'circle', 'arrow', 'text', 'delete', 'undo', 'redo', 'reset'].map((tool) => `
                                    <button class="star-tool-btn ${tool === 'move' ? 'active' : ''}" data-editor-tool="${tool}">${getToolLabel(tool)}</button>
                                `).join('')}
                            </div>
                            <div class="star-canvas-wrap">
                                <canvas id="starPhotoEditorCanvas"></canvas>
                            </div>
                            <div class="star-loading" id="starEditorLoading">正在载入展示图...</div>
                        </div>
                        <div class="star-editor-side">
                            <div class="star-side-title">标注信息</div>
                            <label class="star-form-field">
                                <span class="star-form-label">照片说明</span>
                                <textarea id="starPhotoCaption" class="star-form-textarea">${escapeHtml(photo.caption || '')}</textarea>
                            </label>
                            <label class="star-form-field">
                                <span class="star-form-label">颜色</span>
                                <input type="color" id="starEditorColor" class="star-form-input" value="#00ff9d">
                            </label>
                            <label class="star-form-field">
                                <span class="star-form-label">线条粗细</span>
                                <input type="range" id="starEditorLineWidth" min="2" max="14" value="4">
                            </label>
                            <label class="star-form-field">
                                <span class="star-form-label">文字大小</span>
                                <input type="range" id="starEditorFontSize" min="14" max="60" value="26">
                            </label>
                            <div class="star-editor-note">
                                点击“圆圈”圈出目标，用“箭头”指向目标，用“文字”写上像“猎户座”或“木星”这样的标记。
                            </div>
                            <div class="star-editor-actions">
                                <button id="starEditorClose" class="star-action-btn star-action-btn-muted">取消</button>
                                <button id="starEditorSave" class="star-action-btn">保存标注</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        photoEditorModal.classList.add('active');

        const canvas = document.getElementById('starPhotoEditorCanvas');
        const toolbar = document.getElementById('starEditorToolbar');
        const colorInput = document.getElementById('starEditorColor');
        const lineWidthInput = document.getElementById('starEditorLineWidth');
        const fontSizeInput = document.getElementById('starEditorFontSize');
        const captionInput = document.getElementById('starPhotoCaption');
        const loadingEl = document.getElementById('starEditorLoading');

        if (colorInput) {
            colorInput.addEventListener('input', (event) => {
                state.editor.color = event.target.value;
            });
        }
        if (lineWidthInput) {
            lineWidthInput.addEventListener('input', (event) => {
                state.editor.lineWidth = Number(event.target.value);
            });
        }
        if (fontSizeInput) {
            fontSizeInput.addEventListener('input', (event) => {
                state.editor.fontSize = Number(event.target.value);
            });
        }
        if (toolbar) {
            toolbar.addEventListener('click', (event) => {
                const btn = event.target.closest('[data-editor-tool]');
                if (!btn) return;
                const tool = btn.getAttribute('data-editor-tool');
                handleEditorToolSelection(tool);
            });
        }
        const closeBtn = document.getElementById('starEditorClose');
        if (closeBtn) closeBtn.addEventListener('click', () => closeModal(photoEditorModal));
        const saveBtn = document.getElementById('starEditorSave');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                try {
                    saveBtn.disabled = true;
                    loadingEl.textContent = '正在保存标注...';
                    await savePhotoAnnotations(captionInput?.value.trim() || '');
                    closeModal(photoEditorModal);
                    showStatus('标注已保存。', 'info');
                } catch (error) {
                    console.error('Annotation save failed:', error);
                    alert(error.message || '保存标注失败。');
                } finally {
                    saveBtn.disabled = false;
                    loadingEl.textContent = '';
                }
            });
        }

        loadImageForEditor(photo.displayUrl)
            .then((image) => {
                state.editor.image = image;
                canvas.width = image.width;
                canvas.height = image.height;
                pushEditorHistory();
                redrawEditorCanvas();
                loadingEl.textContent = '';
                bindCanvasInteractions(canvas);
            })
            .catch((error) => {
                console.error('Editor image load failed:', error);
                loadingEl.textContent = '载入图片失败，无法进入编辑。';
            });
    }

    function handleEditorToolSelection(tool) {
        if (tool === 'undo') {
            editorUndo();
            return;
        }
        if (tool === 'redo') {
            editorRedo();
            return;
        }
        if (tool === 'reset') {
            if (!confirm('确定清空这张照片上的全部标注吗？')) return;
            state.editor.annotations = [];
            state.editor.selectedId = null;
            pushEditorHistory();
            redrawEditorCanvas();
            return;
        }

        state.editor.selectedTool = tool;
        const toolbar = document.getElementById('starEditorToolbar');
        if (toolbar) {
            toolbar.querySelectorAll('[data-editor-tool]').forEach((btn) => {
                btn.classList.toggle('active', btn.getAttribute('data-editor-tool') === tool);
            });
        }
    }

    function bindCanvasInteractions(canvas) {
        if (!canvas) return;

        const onPointerDown = (event) => {
            if (!state.editor.image) return;
            const point = getCanvasPoint(canvas, event);
            const tool = state.editor.selectedTool;

            if (tool === 'text') {
                const text = prompt('请输入这个目标的标记文字：', '');
                if (text) {
                    state.editor.annotations.push(createBaseAnnotation('text', {
                        x: point.x,
                        y: point.y,
                        text: text.trim(),
                        color: state.editor.color,
                        fontSize: state.editor.fontSize,
                        w: 0,
                        h: 0,
                        lineWidth: state.editor.lineWidth
                    }));
                    pushEditorHistory();
                    redrawEditorCanvas();
                }
                return;
            }

            if (tool === 'delete') {
                const target = findAnnotationAtPoint(point);
                if (!target) return;
                state.editor.annotations = state.editor.annotations.filter((annotation) => annotation.id !== target.id);
                state.editor.selectedId = null;
                pushEditorHistory();
                redrawEditorCanvas();
                return;
            }

            if (tool === 'move') {
                const target = findAnnotationAtPoint(point);
                if (!target) {
                    state.editor.selectedId = null;
                    redrawEditorCanvas();
                    return;
                }
                state.editor.selectedId = target.id;
                state.editor.dragOrigin = {
                    point,
                    annotation: deepClone(target)
                };
                state.editor.isPointerDown = true;
                redrawEditorCanvas();
                return;
            }

            state.editor.isPointerDown = true;
            state.editor.draft = createBaseAnnotation(tool, {
                x: point.x,
                y: point.y,
                w: 0,
                h: 0,
                text: '',
                color: state.editor.color,
                lineWidth: state.editor.lineWidth,
                fontSize: state.editor.fontSize
            });
            redrawEditorCanvas();
        };

        const onPointerMove = (event) => {
            if (!state.editor.image || !state.editor.isPointerDown) return;
            const point = getCanvasPoint(canvas, event);
            const tool = state.editor.selectedTool;

            if (tool === 'move' && state.editor.dragOrigin && state.editor.selectedId) {
                const target = state.editor.annotations.find((annotation) => annotation.id === state.editor.selectedId);
                if (!target) return;
                const dx = point.x - state.editor.dragOrigin.point.x;
                const dy = point.y - state.editor.dragOrigin.point.y;
                const snapshot = state.editor.dragOrigin.annotation;
                target.x = snapshot.x + dx;
                target.y = snapshot.y + dy;
                redrawEditorCanvas();
                return;
            }

            if (!state.editor.draft) return;
            state.editor.draft.w = point.x - state.editor.draft.x;
            state.editor.draft.h = point.y - state.editor.draft.y;
            redrawEditorCanvas();
        };

        const onPointerUp = () => {
            if (!state.editor.image) return;
            const tool = state.editor.selectedTool;

            if (tool === 'move' && state.editor.isPointerDown) {
                state.editor.isPointerDown = false;
                state.editor.dragOrigin = null;
                pushEditorHistory();
                redrawEditorCanvas();
                return;
            }

            if (!state.editor.draft) {
                state.editor.isPointerDown = false;
                return;
            }

            state.editor.isPointerDown = false;
            const normalized = normalizeAnnotationRect(state.editor.draft);
            state.editor.draft = null;
            if (Math.abs(normalized.w) < 8 && Math.abs(normalized.h) < 8) {
                redrawEditorCanvas();
                return;
            }
            state.editor.annotations.push(normalized);
            pushEditorHistory();
            redrawEditorCanvas();
        };

        canvas.onmousedown = onPointerDown;
        canvas.onmousemove = onPointerMove;
        canvas.onmouseup = onPointerUp;
        canvas.onmouseleave = onPointerUp;
    }

    function redrawEditorCanvas() {
        const canvas = document.getElementById('starPhotoEditorCanvas');
        if (!canvas || !state.editor.image) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(state.editor.image, 0, 0);

        state.editor.annotations.forEach((annotation) => drawAnnotation(ctx, annotation, annotation.id === state.editor.selectedId));
        if (state.editor.draft) {
            drawAnnotation(ctx, normalizeAnnotationRect(state.editor.draft), false, true);
        }
    }

    async function savePhotoAnnotations(caption) {
        const photo = state.editor.photo;
        const session = getActiveSession();
        if (!photo || !session || !state.editor.image) throw new Error('Editor state is incomplete.');

        const annotations = deepClone(state.editor.annotations);
        const photoRef = getJournalRef().collection('sessions').doc(session.id).collection('photos').doc(photo.id);
        const annotatedPath = `star-journals/${state.currentJournalId}/sessions/${session.id}/photos/${photo.id}/annotated-preview.jpg`;

        if (!annotations.length) {
            await deleteStoragePath(photo.annotatedPreviewPath);
            await photoRef.set({
                caption,
                annotations: [],
                hasAnnotations: false,
                annotatedPreviewPath: '',
                annotatedPreviewUrl: '',
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            }, { merge: true });
            return;
        }

        const blob = await exportAnnotatedPreview();
        const uploadResult = await putBlob(annotatedPath, blob);
        await photoRef.set({
            caption,
            annotations,
            hasAnnotations: true,
            annotatedPreviewPath: annotatedPath,
            annotatedPreviewUrl: uploadResult.downloadURL,
            updatedAt: firebase.firestore.FieldValue.serverTimestamp()
        }, { merge: true });
    }

    async function exportAnnotatedPreview() {
        const canvas = document.createElement('canvas');
        canvas.width = state.editor.image.width;
        canvas.height = state.editor.image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(state.editor.image, 0, 0);
        state.editor.annotations.forEach((annotation) => drawAnnotation(ctx, annotation, false));
        return canvasToBlob(canvas, 'image/jpeg', 0.9);
    }

    function pushEditorHistory() {
        const snapshot = deepClone(state.editor.annotations);
        const currentSnapshot = JSON.stringify(state.editor.history[state.editor.historyIndex] || null);
        if (currentSnapshot === JSON.stringify(snapshot)) return;
        state.editor.history = state.editor.history.slice(0, state.editor.historyIndex + 1);
        state.editor.history.push(snapshot);
        state.editor.historyIndex = state.editor.history.length - 1;
    }

    function editorUndo() {
        if (state.editor.historyIndex <= 0) return;
        state.editor.historyIndex -= 1;
        state.editor.annotations = deepClone(state.editor.history[state.editor.historyIndex] || []);
        redrawEditorCanvas();
    }

    function editorRedo() {
        if (state.editor.historyIndex >= state.editor.history.length - 1) return;
        state.editor.historyIndex += 1;
        state.editor.annotations = deepClone(state.editor.history[state.editor.historyIndex] || []);
        redrawEditorCanvas();
    }

    function drawAnnotation(ctx, annotation, isSelected = false, isDraft = false) {
        ctx.save();
        ctx.strokeStyle = annotation.color || '#00ff9d';
        ctx.fillStyle = annotation.color || '#00ff9d';
        ctx.lineWidth = annotation.lineWidth || 4;
        ctx.font = `${annotation.fontSize || 26}px Inter, sans-serif`;
        ctx.globalAlpha = isDraft ? 0.75 : 1;

        if (annotation.type === 'circle') {
            const cx = annotation.x + annotation.w / 2;
            const cy = annotation.y + annotation.h / 2;
            ctx.beginPath();
            ctx.ellipse(cx, cy, Math.abs(annotation.w / 2), Math.abs(annotation.h / 2), 0, 0, Math.PI * 2);
            ctx.stroke();
        }

        if (annotation.type === 'arrow') {
            const startX = annotation.x;
            const startY = annotation.y;
            const endX = annotation.x + annotation.w;
            const endY = annotation.y + annotation.h;
            drawArrow(ctx, startX, startY, endX, endY, annotation.color, annotation.lineWidth || 4);
        }

        if (annotation.type === 'text') {
            ctx.textBaseline = 'top';
            ctx.fillText(annotation.text || '', annotation.x, annotation.y);
        }

        if (isSelected) {
            const box = getAnnotationBounds(ctx, annotation);
            ctx.strokeStyle = '#ffffff';
            ctx.setLineDash([8, 6]);
            ctx.lineWidth = 2;
            ctx.strokeRect(box.x - 6, box.y - 6, box.w + 12, box.h + 12);
        }
        ctx.restore();
    }

    function drawArrow(ctx, startX, startY, endX, endY, color, width) {
        const headLength = 18;
        const angle = Math.atan2(endY - startY, endX - startX);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - headLength * Math.cos(angle - Math.PI / 6), endY - headLength * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(endX - headLength * Math.cos(angle + Math.PI / 6), endY - headLength * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    function findAnnotationAtPoint(point) {
        const canvas = document.getElementById('starPhotoEditorCanvas');
        const ctx = canvas?.getContext('2d');
        if (!ctx) return null;

        const ordered = [...state.editor.annotations].reverse();
        return ordered.find((annotation) => {
            const bounds = getAnnotationBounds(ctx, annotation);
            return point.x >= bounds.x - 10
                && point.x <= bounds.x + bounds.w + 10
                && point.y >= bounds.y - 10
                && point.y <= bounds.y + bounds.h + 10;
        }) || null;
    }

    function getAnnotationBounds(ctx, annotation) {
        if (annotation.type === 'text') {
            const width = Math.max(40, ctx.measureText(annotation.text || '').width);
            const height = annotation.fontSize || 26;
            return { x: annotation.x, y: annotation.y, w: width, h: height };
        }

        if (annotation.type === 'arrow') {
            const x = Math.min(annotation.x, annotation.x + annotation.w);
            const y = Math.min(annotation.y, annotation.y + annotation.h);
            return { x, y, w: Math.abs(annotation.w), h: Math.abs(annotation.h) };
        }

        return { x: annotation.x, y: annotation.y, w: Math.abs(annotation.w), h: Math.abs(annotation.h) };
    }

    function createBaseAnnotation(type, payload) {
        return {
            id: `anno_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
            type,
            x: payload.x || 0,
            y: payload.y || 0,
            w: payload.w || 0,
            h: payload.h || 0,
            text: payload.text || '',
            color: payload.color || '#00ff9d',
            lineWidth: payload.lineWidth || 4,
            fontSize: payload.fontSize || 26,
            rotation: payload.rotation || 0,
            zIndex: payload.zIndex || Date.now()
        };
    }

    function normalizeAnnotationRect(annotation) {
        if (annotation.type === 'text') return annotation;
        const normalized = { ...annotation };
        if (normalized.w < 0) {
            normalized.x += normalized.w;
            normalized.w = Math.abs(normalized.w);
        }
        if (normalized.h < 0) {
            normalized.y += normalized.h;
            normalized.h = Math.abs(normalized.h);
        }
        return normalized;
    }

    function getCanvasPoint(canvas, event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    async function loadImageForEditor(url) {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.decoding = 'async';
        image.src = url;
        await new Promise((resolve, reject) => {
            image.onload = () => resolve();
            image.onerror = () => reject(new Error('编辑器载入图片失败。'));
        });
        return image;
    }

    async function loadImageFromFile(file) {
        const objectUrl = URL.createObjectURL(file);
        try {
            const image = new Image();
            image.decoding = 'async';
            image.src = objectUrl;
            await new Promise((resolve, reject) => {
                image.onload = () => resolve();
                image.onerror = () => reject(new Error(`浏览器无法解码 ${file.name}。请改传 JPEG 或 PNG。`));
            });
            return image;
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    }

    async function makeDerivedBlob(image, targetEdge, quality) {
        const scale = Math.min(1, targetEdge / Math.max(image.width, image.height));
        const width = Math.max(1, Math.round(image.width * scale));
        const height = Math.max(1, Math.round(image.height * scale));
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d', { alpha: false });
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(image, 0, 0, width, height);
        return canvasToBlob(canvas, 'image/jpeg', quality);
    }

    async function putBlob(path, blob) {
        const ref = state.storage.ref().child(path);
        await ref.put(blob, { contentType: blob.type || 'image/jpeg' });
        const downloadURL = await ref.getDownloadURL();
        const dimensions = await readBlobDimensions(blob);
        return {
            downloadURL,
            width: dimensions.width,
            height: dimensions.height
        };
    }

    async function readBlobDimensions(blob) {
        const url = URL.createObjectURL(blob);
        try {
            const image = new Image();
            image.src = url;
            await new Promise((resolve, reject) => {
                image.onload = () => resolve();
                image.onerror = () => reject(new Error('读取压缩图尺寸失败。'));
            });
            return {
                width: image.width,
                height: image.height
            };
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    function canvasToBlob(canvas, mimeType, quality) {
        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (!blob) {
                    reject(new Error('导出标注图片失败。'));
                    return;
                }
                resolve(blob);
            }, mimeType, quality);
        });
    }

    async function deleteStoragePath(path) {
        if (!path) return;
        try {
            await state.storage.ref().child(path).delete();
        } catch (error) {
            if (error && error.code === 'storage/object-not-found') return;
            console.warn('Storage delete skipped:', path, error);
        }
    }

    function showStatus(message, type = 'info') {
        if (!starJournalStatus) return;
        starJournalStatus.textContent = message;
        starJournalStatus.classList.remove('hidden', 'warning', 'error');
        if (type === 'warning') starJournalStatus.classList.add('warning');
        if (type === 'error') starJournalStatus.classList.add('error');
    }

    function closeModal(modal) {
        if (!modal) return;
        modal.classList.remove('active');
        modal.innerHTML = '';
    }

    function cleanupListeners() {
        Object.keys(state.unsubs).forEach((key) => {
            if (typeof state.unsubs[key] === 'function') {
                state.unsubs[key]();
            }
            state.unsubs[key] = null;
        });
    }

    function getJournalRef() {
        return state.db.collection('starJournals').doc(state.currentJournalId);
    }

    function getActiveSession() {
        return state.sessions.find((session) => session.id === state.activeSessionId) || null;
    }

    function findPhoto(photoId) {
        return state.activePhotos.find((photo) => photo.id === photoId) || null;
    }

    function getPhotoPreviewUrl(photo) {
        return photo.annotatedPreviewUrl || photo.displayUrl || photo.thumbUrl || '';
    }

    function renderTagHtml(tags = []) {
        if (!Array.isArray(tags) || !tags.length) return '<span class="star-tag">未标记</span>';
        return tags.map((tag) => `<span class="star-tag">${escapeHtml(tag)}</span>`).join('');
    }

    function parseTags(raw) {
        return Array.from(new Set(raw
            .split(',')
            .map((tag) => tag.trim())
            .filter(Boolean)));
    }

    function buildTimeLabel(session) {
        const start = session.startTime || '--:--';
        const end = session.endTime || '--:--';
        return `${session.sessionDate || '日期未知'} | ${start} -> ${end}`;
    }

    function getToolLabel(tool) {
        const labels = {
            move: '移动',
            circle: '圆圈',
            arrow: '箭头',
            text: '文字',
            delete: '删除',
            undo: '撤销',
            redo: '重做',
            reset: '重置'
        };
        return labels[tool] || tool;
    }

    function normalizeEmail(email) {
        return String(email || '').trim().toLowerCase();
    }

    function normalizeLocation(value) {
        return String(value || '')
            .trim()
            .toLowerCase()
            .replace(/\s+/g, ' ')
            .replace(/[^\w\u4e00-\u9fa5\s-]/g, '');
    }

    function deriveJournalName(email) {
        const local = String(email || 'pair').split('@')[0].replace(/[._-]+/g, ' ').trim();
        return local
            .split(' ')
            .filter(Boolean)
            .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
            .join(' ') || 'Private';
    }

    function getSeason(dateString) {
        const month = Number((dateString || '').split('-')[1] || 0);
        if ([12, 1, 2].includes(month)) return 'winter';
        if ([3, 4, 5].includes(month)) return 'spring';
        if ([6, 7, 8].includes(month)) return 'summer';
        if ([9, 10, 11].includes(month)) return 'autumn';
        return 'unknown';
    }

    function toMillis(value) {
        if (!value) return 0;
        if (typeof value.toMillis === 'function') return value.toMillis();
        if (typeof value.seconds === 'number') return value.seconds * 1000;
        const parsed = new Date(value).getTime();
        return Number.isFinite(parsed) ? parsed : 0;
    }

    function sortSessions(a, b) {
        const dateDiff = (b.sessionDate || '').localeCompare(a.sessionDate || '');
        if (dateDiff !== 0) return dateDiff;
        const timeDiff = (b.startTime || '').localeCompare(a.startTime || '');
        if (timeDiff !== 0) return timeDiff;
        return toMillis(b.updatedAt) - toMillis(a.updatedAt);
    }

    function truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return `${text.slice(0, maxLength - 1)}...`;
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function escapeAttribute(value) {
        return escapeHtml(value).replace(/`/g, '&#96;');
    }

    function formatMultiline(value) {
        return escapeHtml(value).replace(/\n/g, '<br>');
    }

    function deepClone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function waitForFirebaseReady() {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const timer = setInterval(() => {
                attempts += 1;
                if (window.firebase && firebase.apps && firebase.apps.length) {
                    clearInterval(timer);
                    resolve();
                    return;
                }
                if (attempts > 120) {
                    clearInterval(timer);
                    reject(new Error('Firebase 应用初始化超时。'));
                }
            }, 50);
        });
    }
});
