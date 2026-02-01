document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dashboardContainer = document.getElementById('dashboard-container');
    const loginContainer = document.getElementById('login-container');
    const userEmailDisplay = document.getElementById('userEmailDisplay');
    const logoutBtn = document.getElementById('logoutBtn');
    const adminLoginBtn = document.getElementById('portalLoginBtn');

    // Panes
    const projectTabsList = document.getElementById('project-tabs-list');
    const projectDetails = document.getElementById('project-details');
    const eventCards = document.querySelector('.event-cards');
    const taskList = document.querySelector('#task-panel .panel-body .space-y-3');
    const memoList = document.querySelector('#memo-panel .panel-body .space-y-2');
    const memoSearch = document.querySelector('.memo-search');

    // --- SETTINGS BUTTON LOGIC (Global Scope) ---
    // Bind immediately, don't wait for Auth
    // Settings listener removed as per user request

    // State
    let db = null;
    let auth = null;
    let currentUser = null;
    let activeProjectId = null;
    let allData = { projects: [], tasks: [], events: [], memos: [] };
    let isEditingProject = false;

    // Initialize Firebase
    async function init() {
        if (!window.firebaseConfig || window.firebaseConfig.apiKey === "YOUR_API_KEY_HERE") {
            console.error("Firebase not configured");
            return;
        }

        firebase.initializeApp(window.firebaseConfig);
        auth = firebase.auth();
        db = firebase.firestore();

        auth.onAuthStateChanged(async user => {
            currentUser = user;
            if (user) {
                loginContainer.style.display = 'none';
                dashboardContainer.style.display = 'flex';
                dashboardContainer.classList.remove('hidden');
                userEmailDisplay.textContent = user.email;

                // 1. Initialize DB Listeners (CRITICAL: Was missing)
                setupListeners();

                // 2. Settings Event is handled globally now

                // 3. Check for onboarding (Async)
                await checkAndMigrateOrInit(user);

            } else {
                loginContainer.style.display = 'flex';
                dashboardContainer.style.display = 'none';
                dashboardContainer.classList.add('hidden');
                currentUser = null;
            }
        });
    }

    // Helper to get user-scoped collection
    const getUserRef = (col) => db.collection('users').doc(currentUser.uid).collection(col);

    // --- ONBOARDING ---
    async function checkAndMigrateOrInit(user) {
        // Check if user has executed initialization
        const userDocRef = db.collection('users').doc(user.uid);
        const userDoc = await userDocRef.get();

        if (userDoc.exists && userDoc.data().initialized) {
            return; // Already setup
        }

        // New User logic -> Create Demo Data
        console.log("New user detected via system scan. Initializing workspace...");
        await createDemoData();
        await userDocRef.set({ initialized: true, email: user.email }, { merge: true });
    }

    async function createDemoData() {
        const batch = db.batch();

        // Sample Project
        const pRef = getUserRef('projects').doc();
        batch.set(pRef, {
            name: "DEMO_PROTOCOL_INIT",
            startDate: new Date().toISOString().split('T')[0],
            deadline: 'PRESENT',
            description: "Welcome to your personal command center. This system tracks your objectives with precision. Explore the interface.",
            subtasks: [
                { name: "Review System Interface", deadline: new Date().toISOString().split('T')[0], status: "active" },
                { name: "Establish First Objective", deadline: "", status: "pending" },
                { name: "System Initialization", deadline: "", status: "done" }
            ]
        });

        // Sample Tasks (3 items, one overdue/flash)
        const tRef1 = getUserRef('tasks').doc();
        batch.set(tRef1, {
            name: "Complete User Registration",
            deadline: new Date().toISOString().split('T')[0] // Today (Urgent/Flash)
        });
        const tRef2 = getUserRef('tasks').doc();
        batch.set(tRef2, {
            name: "Review Documentation",
            deadline: '2026-12-31'
        });
        const tRef3 = getUserRef('tasks').doc();
        batch.set(tRef3, {
            name: "Configure Settings",
            deadline: '2026-05-20'
        });

        // Sample Events (2 items)
        const eRef1 = getUserRef('events').doc();
        batch.set(eRef1, {
            name: "System Launch",
            date: new Date().toISOString().split('T')[0],
            description: "Initial deployment of the command center."
        });
        const eRef2 = getUserRef('events').doc();
        batch.set(eRef2, {
            name: "Quarterly Review",
            date: "2026-06-30",
            description: "Assess system performance and objective completion."
        });

        // Sample Memos (3 items)
        const mRef1 = getUserRef('memos').doc();
        batch.set(mRef1, {
            name: "SYSTEM_LOG_001",
            date: new Date().toISOString().split('T')[0],
            content: "Access granted. Personal workspace initialized. Ready for input."
        });
        const mRef2 = getUserRef('memos').doc();
        batch.set(mRef2, {
            name: "DESIGN_NOTES",
            date: new Date().toISOString().split('T')[0],
            content: "Interface uses a dark theme with high-contrast accent colors for optimal readability in low-light environments."
        });
        const mRef3 = getUserRef('memos').doc();
        batch.set(mRef3, {
            name: "REMINDER",
            date: new Date().toISOString().split('T')[0],
            content: "Check all active projects for critical path dependencies."
        });

        await batch.commit();
    }

    // Set up Real-time Listeners
    function setupListeners() {
        const collections = ['projects', 'tasks', 'events', 'memos'];
        let loadedCount = 0;
        collections.forEach(col => {
            // Updated to listen to USER SCOPED collection
            getUserRef(col).onSnapshot(snapshot => {
                allData[col] = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
                loadedCount++;
                if (loadedCount >= collections.length) {
                    renderAll();
                }
            });
        });
    }

    function renderAll() {
        renderProjects();
        renderEvents();
        renderTasks();
        renderMemos();
    }

    // --- 1. PROJECTS ---
    // Dynamic resizing logic
    let dynamicFitAll = 4;
    let dynamicFitWithOverflow = 3;

    // Observer to adjust visible tabs based on width
    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            const width = entry.contentRect.width;

            // Tab width 115px + 2px gap = 117px
            // Overflow button ~34px.

            // How many fit if NO overflow button needed?
            const rawFitAll = Math.floor(width / 117);

            // How many fit IF overflow button is needed?
            const rawFitWithOverflow = Math.floor((width - 34) / 117);

            // Logic: 
            // Mobile Optimization: Ensure at least 2 visible if screen isn't tiny
            // But if total space allows only 1, we must respect that.

            let newFitAll = Math.max(1, rawFitAll);
            let newFitWithOverflow = Math.max(2, rawFitWithOverflow);

            // If screen is REALLY small (<280px), fallback to 1
            if (width < 280) newFitWithOverflow = 1;

            if (newFitAll !== dynamicFitAll || newFitWithOverflow !== dynamicFitWithOverflow) {
                dynamicFitAll = newFitAll;
                dynamicFitWithOverflow = newFitWithOverflow;
                renderProjects();
            }
        }
    });

    // Start observing once DOM is ready (in init or here if element exists)
    // We'll queue it next tick to ensure element exists
    setTimeout(() => {
        const pPanel = document.getElementById('project-panel');
        if (pPanel) resizeObserver.observe(pPanel);
    }, 100);

    function renderProjects() {
        if (!projectTabsList) return;
        projectTabsList.innerHTML = '';
        // Restore strictly to CSS class for alignment control
        projectTabsList.className = 'project-tabs';

        // Remove existing dropdown if any (from previous renders to avoid dupe)
        const oldDrop = document.querySelector('.projects-dropdown-menu');
        if (oldDrop) oldDrop.remove();

        if (allData.projects.length === 0) {
            projectTabsList.innerHTML = '<div class="p-2 text-xs text-gray-500 italic">No active projects</div>';
            projectDetails.innerHTML = '<div class="p-10 text-center text-gray-600 italic">Add a project to begin tracking...</div>';
            return;
        }

        if (!activeProjectId) {
            // Sort by startDate DESCENDING (latest/closest to now first)
            const sorted = [...allData.projects].sort((a, b) => (b.startDate || '0000').localeCompare(a.startDate || '0000'));
            if (sorted.length > 0) activeProjectId = sorted[0].id;
        }

        const sortedProjects = [...allData.projects].sort((a, b) => (b.startDate || '0000').localeCompare(a.startDate || '0000'));

        // Determine how many tabs to show
        let MAX_VISIBLE = dynamicFitWithOverflow; // Default to overflow layout

        // If ALL projects fit in the container without an overflow button, use that count.
        if (sortedProjects.length <= dynamicFitAll) {
            MAX_VISIBLE = sortedProjects.length;
        } else {
            // Must use overflow button, so limited to fitWithOverflow
            MAX_VISIBLE = dynamicFitWithOverflow;
        }
        const visibleProjects = sortedProjects.slice(0, MAX_VISIBLE);
        const overflowProjects = sortedProjects.slice(MAX_VISIBLE);

        // Helper to check project urgency (any subtask urgent)
        const hasUrgentTasks = (p) => {
            return (p.subtasks || []).some(s =>
                s.status !== 'done' && s.status !== 'pending' && checkUrgency(s.deadline)
            );
        };

        // Render Visible Tabs
        visibleProjects.forEach(p => {
            const tab = document.createElement('div');
            tab.className = `project-tab ${activeProjectId === p.id ? 'active' : ''}`;

            // Urgency Dot
            if (hasUrgentTasks(p)) {
                tab.innerHTML += `<div class="urgent-dot"></div>`;
            }

            const span = document.createElement('span');
            span.textContent = p.name;
            tab.appendChild(span);

            tab.onclick = () => {
                if (isEditingProject && !confirm('Discard unsaved project changes?')) return;
                isEditingProject = false;
                activeProjectId = p.id;
                renderProjects();
            };
            projectTabsList.appendChild(tab);
        });

        // Render Overflow Button if needed
        if (overflowProjects.length > 0) {
            const container = document.createElement('div');
            container.className = 'project-overflow-container';

            const moreBtn = document.createElement('div');
            moreBtn.className = `project-more-btn ${overflowProjects.some(p => p.id === activeProjectId) ? 'overflow-active' : ''} text-[10px]`;
            moreBtn.innerHTML = '◀';

            container.appendChild(moreBtn);

            // Check if any overflow project has urgency
            if (overflowProjects.some(p => hasUrgentTasks(p))) {
                const dot = document.createElement('div');
                dot.className = 'urgent-dot';
                dot.style.position = 'absolute';
                dot.style.top = '6px';
                dot.style.right = '4px';
                container.appendChild(dot);
            }

            const dropdown = document.createElement('div');
            dropdown.className = 'projects-dropdown-menu';

            overflowProjects.forEach(p => {
                const item = document.createElement('div');
                item.className = `dropdown-item ${activeProjectId === p.id ? 'active' : ''}`;

                const nameSpan = document.createElement('span');
                nameSpan.textContent = p.name;

                // Dot in dropdown
                if (hasUrgentTasks(p)) {
                    nameSpan.innerHTML += `<span class="inline-block w-1.5 h-1.5 bg-red-500 rounded-full ml-2 animate-pulse"></span>`;
                }

                item.appendChild(nameSpan);
                item.onclick = (e) => {
                    e.stopPropagation(); // prevent closing immediately if logic changes
                    if (isEditingProject && !confirm('Discard unsaved project changes?')) return;
                    isEditingProject = false;
                    activeProjectId = p.id;
                    renderProjects();
                };
                dropdown.appendChild(item);
            });

            // Toggle Dropdown
            moreBtn.onclick = (e) => {
                e.stopPropagation();
                // Close others
                document.querySelectorAll('.projects-dropdown-menu').forEach(el => {
                    if (el !== dropdown) el.classList.remove('show');
                });
                const isShowing = dropdown.classList.toggle('show');
                if (isShowing) moreBtn.classList.add('active');
                else moreBtn.classList.remove('active');

                // Reset others if needed (optional)
            };

            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!container.contains(e.target)) {
                    dropdown.classList.remove('show');
                    moreBtn.classList.remove('active');
                }
            });

            container.appendChild(moreBtn);
            container.appendChild(dropdown);
            projectTabsList.appendChild(container);
        }

        const activeProject = allData.projects.find(p => p.id === activeProjectId);
        if (activeProject) {
            if (isEditingProject) renderProjectEditMode(activeProject);
            else renderProjectDetails(activeProject);
        }
    }

    function renderProjectDetails(p) {
        let nearestInfo = "None";
        if (p.subtasks && p.subtasks.length > 0) {
            const upcoming = p.subtasks
                .filter(s => s.status !== 'done' && s.deadline)
                .sort((a, b) => new Date(a.deadline) - new Date(b.deadline));
            if (upcoming.length > 0) {
                nearestInfo = `${upcoming[0].name} (${upcoming[0].deadline})`;
            }
        }

        projectDetails.innerHTML = `
            <div class="p-4 bg-white/[0.02] rounded border border-white/5 relative animate-fade-in shadow-inner">
                <button onclick="toggleProjectEdit('${p.id}')" class="absolute top-4 right-4 text-gray-500 hover:text-accent font-bold text-[10px] tracking-widest">[ EDIT_UNIT ]</button>
                
                <h2 class="text-xl font-bold text-white mb-6 pr-24 uppercase tracking-wide leading-tight">${p.name}</h2>

                <div class="grid grid-cols-2 gap-4 mb-4 border-b border-white/5 pb-4">
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1">Inception</div>
                        <div class="text-sm text-white font-mono">${p.startDate || 'YYYY-MM-DD'}</div>
                    </div>
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1">Termination</div>
                        <div class="text-sm text-white font-mono">${p.deadline === 'PRESENT' ? '<span class="text-accent font-bold">ONGOING</span>' : (p.deadline || 'YYYY-MM-DD')}</div>
                    </div>
                </div>

                <!-- Timeline Progress Bar -->
                <div class="mb-6 relative h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                    ${(() => {
                let percent = 50;
                let isIndefinite = false;
                if (p.deadline === 'PRESENT') {
                    isIndefinite = true;
                } else if (p.startDate && p.deadline) {
                    const start = new Date(p.startDate).getTime();
                    const end = new Date(p.deadline).getTime();
                    const now = new Date().getTime();
                    if (end > start) {
                        percent = ((now - start) / (end - start)) * 100;
                        percent = Math.min(Math.max(percent, 0), 100);
                    }
                }

                // For Indefinite: 50% width, starts from left (default)
                const finalWidth = isIndefinite ? 50 : percent;

                return `
                            <div class="absolute top-0 left-0 h-full bg-accent text-accent progress-active rounded-full" 
                                 style="width: ${finalWidth}%">
                            </div>
                        `;
            })()}
                </div>
                <div class="mb-4">
                    <span class="text-accent/80 text-[10px] uppercase font-bold tracking-tighter">Current_Priority:</span> 
                    <span class="text-white ml-2 text-xs">${nearestInfo}</span>
                </div>
                <p class="text-gray-400 text-xs leading-relaxed mb-6 font-sans">// ${p.description || 'System description: Not provided.'}</p>

                <div class="space-y-1" id="subtasks-container">
                    ${(p.subtasks || [])
                .sort((a, b) => {
                    // Status Weight: Active=1, Pending=2, Done=3
                    const statusWeight = { 'active': 1, 'pending': 2, 'done': 3 };
                    const wa = statusWeight[a.status] || 99;
                    const wb = statusWeight[b.status] || 99;
                    if (wa !== wb) return wa - wb;

                    // Secondary Sort: Date Ascending
                    if (!a.deadline) return 1;
                    if (!b.deadline) return -1;
                    return new Date(a.deadline) - new Date(b.deadline);
                })
                .map((s, idx) => {
                    // Only ACTIVE tasks flash when urgent. Pending/Done do not.
                    const isUrgent = s.status !== 'done' && s.status !== 'pending' && checkUrgency(s.deadline);
                    const statusColors = { 'active': 'text-yellow-400', 'pending': 'text-red-400', 'done': 'text-green-400' };
                    const displayStatus = (s.status || 'active').toUpperCase();
                    const statusClass = statusColors[s.status || 'active'] || 'text-gray-400';

                    return `
                        <div class="subtask-item subtask-${s.status || 'active'} flex justify-between items-center group rounded-sm px-3 py-2 border-l-2">
                            <span class="text-xs truncate mr-2">${s.name}</span>
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] font-bold tracking-wider ${isUrgent ? 'text-yellow-400 animate-pulse' : statusClass}">${displayStatus}</span>
                                <span class="text-xs ${isUrgent ? 'text-yellow-400 animate-pulse font-bold' : 'opacity-60'} font-mono">${s.deadline || ''}</span>
                            </div>
                        </div>
                    `;
                }).join('')}
                </div>
            </div>
        `;
    }

    // --- 2. EVENTS ---
    function renderEvents() {
        if (!eventCards) return;
        eventCards.innerHTML = '';
        if (allData.events.length === 0) {
            eventCards.innerHTML = '<div class="text-xs text-gray-600 italic px-2">No event horizon detected...</div>';
            return;
        }
        allData.events.sort((a, b) => new Date(a.date) - new Date(b.date)).forEach(e => {
            const card = document.createElement('div');
            card.className = 'event-card';
            card.onclick = () => openGenericModal('events', e);
            card.innerHTML = `
                <div class="text-[10px] text-accent font-bold mb-1 font-mono">${e.date}</div>
                <div class="text-xs text-white font-bold mb-1">${e.name}</div>
                <div class="text-[10px] text-gray-500 truncate">${e.description || ''}</div>
            `;
            eventCards.appendChild(card);
        });
    }

    // --- 3. TASKS ---
    function renderTasks() {
        if (!taskList) return;
        taskList.innerHTML = '';
        if (allData.tasks.length === 0) {
            taskList.innerHTML = '<div class="text-xs text-gray-600 italic">Queue empty...</div>';
            return;
        }
        allData.tasks.sort((a, b) => new Date(a.deadline) - new Date(b.deadline)).forEach(t => {
            const item = document.createElement('div');
            item.className = 'flex justify-between items-center group cursor-pointer hover:bg-white/5 p-2 rounded transition-all';
            item.onclick = () => openGenericModal('tasks', t);
            const isUrgent = checkUrgency(t.deadline);
            item.innerHTML = `
                <span class="text-xs text-gray-300">${t.name}</span>
                <span class="text-[10px] font-mono border px-1 rounded ${isUrgent ? 'text-red-500 animate-pulse font-bold border-red-500/50 bg-red-500/10' : 'text-accent border-accent/20 bg-accent/5'}">${t.deadline || ''}</span>
            `;
            taskList.appendChild(item);
        });
    }

    // --- UTILS ---
    function checkUrgency(dateStr) {
        if (!dateStr || dateStr === 'PRESENT') return false;
        const now = new Date();
        const todayStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        // Compare string YYYY-MM-DD
        return dateStr <= todayStr;
    }

    function normalizeDate(inputStr) {
        if (!inputStr) return "";
        if (inputStr === 'PRESENT') return 'PRESENT';
        // Clean and Split
        const parts = inputStr.split(/[-/.]/).filter(p => p.trim() !== "");
        if (parts.length === 0) return "";

        let year, month, day;

        if (parts.length === 1) { // 2026 -> 2026-01-01
            year = parts[0];
            month = "01";
            day = "01";
        } else if (parts.length === 2) { // 2026-05 -> 2026-05-01
            year = parts[0];
            month = parts[1];
            day = "01";
        } else {
            [year, month, day] = parts;
        }

        // Enforce 4-digit year sanity (no 500000+ years)
        const numericYear = parseInt(year) || 2026;
        year = Math.min(Math.max(numericYear, 1900), 2100);

        month = String(Math.min(Math.max(parseInt(month) || 1, 1), 12)).padStart(2, '0');
        day = String(Math.min(Math.max(parseInt(day) || 1, 1), 31)).padStart(2, '0');

        return `${year}-${month}-${day}`;
    }

    // --- 4. MEMOS ---
    function renderMemos() {
        if (!memoList) return;
        const query = (memoSearch?.value || '').toLowerCase();
        memoList.innerHTML = '';

        const filtered = allData.memos
            .filter(m => m.name.toLowerCase().includes(query) || (m.content || '').toLowerCase().includes(query))
            .sort((a, b) => {
                if (a.date && !b.date) return -1;
                if (!a.date && b.date) return 1;
                return new Date(a.date) - new Date(b.date);
            });

        if (filtered.length === 0) {
            memoList.innerHTML = '<div class="text-xs text-gray-600 italic">Vault archive null...</div>';
            return;
        }

        filtered.forEach(m => {
            const item = document.createElement('div');
            item.className = 'flex justify-between text-gray-500 hover:text-white transition-all cursor-pointer p-1 hover:pl-2 border-l border-transparent hover:border-accent';
            item.onclick = () => openGenericModal('memos', m);
            item.innerHTML = `
                <span class="truncate pr-2 text-xs">${m.name}</span>
                <span class="text-[10px] font-mono ${m.date ? 'text-accent' : ''}">${m.date || '---'}</span>
            `;
            memoList.appendChild(item);
        });
    }

    // --- MODALS ---
    window.openGenericModal = (type, data = null) => {
        const modal = document.getElementById('genericModal');
        modal.classList.add('active');

        let fields = '';
        if (type === 'events') {
            fields = `
                <div class="space-y-4">
                    <input type="text" id="m-name" placeholder="Event Name" class="form-input" value="${data?.name || ''}" autocomplete="off">
                    <input type="date" id="m-date" class="form-input" value="${data?.date || ''}">
                    <textarea id="m-desc" placeholder="Details/Description..." class="form-input h-24 pt-2 resize-none">${data?.description || ''}</textarea>
                </div>
            `;
        } else if (type === 'tasks') {
            fields = `
                <div class="space-y-4">
                    <input type="text" id="m-name" placeholder="Task Description" class="form-input" value="${data?.name || ''}" autocomplete="off">
                    <input type="date" id="m-deadline" class="form-input" value="${data?.deadline || ''}">
                </div>
            `;
        } else if (type === 'memos') {
            fields = `
                <div class="space-y-4">
                    <input type="text" id="m-name" placeholder="Memo Title" class="form-input" value="${data?.name || ''}" autocomplete="off">
                    <input type="date" id="m-date" class="form-input" value="${data?.date || ''}">
                    <textarea id="m-content" placeholder="Content/Passwords/Secure Notes..." class="form-input h-32 pt-2 resize-none">${data?.content || ''}</textarea>
                </div>
            `;
        }

        modal.innerHTML = `
            <div class="modal-form glass w-full max-w-sm border border-white/10 shadow-2xl rounded-lg">
                <div class="flex items-center gap-2 mb-6">
                    <div class="w-1 h-3 bg-accent"></div>
                    <h3 class="text-accent text-[10px] font-bold uppercase tracking-[0.2em]">MANAGE :: ${type.slice(0, -1)}</h3>
                </div>
                ${fields}
                <div class="flex justify-between gap-3 pt-6 mt-2 border-t border-white/5">
                    ${data ? `<button onclick="deleteEntry('${type}', '${data.id}')" class="text-red-500/60 hover:text-red-500 text-[10px] font-bold tracking-tighter">TERMINATE</button>` : '<span></span>'}
                    <div class="flex gap-4">
                        <button onclick="closeModal()" class="text-gray-500 hover:text-white text-[10px] font-bold">CANCEL</button>
                        <button onclick="saveEntry('${type}', '${data?.id || ''}')" class="bg-accent hover:bg-white text-black font-bold px-4 py-1.5 text-[10px] rounded transition-all">SYNC_TO_CLOUD</button>
                    </div>
                </div>
            </div>
        `;
    };

    window.closeModal = () => document.getElementById('genericModal').classList.remove('active');

    window.saveEntry = async (type, id) => {
        if (type === 'projects') return; // Projects handle own saves

        const data = {};
        if (type === 'events') {
            data.name = document.getElementById('m-name').value;
            data.date = normalizeDate(document.getElementById('m-date').value);
            data.description = document.getElementById('m-desc').value;
            if (!data.name || !data.date) return alert("Title and date required.");
        } else if (type === 'tasks') {
            data.name = document.getElementById('m-name').value;
            data.deadline = normalizeDate(document.getElementById('m-deadline').value);
            if (!data.name || !data.deadline) return alert("Title and deadline required.");
        } else if (type === 'memos') {
            data.name = document.getElementById('m-name').value;
            data.date = normalizeDate(document.getElementById('m-date').value);
            data.content = document.getElementById('m-content').value;
            if (!data.name) return alert("Title required.");
        }

        if (id) await getUserRef(type).doc(id).update(data);
        else await getUserRef(type).add(data);
        closeModal();
    };

    window.deleteEntry = async (type, id) => {
        if (confirm('Permanently purge this entry?')) {
            await getUserRef(type).doc(id).delete();
            closeModal();
        }
    };

    // --- PROJECT IN-LINE EDITING ---
    window.toggleProjectEdit = (id) => {
        if (!id && isEditingProject) {
            // Cancelled
            isEditingProject = false;
            renderAll();
            return;
        }

        isEditingProject = true;
        const p = allData.projects.find(proj => proj.id === id);
        if (p) renderProjectEditMode(p);
    };

    window.toggleDateInput = (cb) => {
        const input = document.getElementById('edit-p-end');
        if (cb.checked) {
            input.value = '';
            input.disabled = true;
            input.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            input.disabled = false;
            input.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    };

    function renderProjectEditMode(p) {
        const isOngoing = p.deadline === 'PRESENT';
        projectDetails.innerHTML = `
            <div class="p-4 bg-white/[0.03] rounded border border-accent/20 relative animate-fade-in">
                <div class="flex flex-col gap-4 mb-6">
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">PROJECT_NAME</div>
                        <input type="text" id="edit-p-name" value="${p.name}" class="form-input font-bold text-accent" autocomplete="off">
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">START_DATE</div>
                            <input type="date" id="edit-p-start" value="${p.startDate || ''}" class="form-input">
                        </div>
                        <div>
                            <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">END_DATE</div>
                            <div class="flex gap-2 items-center">
                                <input type="date" id="edit-p-end" value="${isOngoing ? '' : (p.deadline || '')}" class="form-input ${isOngoing ? 'opacity-50 cursor-not-allowed' : ''}" ${isOngoing ? 'disabled' : ''}>
                                <label class="flex items-center gap-1 cursor-pointer select-none bg-accent/10 px-2 py-1 rounded border border-accent/20 hover:bg-accent/20 transition-colors">
                                    <input type="checkbox" id="edit-p-ongoing" ${isOngoing ? 'checked' : ''} onchange="toggleDateInput(this)" class="accent-accent">
                                    <span class="text-[10px] text-accent font-bold">NOW</span>
                                </label>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">MISSION_OBJECTIVES</div>
                        <textarea id="edit-p-desc" class="form-input text-xs h-10 resize-none pt-2">${p.description || ''}</textarea>
                    </div>
                </div>

                <div class="flex items-center gap-2 mb-3 px-1 border-b border-white/5 pb-2">
                    <span class="text-[10px] font-bold text-gray-400 uppercase tracking-widest">SUB_SEQUENCES</span>
                    <button onclick="addNewSubtaskRow()" class="text-[10px] text-accent hover:underline">+ ADD</button>
                </div>
                <div id="edit-subtasks-list" class="space-y-2 mb-6 max-h-[150px] overflow-y-auto pr-1 thin-scroll">
                    ${(p.subtasks || []).map((s, idx) => `
                        <div class="flex gap-2 items-center bg-white/[0.02] p-2 rounded">
                            <input type="text" value="${s.name}" class="form-input text-[10px] flex-grow" placeholder="Task Name" autocomplete="off">
                            <input type="date" value="${s.deadline}" class="form-input text-[10px] w-28">
                            <select class="form-input text-[10px] w-20 appearance-none bg-accent/10 border-accent/20 text-accent font-bold">
                                <option value="active" ${s.status === 'active' ? 'selected' : ''}>ACTIVE</option>
                                <option value="pending" ${s.status === 'pending' ? 'selected' : ''}>PENDING</option>
                                <option value="done" ${s.status === 'done' ? 'selected' : ''}>DONE</option>
                            </select>
                            <button onclick="this.parentElement.remove()" class="text-red-500/80 hover:text-red-500 px-1">×</button>
                        </div>
                    `).join('')}
                </div>

                <div class="flex justify-between pt-4 mt-2 border-t border-white/5">
                    <button onclick="deleteProject('${p.id}')" class="text-red-500/60 hover:text-red-500 text-[10px] font-bold">PURGE_UNIT</button>
                    <div class="flex gap-4 items-center">
                        <button onclick="toggleProjectEdit()" class="text-gray-500 hover:text-white text-[10px] font-bold uppercase">ABORT</button>
                        <button onclick="saveProjectChanges('${p.id}')" class="bg-accent hover:bg-white text-black font-bold px-6 py-2 text-[10px] rounded transition-all">EXECUTE_SAVE</button>
                    </div>
                </div>
            </div>
        `;
    }

    // --- SETTINGS & THEME ---




    window.addNewSubtaskRow = () => {
        const container = document.getElementById('edit-subtasks-list');
        const row = document.createElement('div');
        row.className = 'flex gap-2 items-center bg-white/[0.02] p-2 rounded animate-fade-in';
        row.innerHTML = `
            <input type="text" class="form-input text-[10px] flex-grow" placeholder="New subtask..." autocomplete="off">
            <input type="date" class="form-input text-[10px] w-28">
            <select class="form-input text-[10px] w-20 appearance-none bg-accent/10 border-accent/20 text-accent font-bold">
                <option value="active">ACTIVE</option>
                <option value="pending">PENDING</option>
                <option value="done">DONE</option>
            </select>
            <button onclick="this.parentElement.remove()" class="text-red-500/80 hover:text-red-500 px-1">×</button>
        `;
        container.appendChild(row);
    };

    window.saveProjectChanges = async (id) => {
        const subtasks = [];
        const container = document.getElementById('edit-subtasks-list');
        Array.from(container.children).forEach(row => {
            const inputs = row.querySelectorAll('input, select');
            if (inputs[0].value) {
                subtasks.push({
                    name: inputs[0].value,
                    deadline: inputs[1].value,
                    status: inputs[2].value
                });
            }
        });

        const name = document.getElementById('edit-p-name').value;
        const start = document.getElementById('edit-p-start').value;
        const isOngoing = document.getElementById('edit-p-ongoing').checked;
        const end = isOngoing ? 'PRESENT' : document.getElementById('edit-p-end').value;
        const desc = document.getElementById('edit-p-desc').value;

        if (!name) return alert("System identity required.");
        if (start && end !== 'PRESENT' && new Date(start) > new Date(end)) {
            return alert("Error: Termination date cannot precede Inception date.");
        }

        // Logic Check: Subtask Deadline <= Project Deadline
        if (end !== 'PRESENT') {
            for (const s of subtasks) {
                if (s.deadline && new Date(s.deadline) > new Date(end)) {
                    return alert(`Logic Error: Sub-sequence "${s.name}" termination (${s.deadline}) exceeds Unit limit (${end}).`);
                }
            }
        }

        const projectUpdate = {
            name: name,
            startDate: normalizeDate(start),
            deadline: normalizeDate(end),
            description: desc,
            subtasks: subtasks.map(s => ({ ...s, deadline: normalizeDate(s.deadline) }))
        };

        if (id && !isOngoing) {
            // Updating existing via set (merge) or update
            await getUserRef('projects').doc(id).update(projectUpdate);
        } else if (id && isOngoing) {
            // Logic for ongoing might be same, just update
            await getUserRef('projects').doc(id).update(projectUpdate);
        } else {
            // Should not happen as we pre-generate ID on new
        }

        isEditingProject = false;
        renderAll();
    };

    window.deleteProject = async (id) => {
        if (confirm('Command: Purge whole Project unit from cloud storage?')) {
            await getUserRef('projects').doc(id).delete();
            activeProjectId = null;
            isEditingProject = false;
            renderAll();
        }
    };

    // Generic Add for Panels
    document.querySelectorAll('.panel-header button').forEach(btn => {
        const title = btn.parentElement.querySelector('.panel-title').textContent.toLowerCase();
        btn.onclick = async () => {
            if (title.includes('project')) {
                // 1. Generate ID locally first (Instant)
                const newDocRef = getUserRef('projects').doc();

                // 2. Set state immediately
                activeProjectId = newDocRef.id;
                isEditingProject = true; // Flag for edit mode BEFORE render

                // 3. Write data (This triggers the snapshot listener, which calls renderProjects -> renderProjectEditMode)
                await newDocRef.set({ name: 'UNIDENTIFIED_UNIT', subtasks: [] });

                // toggleProjectEdit(newDocRef.id); // Redundant now, render loop handles it
            } else if (title.includes('event')) {
                openGenericModal('events');
            } else if (title.includes('task')) {
                openGenericModal('tasks');
            } else if (title.includes('vault') || title.includes('memo')) {
                openGenericModal('memos');
            }
        };
    });

    // Root Listeners
    if (adminLoginBtn) adminLoginBtn.onclick = () => {
        const provider = new firebase.auth.GoogleAuthProvider();
        auth.signInWithPopup(provider);
    };
    if (logoutBtn) logoutBtn.onclick = () => auth.signOut();
    if (memoSearch) memoSearch.oninput = renderMemos;

    init();
});
