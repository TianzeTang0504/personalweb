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

    const settingsBtn = document.getElementById('settingsBtn');
    const genericModal = document.getElementById('genericModal');

    // --- SETTINGS BUTTON LOGIC ---
    settingsBtn.addEventListener('click', () => {
        if (!currentUser) return;
        showSettingsModal();
    });
    // State
    let db = null;
    let auth = null;
    let currentUser = null;
    let activeProjectId = null;
    let allData = { projects: [], tasks: [], events: [], memos: [] };
    let isEditingProject = false;
    // Listener cleanup
    let dbUnsubscribes = [];

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

                // Sync current user email and initial state to top-level doc
                await db.collection('users').doc(user.uid).set({
                    email: user.email
                }, { merge: true });

                setupListeners();
                await checkAndMigrateOrInit(user);

            } else {
                cleanupListeners();
                loginContainer.style.display = 'flex';
                dashboardContainer.style.display = 'none';
                dashboardContainer.classList.add('hidden');
                currentUser = null;
            }
        });
    }

    /**
     * Settings Modal Logic
     */
    async function showSettingsModal() {
        // 1. Get user configuration
        const userDoc = await db.collection('users').doc(currentUser.uid).get();
        const userData = userDoc.exists ? userDoc.data() : {};
        const isAdmin = userData.role === 'admin';

        // Defaults
        const currentReportTime = userData.reportTime !== undefined ? userData.reportTime : 8;
        const currentTimezone = userData.timezone || 'Asia/Shanghai';

        let modalHtml = `
            <div class="modal-content glass p-6 max-w-lg w-full relative animate-scale-in">
                <button class="absolute top-4 right-4 text-gray-500 hover:text-white modal-close-btn">&times;</button>
                <h2 class="text-xl font-bold text-white mb-6 tracking-widest border-l-4 border-accent pl-3">SYSTEM_SETTINGS</h2>
                
                <div class="space-y-6">
                    <!-- User Section -->
                    <div class="p-4 border border-white/5 bg-white/5 rounded">
                        <p class="text-[10px] text-accent mb-4 uppercase opacity-50 font-bold tracking-tighter">情报首选项 / INTELLIGENCE PREFERENCES</p>
                        
                         <!-- Toggle -->
                        <div class="flex items-center justify-between mb-4">
                            <span class="text-xs text-gray-300 font-mono">接收每日简报 (Email)</span>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" id="userReportToggle" class="sr-only peer" ${userData.receiveReport !== false ? 'checked' : ''}>
                                <div class="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-accent"></div>
                            </label>
                        </div>

                        <!-- Time Settings -->
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-[10px] text-gray-500 mb-1 font-mono uppercase">推送时间 (小时/24H)</label>
                                <select id="userReportTime" class="w-full bg-[#121212] border border-white/10 text-white text-xs rounded px-2 py-2 focus:border-accent outline-none font-mono hover:border-accent/50 transition-colors cursor-pointer appearance-none">
                                    ${[...Array(24).keys()].map(h => `<option value="${h}" class="bg-[#121212] text-white" ${h === currentReportTime ? 'selected' : ''}>${String(h).padStart(2, '0')}:00</option>`).join('')}
                                </select>
                            </div>
                            <div>
                                <label class="block text-[10px] text-gray-500 mb-1 font-mono uppercase">所在时区</label>
                                <select id="userTimezone" class="w-full bg-[#121212] border border-white/10 text-white text-xs rounded px-2 py-2 focus:border-accent outline-none font-mono hover:border-accent/50 transition-colors cursor-pointer appearance-none">
                                     <option value="Asia/Shanghai" class="bg-[#121212] text-white" ${currentTimezone === 'Asia/Shanghai' ? 'selected' : ''}>Asia/Shanghai (UTC+8)</option>
                                     <option value="America/Los_Angeles" class="bg-[#121212] text-white" ${currentTimezone === 'America/Los_Angeles' ? 'selected' : ''}>Pacific Time (UTC-7)</option>
                                     <option value="America/New_York" class="bg-[#121212] text-white" ${currentTimezone === 'America/New_York' ? 'selected' : ''}>Eastern Time (UTC-4)</option>
                                     <option value="Europe/London" class="bg-[#121212] text-white" ${currentTimezone === 'Europe/London' ? 'selected' : ''}>London (UTC+0)</option>
                                     <option value="Europe/Paris" class="bg-[#121212] text-white" ${currentTimezone === 'Europe/Paris' ? 'selected' : ''}>Paris (UTC+1)</option>
                                     <option value="Asia/Tokyo" class="bg-[#121212] text-white" ${currentTimezone === 'Asia/Tokyo' ? 'selected' : ''}>Tokyo (UTC+9)</option>
                                     <option value="Australia/Sydney" class="bg-[#121212] text-white" ${currentTimezone === 'Australia/Sydney' ? 'selected' : ''}>Sydney (UTC+10)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    ${isAdmin ? `
                        <!-- Admin Section -->
                        <div class="p-4 border border-accent/20 bg-accent/5 rounded">
                            <p class="text-[10px] text-accent mb-4 uppercase font-bold tracking-tighter">> SUPER_ADMIN_PANEL :: USER_MANAGEMENT</p>
                            <div id="admin-user-list" class="space-y-3 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
                                <div class="text-center py-4 text-xs opacity-50 italic">Scanning encrypted directories...</div>
                            </div>
                        </div>
                    ` : ''}
                </div>

                <div class="mt-8 pt-4 border-t border-white/5 text-[9px] text-gray-600 font-mono flex justify-between">
                    <span>UID: ${currentUser.uid}</span>
                    <span>ACCESS_LEVEL: ${isAdmin ? 'SUPER_USER' : 'CLIENT'}</span>
                </div>
            </div>
        `;

        genericModal.innerHTML = modalHtml;
        genericModal.classList.add('active');

        // Toggle logic for self
        const userToggle = document.getElementById('userReportToggle');
        userToggle.addEventListener('change', async (e) => {
            await db.collection('users').doc(currentUser.uid).update({
                receiveReport: e.target.checked
            });
        });

        // Time logic
        const timeSelect = document.getElementById('userReportTime');
        timeSelect.addEventListener('change', async (e) => {
            await db.collection('users').doc(currentUser.uid).update({
                reportTime: parseInt(e.target.value)
            });
        });

        // Timezone logic
        const tzSelect = document.getElementById('userTimezone');
        tzSelect.addEventListener('change', async (e) => {
            await db.collection('users').doc(currentUser.uid).update({
                timezone: e.target.value
            });
        });

        // Close logic
        const closeBtn = genericModal.querySelector('.modal-close-btn');
        closeBtn.onclick = () => genericModal.classList.remove('active');

        // If admin, load all users
        if (isAdmin) {
            const adminUserList = document.getElementById('admin-user-list');
            const usersSnap = await db.collection('users').get();
            adminUserList.innerHTML = '';

            usersSnap.forEach(doc => {
                const u = doc.data();
                const uId = doc.id;
                // Don't show self in manager or show with protect?
                const userRow = document.createElement('div');
                userRow.className = "flex items-center justify-between py-2 border-b border-white/5 last:border-0";
                userRow.innerHTML = `
                    <div class="flex flex-col">
                        <span class="text-xs text-white truncate max-w-[150px]">${u.email || 'UNNAMED_SEC'}</span>
                        <span class="text-[8px] text-gray-500 font-mono">${uId.substring(0, 8)}...</span>
                    </div>
                    <label class="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" class="sr-only peer admin-toggle" data-uid="${uId}" ${u.receiveReport !== false ? 'checked' : ''}>
                        <div class="w-8 h-4 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-accent/80"></div>
                    </label>
                `;
                adminUserList.appendChild(userRow);
            });

            // Admin Toggle Listeners
            adminUserList.querySelectorAll('.admin-toggle').forEach(tgl => {
                tgl.addEventListener('change', async (e) => {
                    const targetUid = e.target.getAttribute('data-uid');
                    await db.collection('users').doc(targetUid).update({
                        receiveReport: e.target.checked
                    });
                });
            });
        }
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
        const today = new Date();
        const yyyy = today.getFullYear();
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const dd = String(today.getDate()).padStart(2, '0');
        const todayStr = `${yyyy}-${mm}-${dd}`;

        // Sample Project (Chinese)
        const pRef = getUserRef('projects').doc();
        batch.set(pRef, {
            name: "指挥中心初始化",
            startDate: todayStr,
            deadline: 'PRESENT',
            description: "欢迎使用您的个人指挥中心。本系统将协助您以极高的精度追踪目标与任务。请探索各个模块。",
            subtasks: [
                { name: "熟悉系统界面布局", deadline: todayStr, status: "active" }, // Urgent
                { name: "建立第一个长期目标", deadline: "", status: "pending" },
                { name: "用户身份验证", deadline: "", status: "done" }
            ]
        });

        // Sample Tasks (2 items: 1 Urgent, 1 Normal)
        const tRef1 = getUserRef('tasks').doc();
        batch.set(tRef1, {
            name: "完成邮箱验证 (紧急)",
            deadline: todayStr // Today
        });
        const tRef2 = getUserRef('tasks').doc();
        batch.set(tRef2, {
            name: "阅读系统操作手册",
            deadline: '2026-12-31' // Far future
        });

        // Sample Events (2 items: 1 Urgent, 1 Normal)
        const eRef1 = getUserRef('events').doc();
        batch.set(eRef1, {
            name: "系统启动仪式",
            date: todayStr,
            description: "指挥中心正式上线运行。"
        });
        const eRef2 = getUserRef('events').doc();
        batch.set(eRef2, {
            name: "年度效能评估",
            date: "2026-12-31",
            description: "评估系统运行效率与目标完成度。"
        });

        // Sample Memo (1 item)
        const mRef1 = getUserRef('memos').doc();
        batch.set(mRef1, {
            name: "系统日志_001",
            date: todayStr,
            content: "访问权限已确认。个人工作区初始化完成。等待指令输入..."
        });

        await batch.commit();
    }

    // Set up Real-time Listeners
    function setupListeners() {
        cleanupListeners(); // Ensure no duplicates
        const collections = ['projects', 'tasks', 'events', 'memos'];
        let loadedCount = 0;
        collections.forEach(col => {
            const unsub = getUserRef(col).onSnapshot(snapshot => {
                allData[col] = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
                loadedCount++;
                // Initial load check
                if (loadedCount >= collections.length) {
                    renderAll();
                } else if (loadedCount > collections.length) {
                    // Subsequent updates
                    renderAll();
                }
            });
            dbUnsubscribes.push(unsub);
        });
    }

    function cleanupListeners() {
        dbUnsubscribes.forEach(unsub => unsub());
        dbUnsubscribes = [];
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
        let nearestInfo = "无";
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
                <button onclick="toggleProjectEdit('${p.id}')" class="absolute top-4 right-4 text-gray-500 hover:text-accent font-bold text-[10px] tracking-widest">[ 编辑项目 ]</button>
                
                <h2 class="text-xl font-bold text-white mb-6 pr-24 uppercase tracking-wide leading-tight">${p.name}</h2>

                <div class="grid grid-cols-2 gap-4 mb-4 border-b border-white/5 pb-4">
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1">启动日期</div>
                        <div class="text-sm text-white font-mono">${p.startDate || 'YYYY-MM-DD'}</div>
                    </div>
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1">截止日期</div>
                        <div class="text-sm text-white font-mono">${p.deadline === 'PRESENT' ? '<span class="text-accent font-bold">进行中</span>' : (p.deadline || 'YYYY-MM-DD')}</div>
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
                    <span class="text-accent/80 text-[10px] uppercase font-bold tracking-tighter">当前优先级:</span> 
                    <span class="text-white ml-2 text-xs">${nearestInfo}</span>
                </div>
                <p class="text-gray-400 text-xs leading-relaxed mb-6 font-sans">// ${p.description || '暂无描述'}</p>

                <div class="space-y-1" id="subtasks-container">
                    ${(p.subtasks || [])
                .sort((a, b) => {
                    // 1. Status Weight: Active=1, Pending=2, Done=3
                    const statusWeight = { 'active': 1, 'pending': 2, 'done': 3 };
                    const wa = statusWeight[a.status] || 1;
                    const wb = statusWeight[b.status] || 1;
                    if (wa !== wb) return wa - wb;

                    // 2. Date Presence (No deadline -> Bottom of group)
                    if (!a.deadline && !b.deadline) return 0;
                    if (!a.deadline) return 1;
                    if (!b.deadline) return -1;

                    // 3. Date Ascending (Urgent first)
                    return new Date(a.deadline) - new Date(b.deadline);
                })
                .map((s, idx) => {
                    // Only ACTIVE tasks flash when urgent. Pending/Done do not.
                    const isUrgent = s.status !== 'done' && s.status !== 'pending' && checkUrgency(s.deadline);
                    const statusColors = { 'active': 'text-yellow-400', 'pending': 'text-red-400', 'done': 'text-green-400' };

                    const statusMap = { 'active': '进行中', 'pending': '已挂起', 'done': '已完成' };
                    const displayStatus = statusMap[s.status || 'active'] || '进行中';

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
            eventCards.innerHTML = '<div class="text-xs text-gray-600 italic px-2">暂无日程安排...</div>';
            return;
        }
        allData.events.sort((a, b) => {
            const statusA = a.status === 'done' ? 1 : 0;
            const statusB = b.status === 'done' ? 1 : 0;
            if (statusA !== statusB) return statusA - statusB;
            return new Date(b.date) - new Date(a.date);
        }).forEach(e => {
            const card = document.createElement('div');
            const isDone = e.status === 'done';
            const isUrgent = !isDone && checkUrgency(e.date);
            card.className = `event-card ${isDone ? 'opacity-50' : ''} ${isUrgent ? 'border-red-500/50 shadow-[0_0_10px_rgba(239,68,68,0.2)]' : ''}`;
            card.onclick = () => openGenericModal('events', e);
            card.innerHTML = `
                <div class="text-[10px] ${isDone ? 'text-gray-500' : (isUrgent ? 'text-red-500 animate-pulse font-bold' : 'text-accent')} font-bold mb-1 font-mono">${e.date} ${isDone ? '[已完成]' : ''}</div>
                <div class="text-xs ${isDone ? 'text-gray-500 line-through' : 'text-white'} font-bold mb-1">${e.name}</div>
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
            taskList.innerHTML = '<div class="text-xs text-gray-600 italic">暂无任务...</div>';
            return;
        }
        allData.tasks.sort((a, b) => {
            const statusA = a.status === 'done' ? 1 : 0;
            const statusB = b.status === 'done' ? 1 : 0;
            if (statusA !== statusB) return statusA - statusB;

            // 按照DDL排序：紧急（日期早/小）的放上面，不紧急（日期晚/大）的放下面
            if (!a.deadline && !b.deadline) return 0;
            if (!a.deadline) return 1;
            if (!b.deadline) return -1;
            return new Date(a.deadline) - new Date(b.deadline);
        }).forEach(t => {
            const item = document.createElement('div');
            item.className = 'flex justify-between items-center group cursor-pointer hover:bg-white/5 p-2 rounded transition-all';
            item.onclick = () => openGenericModal('tasks', t);
            const isDone = t.status === 'done';
            const isUrgent = !isDone && checkUrgency(t.deadline);
            item.innerHTML = `
                <span class="text-xs ${isDone ? 'text-gray-600 line-through' : 'text-gray-300'} flex-grow truncate mr-4">${t.name}</span>
                <span class="text-[10px] font-mono border px-1 rounded flex-shrink-0 ${isDone ? 'text-gray-600 border-gray-600/30' : (isUrgent ? 'text-red-500 animate-pulse font-bold border-red-500/50 bg-red-500/10' : 'text-accent border-accent/20 bg-accent/5')}">${t.deadline || ''}</span>
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
            memoList.innerHTML = '<div class="text-xs text-gray-600 italic">暂无备忘...</div>';
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
                    <input type="text" id="m-name" placeholder="事件名称" class="form-input" value="${data?.name || ''}" autocomplete="off">
                    <input type="date" id="m-date" class="form-input" value="${data?.date || ''}">
                    <textarea id="m-desc" placeholder="详情描述..." class="form-input h-24 pt-2 resize-none">${data?.description || ''}</textarea>
                    <div class="flex items-center gap-2">
                        <label class="text-[10px] text-gray-500 uppercase">状态:</label>
                        <select id="m-status" class="form-input text-[10px] w-24 appearance-none bg-accent/10 border-accent/20 text-accent font-bold h-8">
                            <option value="active" ${data?.status !== 'done' ? 'selected' : ''}>进行中</option>
                            <option value="done" ${data?.status === 'done' ? 'selected' : ''}>已完成</option>
                        </select>
                    </div>
                </div>
            `;
        } else if (type === 'tasks') {
            fields = `
                <div class="space-y-4">
                    <input type="text" id="m-name" placeholder="任务描述" class="form-input" value="${data?.name || ''}" autocomplete="off">
                    <input type="date" id="m-deadline" class="form-input" value="${data?.deadline || ''}">
                    <div class="flex items-center gap-2">
                        <label class="text-[10px] text-gray-500 uppercase">状态:</label>
                        <select id="m-status" class="form-input text-[10px] w-24 appearance-none bg-accent/10 border-accent/20 text-accent font-bold h-8">
                            <option value="active" ${data?.status !== 'done' ? 'selected' : ''}>进行中</option>
                            <option value="done" ${data?.status === 'done' ? 'selected' : ''}>已完成</option>
                        </select>
                    </div>
                </div>
            `;
        } else if (type === 'memos') {
            fields = `
                <div class="space-y-4">
                    <input type="text" id="m-name" placeholder="备忘标题" class="form-input" value="${data?.name || ''}" autocomplete="off">
                    <input type="date" id="m-date" class="form-input" value="${data?.date || ''}">
                    <textarea id="m-content" placeholder="内容详情/加密笔记..." class="form-input h-32 pt-2 resize-none">${data?.content || ''}</textarea>
                </div>
            `;
        }

        modal.innerHTML = `
            <div class="modal-form glass w-full max-w-sm border border-white/10 shadow-2xl rounded-lg">
                <div class="flex items-center gap-2 mb-6">
                    <div class="w-1 h-3 bg-accent"></div>
                    <h3 class="text-accent text-[10px] font-bold uppercase tracking-[0.2em]">管理 :: ${type}</h3>
                </div>
                ${fields}
                <div class="flex justify-between gap-3 pt-6 mt-2 border-t border-white/5">
                    ${data ? `<button onclick="deleteEntry('${type}', '${data.id}')" class="text-red-500/60 hover:text-red-500 text-[10px] font-bold tracking-tighter">删除条目</button>` : '<span></span>'}
                    <div class="flex gap-4">
                        <button onclick="closeModal()" class="text-gray-500 hover:text-white text-[10px] font-bold">取消</button>
                        <button onclick="saveEntry('${type}', '${data?.id || ''}')" class="bg-accent hover:bg-white text-black font-bold px-4 py-1.5 text-[10px] rounded transition-all">保存更改</button>
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
            data.status = document.getElementById('m-status').value;
            if (!data.name || !data.date) return alert("Title and date required.");
        } else if (type === 'tasks') {
            data.name = document.getElementById('m-name').value;
            data.deadline = normalizeDate(document.getElementById('m-deadline').value);
            data.status = document.getElementById('m-status').value;
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
                        <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">项目名称</div>
                        <input type="text" id="edit-p-name" value="${p.name}" class="form-input font-bold text-accent" autocomplete="off">
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">启动日期</div>
                            <input type="date" id="edit-p-start" value="${p.startDate || ''}" class="form-input">
                        </div>
                        <div>
                            <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">结束日期</div>
                            <div class="flex gap-2 items-center">
                                <input type="date" id="edit-p-end" value="${isOngoing ? '' : (p.deadline || '')}" class="form-input ${isOngoing ? 'opacity-50 cursor-not-allowed' : ''}" ${isOngoing ? 'disabled' : ''}>
                                <label class="flex items-center gap-1 cursor-pointer select-none bg-accent/10 px-2 py-1 rounded border border-accent/20 hover:bg-accent/20 transition-colors">
                                    <input type="checkbox" id="edit-p-ongoing" ${isOngoing ? 'checked' : ''} onchange="toggleDateInput(this)" class="accent-accent">
                                    <span class="text-[10px] text-accent font-bold">长期/进行中</span>
                                </label>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1 px-1">项目/任务目标</div>
                        <textarea id="edit-p-desc" class="form-input text-xs h-10 resize-none pt-2">${p.description || ''}</textarea>
                    </div>
                </div>

                <div class="flex items-center gap-2 mb-3 px-1 border-b border-white/5 pb-2">
                    <span class="text-[10px] font-bold text-gray-400 uppercase tracking-widest">子任务序列</span>
                    <button onclick="addNewSubtaskRow()" class="text-[10px] text-accent hover:underline">+ 添加</button>
                </div>
                <div id="edit-subtasks-list" class="space-y-2 mb-6 max-h-[150px] overflow-y-auto pr-1 thin-scroll">
                    ${(p.subtasks || []).map((s, idx) => `
                        <div class="flex gap-2 items-center bg-white/[0.02] p-2 rounded">
                            <input type="text" value="${s.name}" class="form-input text-[10px] flex-grow" placeholder="任务名称" autocomplete="off">
                            <input type="date" value="${s.deadline}" class="form-input text-[10px] w-28">
                            <select class="form-input text-[10px] w-20 appearance-none bg-accent/10 border-accent/20 text-accent font-bold">
                                <option value="active" ${s.status === 'active' ? 'selected' : ''}>进行中</option>
                                <option value="pending" ${s.status === 'pending' ? 'selected' : ''}>已挂起</option>
                                <option value="done" ${s.status === 'done' ? 'selected' : ''}>已完成</option>
                            </select>
                            <button onclick="this.parentElement.remove()" class="text-red-500/80 hover:text-red-500 px-1">×</button>
                        </div>
                    `).join('')}
                </div>

                <div class="flex justify-between pt-4 mt-2 border-t border-white/5">
                    <button onclick="deleteProject('${p.id}')" class="text-red-500/60 hover:text-red-500 text-[10px] font-bold">删除项目</button>
                    <div class="flex gap-4 items-center">
                        <button onclick="toggleProjectEdit()" class="text-gray-500 hover:text-white text-[10px] font-bold uppercase">取消</button>
                        <button onclick="saveProjectChanges('${p.id}')" class="bg-accent hover:bg-white text-black font-bold px-6 py-2 text-[10px] rounded transition-all">保存更改</button>
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
        container.prepend(row);
        container.scrollTop = 0;
        row.querySelector('input').focus();
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
    const emailInput = document.getElementById('emailInput');
    const passwordInput = document.getElementById('passwordInput');
    const emailLoginBtn = document.getElementById('emailLoginBtn');

    // Toggle Elements
    const showRegisterBtn = document.getElementById('showRegisterBtn');
    const showForgotBtn = document.getElementById('showForgotBtn');
    const showLoginBtns = document.querySelectorAll('.showLoginBtn'); // Class for multiple back buttons

    // Views
    const loginFormView = document.getElementById('login-form-view');
    const registerFormView = document.getElementById('register-form-view');
    const forgotFormView = document.getElementById('forgot-form-view');

    // Action Elements
    const registerBtn = document.getElementById('registerBtn');
    const regEmailInput = document.getElementById('regEmailInput');
    const regPasswordInput = document.getElementById('regPasswordInput');

    const resetPassBtn = document.getElementById('resetPassBtn');
    const resetEmailInput = document.getElementById('resetEmailInput');

    // Toggle Logic
    if (showRegisterBtn) showRegisterBtn.onclick = () => {
        loginFormView.classList.add('hidden');
        registerFormView.classList.remove('hidden');
        forgotFormView.classList.add('hidden');
    };

    if (showForgotBtn) showForgotBtn.onclick = () => {
        loginFormView.classList.add('hidden');
        registerFormView.classList.add('hidden');
        forgotFormView.classList.remove('hidden');
    };

    showLoginBtns.forEach(btn => {
        btn.onclick = () => {
            registerFormView.classList.add('hidden');
            forgotFormView.classList.add('hidden');
            loginFormView.classList.remove('hidden');
        };
    });

    // Reset Password Logic
    if (resetPassBtn) {
        resetPassBtn.onclick = async () => {
            const email = resetEmailInput.value;
            if (!email) return alert("Please enter your registered email address.");

            try {
                await auth.sendPasswordResetEmail(email);
                alert(`Recovery Protocol Initiated: Reset link sent to ${email}. Check your inbox.`);
                // Return to login
                forgotFormView.classList.add('hidden');
                loginFormView.classList.remove('hidden');
            } catch (error) {
                console.error("Reset failed", error);
                alert("Recovery Failed: " + error.message);
            }
        };
    }

    // Login Logic
    if (emailLoginBtn) {
        emailLoginBtn.addEventListener('click', async () => {
            const email = emailInput.value;
            const password = passwordInput.value;
            if (!email || !password) return alert("Credentials required.");

            try {
                const userCredential = await auth.signInWithEmailAndPassword(email, password);
                if (!userCredential.user.emailVerified) {
                    // Check specifically if it's the admin or a verified user constraint
                    // For now, allow but warn. Or enforce? 
                    // User asked for "prevent invalid emails".
                    // We will enforce it.
                    if (userCredential.user.emailVerified === false) {
                        alert("ACCESS DENIED: Email verification pending. Please check your inbox.");
                        await auth.signOut();
                        return;
                    }
                }
            } catch (error) {
                console.error("Login failed", error);
                alert("Access Denied: " + error.message);
            }
        });
    }

    // Register Logic
    if (registerBtn) {
        registerBtn.onclick = async () => {
            const email = regEmailInput.value;
            const password = regPasswordInput.value;
            if (!email || !password) return alert("All fields required for initialization.");

            try {
                const userCredential = await auth.createUserWithEmailAndPassword(email, password);
                await userCredential.user.sendEmailVerification();

                alert(`PROTOCOL INITIATED: Verification link sent to ${email}. Please verify to access system.`);
                await auth.signOut();

                // Return to login
                registerFormView.classList.add('hidden');
                loginFormView.classList.remove('hidden');
                emailInput.value = email;
            } catch (error) {
                console.error("Registration failed", error);
                alert("Initialization Failed: " + error.message);
            }
        };
    }

    if (adminLoginBtn) adminLoginBtn.onclick = () => {
        const provider = new firebase.auth.GoogleAuthProvider();
        auth.signInWithPopup(provider).catch(error => {
            console.error("Google Auth failed", error);
            alert("Google Authentication failed: " + error.message);
        });
    };
    if (logoutBtn) logoutBtn.onclick = () => auth.signOut();
    if (memoSearch) memoSearch.oninput = renderMemos;

    init();
});
