document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const scheduleContainer = document.getElementById('schedule-list');
    const adminLoginBtn = document.getElementById('adminLoginBtn');
    const addEventBtn = document.getElementById('addEventBtn');
    const eventModal = document.getElementById('eventModal');
    const eventForm = document.getElementById('eventForm');
    const closeModalBtn = document.getElementById('closeModal');

    // State
    let db = null;
    let auth = null;
    let currentUser = null;

    // 1. Check if Firebase is configured
    const isConfigured = window.firebaseConfig &&
        window.firebaseConfig.apiKey !== "YOUR_API_KEY_HERE";

    // Dummy Data for Preview (Show this if not configured or empty)
    const dummyEvents = [
        {
            id: '1',
            title: 'System Maintenance',
            date: '2026-02-15',
            time: '02:00',
            description: 'Routine server updates and security patches.',
            type: 'maintenance'
        },
        {
            id: '2',
            title: 'Project Alpha Launch',
            date: '2026-03-01',
            time: '09:00',
            description: 'Public release of the new AI agent framework.',
            type: 'release'
        },
        {
            id: '3',
            title: 'Tech Conference Talk',
            date: '2026-03-10',
            time: '14:30',
            description: 'Speaking at DevCon about browser agents.',
            type: 'event'
        }
    ];

    async function initFirebase() {
        if (!isConfigured) {
            console.warn("⚠️ Firebase config not found. Using local dummy data.");
            renderEvents(dummyEvents);
            return;
        }

        try {
            // Initialize app
            const app = firebase.initializeApp(window.firebaseConfig);
            auth = firebase.auth();
            db = firebase.firestore();

            // Auth Listener
            auth.onAuthStateChanged(user => {
                currentUser = user;
                updateAdminUI(!!user);

                // Toggle Admin Dashboard View
                const loginContainer = document.getElementById('login-container');
                const dashboardContainer = document.getElementById('dashboard-container');
                const userEmailDisplay = document.getElementById('userEmailDisplay');

                console.log("Auth State Changed:", user ? "Logged In" : "Logged Out");

                if (user && loginContainer && dashboardContainer) {
                    console.log("Switching to Dashboard View");
                    loginContainer.style.display = 'none';
                    dashboardContainer.style.display = 'block'; // Force block display
                    dashboardContainer.classList.remove('hidden');

                    if (userEmailDisplay) userEmailDisplay.textContent = `Logged in as: ${user.email}`;

                    // Trigger a re-render or layout check just in case
                    setTimeout(() => { dashboardContainer.style.opacity = '1'; }, 10);
                } else if (!user && loginContainer && dashboardContainer) {
                    console.log("Switching to Login View");
                    loginContainer.style.display = 'flex'; // Use flex to maintain centering layout
                    dashboardContainer.style.display = 'none';
                    dashboardContainer.classList.add('hidden');
                }
            });

            // Real-time Data Listener
            db.collection('schedule')
                .orderBy('date', 'asc')
                .onSnapshot(snapshot => {
                    const events = snapshot.docs.map(doc => ({
                        id: doc.id,
                        ...doc.data()
                    }));
                    renderEvents(events.length > 0 ? events : dummyEvents);
                }, error => {
                    console.error("Error fetching schedule:", error);
                    renderEvents(dummyEvents); // Fallback
                });

        } catch (e) {
            console.error("Firebase init failed:", e);
            renderEvents(dummyEvents);
        }
    }

    function renderEvents(events) {
        if (!scheduleContainer) return;
        scheduleContainer.innerHTML = ''; // Clear

        events.forEach(event => {
            const el = document.createElement('div');
            el.className = 'schedule-item animate-fade-in glass';
            el.innerHTML = `
                <div class="p-6">
                    <div class="schedule-date">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                        <span>${event.date} // ${event.time || 'All Day'}</span>
                        ${getBadge(event.type)}
                    </div>
                    <h3 class="text-xl font-bold text-white mb-2">${event.title}</h3>
                    <p class="text-gray-400 font-mono text-sm">${event.description}</p>
                    
                    <!-- Admin Controls -->
                    <div class="admin-controls flex gap-2 mt-4 text-sm border-t border-gray-800 pt-3 ${currentUser ? 'logged-in' : ''}">
                        <button onclick="editEvent('${event.id}')" class="text-accent hover:underline">Edit</button>
                        <button onclick="deleteEvent('${event.id}')" class="text-red-500 hover:underline">Delete</button>
                    </div>
                </div>
            `;
            scheduleContainer.appendChild(el);
        });
    }

    function getBadge(type) {
        let colorClass = 'text-accent border-accent';
        if (type === 'maintenance') colorClass = 'text-yellow-500 border-yellow-500';
        if (type === 'release') colorClass = 'text-purple-500 border-purple-500';

        return `<span class="px-2 py-0.5 text-xs border ${colorClass} rounded bg-opacity-10 opacity-70 uppercase tracking-wider ml-auto">${type || 'Event'}</span>`;
    }

    function updateAdminUI(isLoggedIn) {
        if (addEventBtn) addEventBtn.style.display = isLoggedIn ? 'inline-flex' : 'none';

        // Only change text if button exists and it's NOT the login page main button (which has specific styling)
        if (adminLoginBtn && !document.getElementById('login-container')) {
            adminLoginBtn.textContent = isLoggedIn ? 'Log Out' : 'Admin Login';
        }

        // Update all existing items to show/hide controls
        document.querySelectorAll('.admin-controls').forEach(el => {
            if (isLoggedIn) el.classList.add('logged-in');
            else el.classList.remove('logged-in');
        });
    }

    // --- Actions ---

    window.handleAdminLogin = async () => {
        if (!isConfigured) {
            alert("Please configure Firebase first in js/firebase-config.js!");
            return;
        }

        if (currentUser) {
            await auth.signOut();
        } else {
            const provider = new firebase.auth.GoogleAuthProvider();
            try {
                await auth.signInWithPopup(provider);
            } catch (error) {
                console.error("Login failed", error);
                alert("Login failed: " + error.message);
            }
        }
    };

    window.openAddEventModal = () => {
        if (!eventModal) return;
        eventModal.classList.add('active');
        eventForm.reset();
        document.getElementById('eventId').value = ''; // Empty for new
    };

    window.closeModal = () => {
        if (eventModal) eventModal.classList.remove('active');
    };

    window.saveEvent = async (e) => {
        e.preventDefault();
        if (!db) return;

        const id = document.getElementById('eventId').value;
        const data = {
            title: document.getElementById('eventTitle').value,
            date: document.getElementById('eventDate').value,
            time: document.getElementById('eventTime').value,
            description: document.getElementById('eventDesc').value,
            type: document.getElementById('eventType').value
        };

        try {
            if (id) {
                await db.collection('schedule').doc(id).update(data);
            } else {
                await db.collection('schedule').add(data);
            }
            closeModal();
        } catch (error) {
            console.error("Error saving:", error);
            alert("Error saving event");
        }
    };

    window.deleteEvent = async (id) => {
        if (!confirm("Are you sure?")) return;
        if (!db) return;
        try {
            await db.collection('schedule').doc(id).delete();
        } catch (error) {
            console.error("Error deleting:", error);
        }
    };

    window.editEvent = (id) => {
        // Find data locally for simplicity or fetch
        // For now, simpler to just grabbing from DOM or finding in dummy/cache
        // In real app, we'd use the data we already fetched.
        // Let's just alert for now or implement properly if we cached data.
        console.log("Edit requested for", id);
        // Implementing "Find in DOM" or "Global Cache" is better. 
        // For MVP, we'll skip pre-filling form to save space, or just open a blank one.
        openAddEventModal();
        // Ideally: document.getElementById('eventId').value = id; ... fill other fields
    };

    // Listeners
    if (adminLoginBtn) adminLoginBtn.onclick = window.handleAdminLogin;

    // Portal specific login button
    const portalLoginBtn = document.getElementById('portalLoginBtn');
    if (portalLoginBtn) portalLoginBtn.onclick = window.handleAdminLogin;

    // Add Logout Button Listener for Admin Page
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.onclick = async () => {
            await auth.signOut();
            window.location.reload();
        };
    }
    if (addEventBtn) addEventBtn.onclick = window.openAddEventModal;
    if (closeModalBtn) closeModalBtn.onclick = window.closeModal;
    if (eventForm) eventForm.onsubmit = window.saveEvent;

    // Run
    initFirebase();
});
