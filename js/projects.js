document.addEventListener('DOMContentLoaded', () => {
    // Projects data for the "Load More" functionality
    const additionalProjects = [
        {
            title: "Voice Recognition System",
            description: "A real-time speech recognition system that accurately transcribes audio input into text.",
            tags: ["TensorFlow", "PyTorch", "WebRTC"],
            icon: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>`,
            id: "project4"
        },
        {
            title: "Computer Vision for Healthcare",
            description: "AI models that assist radiologists in detecting irregularities in medical images.",
            tags: ["Python", "OpenCV", "Deep Learning"],
            icon: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>`,
            id: "project5"
        },
        {
            title: "Recommendation Engine",
            description: "A content recommendation system using collaborative filtering and neural networks.",
            tags: ["Machine Learning", "Flask", "PostgreSQL"],
            icon: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`,
            id: "project6"
        }
    ];

    // Function to create project element
    function createProjectElement(project) {
        const div = document.createElement('div');
        div.className = 'bg-white rounded-lg shadow-sm overflow-hidden border border-gray-100 transition-all hover:shadow-md';
        div.innerHTML = `
            <div class="p-6">
                <div class="flex items-center mb-4">
                    <div class="bg-accent/10 rounded-md p-3">
                        ${project.icon}
                    </div>
                    <h3 class="ml-3 text-lg font-semibold text-gray-900">${project.title}</h3>
                </div>
                <p class="text-gray-600 mb-4">
                    ${project.description}
                </p>
                <div class="flex flex-wrap gap-2 mb-4">
                    ${project.tags.map(tag => `<span class="text-xs bg-gray-100 px-2 py-1 rounded-md">${tag}</span>`).join('')}
                </div>
                <a href="#" class="text-sm text-accent font-medium hover:underline view-project" data-id="${project.id}">View Project →</a>
            </div>
        `;
        
        // Add animation to make the new elements fade in
        div.style.opacity = '0';
        div.style.animation = 'fadeIn 0.5s forwards';
        
        return div;
    }

    // Load more projects functionality
    const loadMoreButton = document.getElementById('loadMoreProjects');
    if (loadMoreButton) {
        loadMoreButton.addEventListener('click', (e) => {
            e.preventDefault();
            const projectsGrid = document.querySelector('#projects .grid');
            
            if (projectsGrid) {
                // Add new projects to the grid
                additionalProjects.forEach(project => {
                    const projectElement = createProjectElement(project);
                    projectsGrid.appendChild(projectElement);
                });
                
                // Hide the button after loading all projects
                loadMoreButton.style.display = 'none';
                
                // Add event listeners to the new project links
                projectsGrid.querySelectorAll('.view-project').forEach(link => {
                    link.addEventListener('click', handleProjectClick);
                });
            }
        });
    }
    
    // Add click event listeners to project links
    document.querySelectorAll('.view-project').forEach(link => {
        link.addEventListener('click', handleProjectClick);
    });

    function handleProjectClick(e) {
        e.preventDefault();
        const projectId = this.getAttribute('data-id');
        window.contentManager.showContent('project', projectId);
    }
});
