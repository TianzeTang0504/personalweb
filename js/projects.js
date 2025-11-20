document.addEventListener('DOMContentLoaded', () => {
    // All Projects Data
    const projectsData = [
        {
            title: "NLP API",
            description: "A robust API for text analysis, sentiment detection, and language translation using transformer models.",
            tags: ["Python", "Flask", "HuggingFace"],
            icon: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>`,
            id: "project1"
        },
        {
            title: "Image Generation App",
            description: "This web application generates creative images from text prompts using Stable Diffusion models.",
            tags: ["React", "Node.js", "Stable Diffusion"],
            icon: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>`,
            id: "project2"
        }
    ];

    const projectsGrid = document.querySelector('#projects .grid');
    const loadMoreButton = document.getElementById('loadMoreProjects');
    const INITIAL_COUNT = 3;
    let currentCount = 0;

    // Function to create project element
    function createProjectElement(project) {
        const div = document.createElement('div');
        // Updated to match dark theme / glassmorphism
        div.className = 'project-card glass rounded-lg overflow-hidden border border-gray-800 group';
        div.innerHTML = `
            <div class="p-6">
                <div class="flex items-center mb-4">
                    <div class="bg-accent/10 rounded-md p-3 border border-accent/30">
                        ${project.icon}
                    </div>
                    <h3 class="ml-3 text-lg font-semibold text-white group-hover:text-accent transition-colors">${project.title}</h3>
                </div>
                <p class="text-gray-400 mb-4 text-sm">
                    ${project.description}
                </p>
                <div class="flex flex-wrap gap-2 mb-4">
                    ${project.tags.map(tag => `<span class="project-tag">${tag}</span>`).join('')}
                </div>
                <a href="#" class="text-sm text-accent font-medium hover:underline view-project inline-flex items-center" data-id="${project.id}">
                    View Project <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                </a>
            </div>
        `;
        
        // Add animation to make the new elements fade in
        div.style.opacity = '0';
        div.style.animation = 'fadeIn 0.5s forwards';
        
        return div;
    }

    function init() {
        if (projectsGrid) {
            projectsGrid.innerHTML = ''; // Clear HTML content
            
            // Render all projects since count is 2 which is < INITIAL_COUNT
            const initialProjects = projectsData.slice(0, INITIAL_COUNT);
            initialProjects.forEach(project => {
                projectsGrid.appendChild(createProjectElement(project));
            });
            currentCount = INITIAL_COUNT;

            // Handle button visibility
            if (projectsData.length > INITIAL_COUNT) {
                if (loadMoreButton) {
                    loadMoreButton.style.display = 'inline-flex';
                    loadMoreButton.onclick = (e) => {
                        e.preventDefault();
                        const remainingProjects = projectsData.slice(currentCount);
                        remainingProjects.forEach(project => {
                            projectsGrid.appendChild(createProjectElement(project));
                        });
                        currentCount += remainingProjects.length;
                        loadMoreButton.style.display = 'none';
                    };
                }
            } else {
                if (loadMoreButton) loadMoreButton.style.display = 'none';
            }
        }
        
        // Event delegation on the grid
        if (projectsGrid) {
            projectsGrid.addEventListener('click', (e) => {
                const link = e.target.closest('.view-project');
                if (link) {
                    handleProjectClick.call(link, e);
                }
            });
        }
    }

    function handleProjectClick(e) {
        e.preventDefault();
        const projectId = this.getAttribute('data-id');
        if (window.contentManager) {
            window.contentManager.showContent('project', projectId);
        }
    }

    init();
});
