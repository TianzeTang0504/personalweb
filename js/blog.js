document.addEventListener('DOMContentLoaded', () => {
    // All Blog Data
    const blogData = [
        {
            title: "Use Agent AI and Cursor to build a personal website",
            excerpt: "This website is built using Flowith, Cursor, and Tencent Edgeone Pages.",
            date: "2025-04-12",
            readTime: "5 min read",
            tags: ["AI Tools", "Web Dev"],
            id: "blog1"
        },
        {
            title: "Deep Learning in Medical Imaging",
            excerpt: "Exploring how CNNs are revolutionizing early diagnosis in radiology.",
            date: "2025-03-28",
            readTime: "8 min read",
            tags: ["Deep Learning", "Medical AI"],
            id: "blog2"
        }
    ];

    const blogGrid = document.querySelector('#blog .grid');
    let loadMoreButton = document.getElementById('viewAllBlogs');
    if (!loadMoreButton) {
        loadMoreButton = document.querySelector('#blog .text-center a');
    }
    
    const INITIAL_COUNT = 3;
    let currentCount = 0;

    function createBlogElement(blog) {
        const div = document.createElement('div');
        div.className = 'blog-card glass';
        
        div.innerHTML = `
            <div class="p-6">
                <div class="flex items-center text-xs text-accent mb-3 font-mono">
                    <svg class="h-4 w-4 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd" />
                    </svg>
                    <span>${blog.date}</span>
                </div>
                <h3 class="text-lg font-semibold text-white mb-2 hover:text-accent transition-colors">${blog.title}</h3>
                <p class="text-gray-400 mb-4 text-sm">
                    ${blog.excerpt}
                </p>
                
                <div class="flex flex-wrap gap-2 mb-4">
                    ${blog.tags ? blog.tags.map(tag => `<span class="px-2 py-1 bg-accent/10 text-accent rounded text-xs font-mono border border-accent/20">${tag}</span>`).join('') : ''}
                </div>

                <a href="#" class="text-sm text-accent font-medium hover:underline view-blog inline-flex items-center" data-id="${blog.id}">
                    read_more() <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                </a>
            </div>
        `;
        
        div.style.opacity = '0';
        div.style.animation = 'fadeIn 0.5s forwards';
        
        return div;
    }

    function init() {
        if (blogGrid) {
            blogGrid.innerHTML = ''; 
            
            const initialBlogs = blogData.slice(0, INITIAL_COUNT);
            initialBlogs.forEach(blog => {
                blogGrid.appendChild(createBlogElement(blog));
            });
            currentCount = INITIAL_COUNT;

            if (blogData.length > INITIAL_COUNT) {
                if (loadMoreButton) {
                    loadMoreButton.style.display = 'inline-flex';
                    
                    loadMoreButton.onclick = (e) => {
                        e.preventDefault();
                        const remainingBlogs = blogData.slice(currentCount);
                        remainingBlogs.forEach(blog => {
                            blogGrid.appendChild(createBlogElement(blog));
                        });
                        currentCount += remainingBlogs.length;
                        loadMoreButton.style.display = 'none';
                    };
                }
            } else {
                if (loadMoreButton) loadMoreButton.style.display = 'none';
            }
        }
    }

    if (blogGrid) {
        blogGrid.addEventListener('click', (e) => {
            const link = e.target.closest('.view-blog');
            if (link) {
                e.preventDefault();
                const blogId = link.getAttribute('data-id');
                if (window.contentManager) {
                    window.contentManager.showContent('blog', blogId);
                }
            }
        });
    }

    init();
});
