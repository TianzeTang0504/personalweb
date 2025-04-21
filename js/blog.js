document.addEventListener('DOMContentLoaded', () => {
    // Blog data for the "Load More" functionality
    const additionalBlogs = [
        // {
        //     title: "The Future of AI in Healthcare",
        //     excerpt: "Exploring how artificial intelligence is revolutionizing patient care and medical research.",
        //     date: "2024-02-15",
        //     readTime: "8 min read",
        //     id: "blog4"
        // },
        // {
        //     title: "Building Scalable Microservices",
        //     excerpt: "Best practices and patterns for designing and implementing microservices architecture.",
        //     date: "2024-02-10",
        //     readTime: "10 min read",
        //     id: "blog5"
        // },
        // {
        //     title: "The Ethics of AI Development",
        //     excerpt: "A deep dive into the ethical considerations and responsibilities in AI development.",
        //     date: "2024-02-05",
        //     readTime: "12 min read",
        //     id: "blog6"
        // }
    ];

    // Function to create blog post element
    function createBlogElement(blog) {
        const div = document.createElement('div');
        div.className = 'bg-white rounded-lg shadow-lg overflow-hidden';
        div.innerHTML = `
            <div class="p-6">
                <div class="flex items-center mb-4">
                    <span class="text-sm text-gray-500">${blog.date}</span>
                    <span class="mx-2 text-gray-300">|</span>
                    <span class="text-sm text-gray-500">${blog.readTime}</span>
                </div>
                <h3 class="text-xl font-bold mb-2">${blog.title}</h3>
                <p class="text-gray-600 mb-4">${blog.excerpt}</p>
                <div class="flex flex-wrap gap-2 mb-4">
                    ${blog.tags.map(tag => `<span class="px-3 py-1 bg-accent/10 text-accent rounded-full text-sm">${tag}</span>`).join('')}
                </div>
                <a href="#" class="text-accent hover:text-accent-dark font-medium view-blog" data-id="${blog.id}">Read More →</a>
            </div>
        `;
        
        // Add animation to make the new elements fade in
        div.style.opacity = '0';
        div.style.animation = 'fadeIn 0.5s forwards';
        
        return div;
    }

    // Load more blogs functionality
    const loadMoreButton = document.querySelector('#blog .text-center a');
    if (loadMoreButton) {
        loadMoreButton.addEventListener('click', (e) => {
            e.preventDefault();
            const blogGrid = document.querySelector('#blog .grid');
            if (blogGrid) {
                additionalBlogs.forEach(blog => {
                    const blogElement = createBlogElement(blog);
                    blogGrid.appendChild(blogElement);
                });
                
                // Add click event listeners to new blog links
                blogGrid.querySelectorAll('.view-blog').forEach(link => {
                    link.addEventListener('click', handleBlogClick);
                });
                
                // Hide the load more button after loading all posts
                loadMoreButton.style.display = 'none';
            }
        });
    }

    // Simulated blog post reading time calculation
    const blogExcerpts = document.querySelectorAll('#blog .text-gray-600');
    
    blogExcerpts.forEach(excerpt => {
        const text = excerpt.textContent;
        const wordCount = text.split(/\s+/).length;
        const readingTimeMinutes = Math.ceil(wordCount / 200); // Assuming average reading speed of 200 wpm
        
    });
    
    // Add click event listeners to blog post links
    document.querySelectorAll('.view-blog').forEach(link => {
        link.addEventListener('click', handleBlogClick);
    });

    function handleBlogClick(e) {
        e.preventDefault();
        const blogId = this.getAttribute('data-id');
        window.contentManager.showContent('blog', blogId);
    }
});
