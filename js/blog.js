document.addEventListener('DOMContentLoaded', () => {
    // Blog post data for a potential "Load More" functionality
    const additionalBlogPosts = [
        {
            title: "Reinforcement Learning: A Practical Introduction",
            excerpt: "Learn the basics of reinforcement learning and how it's being applied across different domains.",
            date: "January 15, 2023",
            readTime: "7 min read",
            tags: ["AI", "Machine Learning"],
            id: "blog4"
        },
        {
            title: "Setting Up Your Deep Learning Environment",
            excerpt: "A comprehensive guide to configuring your machine for deep learning with CUDA, PyTorch, and TensorFlow.",
            date: "December 3, 2022",
            readTime: "10 min read",
            tags: ["Deep Learning", "Setup"],
            id: "blog5"
        },
        {
            title: "Transfer Learning: Leveraging Pre-Trained Models",
            excerpt: "How to use transfer learning to achieve state-of-the-art results with limited data and computational resources.",
            date: "November 18, 2022",
            readTime: "8 min read",
            tags: ["Deep Learning", "Transfer Learning"],
            id: "blog6"
        }
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
                additionalBlogPosts.forEach(blog => {
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
        
        // Add reading time indicator if it doesn't exist yet
        if (!excerpt.nextElementSibling || !excerpt.nextElementSibling.classList.contains('reading-time')) {
            const readingTime = document.createElement('div');
            readingTime.className = 'text-xs text-gray-500 mb-4 reading-time';
            readingTime.innerHTML = `
                <svg class="inline-block h-4 w-4 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd" />
                </svg>
                ${readingTimeMinutes} min read
            `;
            excerpt.after(readingTime);
        }
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
