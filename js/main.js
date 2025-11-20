document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobileMenuButton');
    const mobileMenu = document.getElementById('mobileMenu');
    const mainContent = document.getElementById('mainContent');
    const contentView = document.getElementById('contentView');
    const contentTopButton = document.getElementById('contentTopButton');

    // --- Music Button Logic (Global Nav) ---
    const navMusicBtn = document.getElementById('navMusicBtn');
    if (navMusicBtn && window.sharedAudio) {
        const musicIcon = navMusicBtn.querySelector('svg');
        
        // Handle Click
        navMusicBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            window.sharedAudio.toggle();
        });

        // Sync State
        window.sharedAudio.subscribe((isPlaying) => {
            if (isPlaying) {
                navMusicBtn.classList.remove('text-gray-400'); // Remove default color to ensure green works
                navMusicBtn.classList.add('text-accent');
                musicIcon.classList.add('rotating');
            } else {
                navMusicBtn.classList.remove('text-accent');
                navMusicBtn.classList.add('text-gray-400'); // Restore default color
                musicIcon.classList.remove('rotating');
            }
        });
    }
    // ---------------------------------------

    // 控制回到顶部按钮的显示和隐藏
    const toggleTopButton = () => {
        if (contentView && !contentView.classList.contains('hidden')) {
            if (window.scrollY > 300) {
                contentTopButton.classList.remove('hidden');
                contentTopButton.classList.add('flex'); // 确保使用 flex 布局以居中图标
            } else {
                contentTopButton.classList.add('hidden');
                contentTopButton.classList.remove('flex');
            }
        }
    };

    // 添加滚动事件监听器
    window.addEventListener('scroll', toggleTopButton);

    // 回到顶部按钮点击事件
    if (contentTopButton) {
        contentTopButton.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });

        // Close mobile menu when clicking a link
        const mobileLinks = mobileMenu.querySelectorAll('a');
        mobileLinks.forEach(link => {
            link.addEventListener('click', () => {
                mobileMenu.classList.add('hidden');
            });
        });
    }

    // Handle navigation links
    const handleNavLink = (e) => {
        e.preventDefault();
        const targetId = e.currentTarget.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        
        // 如果当前在内容视图中
        if (contentView && !contentView.classList.contains('hidden')) {
            // 先返回到主页面
            if (window.contentManager) {
                window.contentManager.hideContent();
            } else {
                contentView.classList.add('hidden');
                mainContent.classList.remove('hidden');
            }
            
            // 等待过渡动画完成后滚动到目标部分
            setTimeout(() => {
                if (targetId === 'home') {
                    window.scrollTo({
                        top: 0,
                        behavior: 'smooth'
                    });
                } else if (targetSection) {
                    targetSection.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }, 300);
        } else {
            // 如果已经在主页面，直接滚动到目标部分
            if (targetId === 'home') {
                window.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            } else if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    };

    // Add click event listeners to all navigation links including logo
    const allNavLinks = document.querySelectorAll('a[href^="#"]');
    allNavLinks.forEach(link => {
        link.addEventListener('click', handleNavLink);
    });

    // Active section highlighting
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');

    window.addEventListener('scroll', () => {
        // Only track scroll position when on main view
        if (mainContent.classList.contains('hidden')) {
            return;
        }
        
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop - 160) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('text-accent');
            link.classList.remove('bg-white/5');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('text-accent');
                link.classList.add('bg-white/5');
            }
        });
    });

    // === Typewriter Effect for Terminal ===
    const typeWriterElement = document.getElementById('typewriter-text');
    
    if (typeWriterElement) {
        const textLines = [
            { text: "class Scientist(Human):", class: "code-keyword" },
            { text: "    def __init__(self):", class: "" },
            { text: "        self.name = 'Tianze Tang'", class: "" },
            { text: "        self.role = 'AI Researcher'", class: "" },
            { text: "        self.lab = 'MedUniWien'", class: "" },
            { text: "        self.stack = ['PyTorch', 'Generative', 'LMM']", class: "" },
            { text: "    ", class: "" },
            { text: "    def research(self):", class: "code-function" },
            { text: "        # Medical Imaging & GenAI", class: "code-comment" },
            { text: "        return Innovation.create()", class: "" }
        ];

        let lineIndex = 0;
        let charIndex = 0;
        
        function type() {
            if (lineIndex < textLines.length) {
                const currentLine = textLines[lineIndex];
                
                // Create line div if starting new line
                if (charIndex === 0) {
                    const lineDiv = document.createElement('div');
                    if (currentLine.class) lineDiv.className = currentLine.class;
                    typeWriterElement.appendChild(lineDiv);
                }
                
                // Append character
                const currentDiv = typeWriterElement.lastElementChild;
                currentDiv.textContent += currentLine.text.charAt(charIndex);
                charIndex++;
                
                // Check if line finished
                if (charIndex >= currentLine.text.length) {
                    lineIndex++;
                    charIndex = 0;
                    setTimeout(type, 100); // Delay between lines
                } else {
                    setTimeout(type, 30); // Typing speed
                }
            }
        }
        
        // Start typing with a slight delay
        setTimeout(type, 1000);
    }
});
