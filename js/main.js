document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobileMenuButton');
    const mobileMenu = document.getElementById('mobileMenu');
    const mainContent = document.getElementById('mainContent');
    const contentView = document.getElementById('contentView');
    const contentTopButton = document.getElementById('contentTopButton');

    // 控制回到顶部按钮的显示和隐藏
    const toggleTopButton = () => {
        if (contentView && !contentView.classList.contains('hidden')) {
            if (window.scrollY > 300) {
                contentTopButton.classList.remove('hidden');
            } else {
                contentTopButton.classList.add('hidden');
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
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('text-accent');
            }
        });
    });
});
