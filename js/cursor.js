document.addEventListener('DOMContentLoaded', () => {
    // Check if device is touch-enabled or prefers reduced motion
    const isTouchDevice = window.matchMedia("(hover: none)").matches;
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (isTouchDevice || prefersReducedMotion) {
        return; // Do not initialize custom cursor
    }

    // Create cursor elements
    const cursorDot = document.createElement('div');
    cursorDot.classList.add('cursor-dot');
    
    const cursorOutline = document.createElement('div');
    cursorOutline.classList.add('cursor-outline');

    document.body.appendChild(cursorDot);
    document.body.appendChild(cursorOutline);

    // Enable custom cursor styles
    document.body.classList.add('custom-cursor-enabled');

    let cursorX = 0;
    let cursorY = 0;
    let outlineX = 0;
    let outlineY = 0;

    // Mouse move listener
    document.addEventListener('mousemove', (e) => {
        cursorX = e.clientX;
        cursorY = e.clientY;
        
        // Update dot position immediately
        cursorDot.style.left = `${cursorX}px`;
        cursorDot.style.top = `${cursorY}px`;
        
        // Outline will follow in animation loop
    });

    // Animation loop for smooth following
    const animate = () => {
        // Smooth interpolation (lerp)
        // Adjust 0.15 to change the delay speed (lower is slower/more delay)
        outlineX += (cursorX - outlineX) * 0.15;
        outlineY += (cursorY - outlineY) * 0.15;

        cursorOutline.style.left = `${outlineX}px`;
        cursorOutline.style.top = `${outlineY}px`;

        requestAnimationFrame(animate);
    };
    animate();

    // Interactive elements hover effect
    // Select all links, buttons, and specific interactive classes
    const interactiveElements = document.querySelectorAll('a, button, .btn-primary, .btn-secondary, input, textarea, .interactive');

    interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', () => {
            document.body.classList.add('hovering');
        });
        
        el.addEventListener('mouseleave', () => {
            document.body.classList.remove('hovering');
        });
    });

    // Click effect
    document.addEventListener('mousedown', () => {
        document.body.classList.add('clicking');
    });

    document.addEventListener('mouseup', () => {
        document.body.classList.remove('clicking');
    });
    
    // Also handle links added dynamically (optional observer could be better but this is simple)
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                         const newInteractive = node.querySelectorAll ? node.querySelectorAll('a, button, .btn-primary, .btn-secondary') : [];
                         if (node.matches && node.matches('a, button, .btn-primary, .btn-secondary')) {
                             addHoverListeners(node);
                         }
                         newInteractive.forEach(addHoverListeners);
                    }
                });
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    function addHoverListeners(el) {
        el.addEventListener('mouseenter', () => document.body.classList.add('hovering'));
        el.addEventListener('mouseleave', () => document.body.classList.remove('hovering'));
    }
});

