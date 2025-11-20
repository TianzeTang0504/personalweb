class GeekBackground {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d', { alpha: true });
        // Handle high DPI displays
        this.dpr = window.devicePixelRatio || 1;
        
        this.particles = [];
        this.mouse = { x: null, y: null };
        this.particleCount = 50; // Reduced count for cleaner look
        
        this.init();
    }

    init() {
        // Setup canvas
        this.canvas.id = 'geek-bg';
        this.canvas.style.position = 'fixed';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.zIndex = '-1';
        // Keep the gradient background
        this.canvas.style.background = 'radial-gradient(circle at 50% -20%, #1c3a33 0%, #050505 80%)';
        document.body.appendChild(this.canvas);

        // Resize handling
        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Mouse tracking
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.x;
            this.mouse.y = e.y;
        });

        window.addEventListener('mouseout', () => {
            this.mouse.x = null;
            this.mouse.y = null;
        });

        // Create particles
        this.createParticles();

        // Start animation
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth * this.dpr;
        this.canvas.height = window.innerHeight * this.dpr;
        
        this.ctx.scale(this.dpr, this.dpr);
        
        // Fix CSS size
        this.canvas.style.width = window.innerWidth + 'px';
        this.canvas.style.height = window.innerHeight + 'px';
        
        // Adjust particle count based on screen area (logic remains same but using window dimensions)
        const area = window.innerWidth * window.innerHeight;
        this.particleCount = Math.floor(area / 2000); 
    }

    createParticles() {
        this.particles = [];
        for (let i = 0; i < this.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.2, // Very slow movement
                vy: (Math.random() - 0.5) * 0.2, // Very slow movement
                size: Math.random() * 1.5 + 0.5, // Smaller particles
                alpha: Math.random() * 0.5 + 0.1, // Varying initial opacity
                alphaSpeed: (Math.random() - 0.5) * 0.005 // Pulse speed
            });
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw particles
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];

            // Move
            p.x += p.vx;
            p.y += p.vy;
            
            // Pulse opacity
            p.alpha += p.alphaSpeed;
            if (p.alpha <= 0.1 || p.alpha >= 0.6) {
                p.alphaSpeed *= -1;
            }

            // Bounce off edges
            if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;

            // Mouse interaction (Subtle repel)
            if (this.mouse.x != null) {
                const dx = p.x - this.mouse.x;
                const dy = p.y - this.mouse.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Reduced interaction range and force
                if (distance < 120) {
                    // Gentle push away
                    const force = (120 - distance) / 120;
                    // Add a minimal threshold to prevent "sticking" or jittering at dead center
                    if (force > 0.01) {
                         const moveX = (dx / distance) * force * 0.8; // Slightly stronger but smoother push
                         const moveY = (dy / distance) * force * 0.8;
                         
                         p.x += moveX; // Repel
                         p.y += moveY;
                    }
                }
            }

            // Draw particle (No lines!)
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(0, 255, 157, ${p.alpha})`;
            this.ctx.fill();
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GeekBackground();
});
