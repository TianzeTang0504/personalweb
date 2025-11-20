document.addEventListener('DOMContentLoaded', () => {
    initMusicPlayer();
    initSnakeGame();
});

/**
 * Music Player Widget Logic
 * Simulates playing state and toggles animation/icons
 */
function initMusicPlayer() {
    const btn = document.getElementById('musicToggleBtn');
    if(!btn) return;

    const widget = btn.closest('.glass');
    // Select visualizer bars only within this widget
    const visualizerBars = widget.querySelectorAll('.animate-pulse');

    let visualizerInterval = null;
    let rays = [];
    const audioManager = window.sharedAudio;

    // Initialize: Pause animation
    visualizerBars.forEach(b => b.style.animationPlayState = 'paused');

    // Create circular visualizer rays around the button
    function createSunburst() {
        const container = btn.parentElement; // The backdrop blur div
        const count = 24; // Number of rays
        const radius = 26; // Start distance from center (btn radius is 20px + padding)
        
        for(let i=0; i<count; i++) {
            const ray = document.createElement('div');
            ray.className = 'absolute bg-accent/80 pointer-events-none rounded-full transition-all duration-300 ease-out opacity-0';
            
            // Initial positioning
            const angle = (i / count) * 360;
            // ray width = 2px, height = variable
            ray.style.width = '2px';
            ray.style.height = '4px'; // Base height
            ray.style.top = '50%';
            ray.style.left = '50%';
            
            // Transform: rotate to angle, then push out by radius
            ray.style.transform = `translate(-50%, -50%) rotate(${angle}deg) translateY(-${radius}px)`;
            
            container.insertBefore(ray, btn);
            rays.push({
                el: ray,
                angle: angle,
                baseRadius: radius
            });
        }
    }

    createSunburst();

    function updateVisualizer() {
        rays.forEach(ray => {
            // Randomize height to simulate frequency data
            // Base height 4px, max extra 12px
            const extra = Math.random() * 16;
            ray.el.style.height = `${4 + extra}px`;
            
            // Adjust translateY to make it grow OUTWARDS
            // If height increases, we need to push the center further out so the inner edge stays put.
            // Default center is at radius. 
            // If height becomes 4+extra, the new center needs to be at radius + extra/2.
            const currentHeight = 4 + extra;
            const dist = ray.baseRadius + (extra / 2);
            ray.el.style.transform = `translate(-50%, -50%) rotate(${ray.angle}deg) translateY(-${dist}px)`;
            
            ray.el.style.opacity = 0.3 + (extra/20); // Brighter when louder
        });
    }

    function resetVisualizer() {
        rays.forEach(ray => {
            ray.el.style.height = '4px';
            // Reset position
            ray.el.style.transform = `translate(-50%, -50%) rotate(${ray.angle}deg) translateY(-${ray.baseRadius}px)`;
            ray.el.style.opacity = '0'; // Hide rays when paused
        });
    }

    // Subscribe to global audio state
    audioManager.subscribe((isPlaying) => {
        const container = btn.parentElement;
        
        if (isPlaying) {
            // Keep visible when playing by enforcing opacity: 1
            container.style.opacity = '1';
            
            // Switch to Pause Icon (Spinning)
            btn.innerHTML = `<svg class="w-4 h-4 animate-[spin_6s_linear_infinite]" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" /></svg>`;
            visualizerBars.forEach(b => b.style.animationPlayState = 'running');
            
            // Start Sunburst Animation
            if (visualizerInterval) clearInterval(visualizerInterval);
            visualizerInterval = setInterval(updateVisualizer, 200);
            updateVisualizer();
            
        } else {
            // Restore hover visibility
            container.style.opacity = '';

            // Switch to Play Icon
            btn.innerHTML = `<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd" /></svg>`;
            visualizerBars.forEach(b => b.style.animationPlayState = 'paused');
            
            if (visualizerInterval) {
                clearInterval(visualizerInterval);
                visualizerInterval = null;
            }
            resetVisualizer();
        }
    });

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        audioManager.toggle();
    });
}

/**
 * Snake Game Widget Logic
 */
function initSnakeGame() {
    const canvas = document.getElementById('gameCanvas');
    const startBtn = document.getElementById('gameStartBtn');
    const scoreEl = document.getElementById('scoreVal');
    const overlay = document.getElementById('gameOverlay');

    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    // Resize canvas to match CSS size
    function resize() {
        const rect = canvas.parentElement.getBoundingClientRect();
        // Set logical resolution lower for retro feel, or high for crispness?
        // Let's match 1:1 pixels for crisp rendering
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    // Initial resize
    resize(); 

    // Game State
    const gridSize = 15; // Size of snake squares
    let snake = [];
    let food = {x: 0, y: 0};
    let direction = 'right';
    let nextDirection = 'right';
    let score = 0;
    let gameLoop = null;
    let speed = 120;
    let cols, rows;

    function startGame() {
        resize(); // Ensure size is correct
        cols = Math.floor(canvas.width / gridSize);
        rows = Math.floor(canvas.height / gridSize);
        
        snake = [
            {x: Math.floor(cols/2), y: Math.floor(rows/2)},
            {x: Math.floor(cols/2)-1, y: Math.floor(rows/2)},
            {x: Math.floor(cols/2)-2, y: Math.floor(rows/2)}
        ];
        direction = 'right';
        nextDirection = 'right';
        score = 0;
        scoreEl.textContent = score;
        spawnFood();
        
        if (gameLoop) clearInterval(gameLoop);
        gameLoop = setInterval(update, speed);
        
        overlay.classList.add('hidden');
        // Focus window to capture keys if needed, though we listen globally
    }

    function spawnFood() {
        food = {
            x: Math.floor(Math.random() * cols),
            y: Math.floor(Math.random() * rows)
        };
        // Retry if food on snake
        for(let part of snake) {
            if(part.x === food.x && part.y === food.y) {
                spawnFood();
                break;
            }
        }
    }

    function update() {
        direction = nextDirection;
        const head = { ...snake[0] };

        if (direction === 'right') head.x++;
        else if (direction === 'left') head.x--;
        else if (direction === 'up') head.y--;
        else if (direction === 'down') head.y++;

        // Collision with walls
        if (head.x < 0 || head.x >= cols || head.y < 0 || head.y >= rows || checkSelfCollision(head)) {
            gameOver();
            return;
        }

        snake.unshift(head);

        // Eat food
        if (head.x === food.x && head.y === food.y) {
            score += 10;
            scoreEl.textContent = score;
            spawnFood();
            // Slightly increase speed
            // if(speed > 60) {
            //     clearInterval(gameLoop);
            //     speed -= 2;
            //     gameLoop = setInterval(update, speed);
            // }
        } else {
            snake.pop();
        }

        draw();
    }

    function checkSelfCollision(head) {
        for (let i = 0; i < snake.length; i++) {
            if (head.x === snake[i].x && head.y === snake[i].y) return true;
        }
        return false;
    }

    function draw() {
        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Optional: Draw Grid
        ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
        for(let i=0; i<cols; i++) {
            for(let j=0; j<rows; j++) {
                if((i+j)%2 === 0) ctx.fillRect(i*gridSize, j*gridSize, gridSize, gridSize);
            }
        }

        // Draw Snake
        snake.forEach((part, index) => {
            if (index === 0) {
                ctx.fillStyle = '#ffffff'; // Head
            } else {
                ctx.fillStyle = '#00ff9d'; // Body (Accent color)
            }
            
            // Add a little padding for "blocky" look
            const pad = 1;
            ctx.fillRect(
                part.x * gridSize + pad, 
                part.y * gridSize + pad, 
                gridSize - pad*2, 
                gridSize - pad*2
            );
        });

        // Draw Food
        ctx.fillStyle = '#ef4444'; // Red
        const pad = 2;
        ctx.fillRect(
            food.x * gridSize + pad, 
            food.y * gridSize + pad, 
            gridSize - pad*2, 
            gridSize - pad*2
        );
    }

    function gameOver() {
        clearInterval(gameLoop);
        overlay.classList.remove('hidden');
        overlay.querySelector('p').textContent = 'GAME OVER';
        overlay.querySelector('button').textContent = 'RESTART';
    }

    // Global Key Listeners
    document.addEventListener('keydown', (e) => {
        if (overlay.classList.contains('hidden')) { // Only when game is running
            const key = e.key;
            
            // Prevent scrolling
            if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight', ' '].includes(key)) {
                e.preventDefault();
            }

            if (key === 'ArrowUp' && direction !== 'down') nextDirection = 'up';
            else if (key === 'ArrowDown' && direction !== 'up') nextDirection = 'down';
            else if (key === 'ArrowLeft' && direction !== 'right') nextDirection = 'left';
            else if (key === 'ArrowRight' && direction !== 'left') nextDirection = 'right';
        }
    });

    startBtn.addEventListener('click', startGame);
}
