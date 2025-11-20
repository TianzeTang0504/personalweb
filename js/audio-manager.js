/**
 * Shared Audio Manager
 * Manages a single Audio instance across the entire application
 * to ensure consistent playback state between Sidebar and Floating Player.
 */
class SharedAudioManager {
    constructor() {
        if (SharedAudioManager.instance) {
            return SharedAudioManager.instance;
        }
        SharedAudioManager.instance = this;

        this.audio = new Audio('assets/Stardew_Valley_Overture.mp3');
        this.audio.loop = true;
        this.audio.volume = 0.5;
        this.isPlaying = false;

        // Event listeners for state synchronization
        this.listeners = [];
    }

    play() {
        if (this.isPlaying) return Promise.resolve();
        
        return this.audio.play().then(() => {
            this.isPlaying = true;
            this.notifyListeners();
        }).catch(err => {
            console.warn('Audio play failed:', err);
            // Don't update state if play failed
        });
    }

    pause() {
        if (!this.isPlaying) return;
        
        this.audio.pause();
        this.isPlaying = false;
        this.notifyListeners();
    }

    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    // Subscribe to state changes
    subscribe(callback) {
        this.listeners.push(callback);
        // Immediately notify current state
        callback(this.isPlaying);
        return () => {
            this.listeners = this.listeners.filter(cb => cb !== callback);
        };
    }

    notifyListeners() {
        this.listeners.forEach(cb => cb(this.isPlaying));
    }
}

// Export global instance
window.sharedAudio = new SharedAudioManager();


