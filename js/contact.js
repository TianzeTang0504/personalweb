document.addEventListener('DOMContentLoaded', () => {
    const contactForm = document.getElementById('contactForm');
    const formSuccess = document.getElementById('formSuccess');
    
    if (contactForm) {
        contactForm.addEventListener('submit', handleFormSubmit);
    }

    function handleFormSubmit(e) {
        e.preventDefault();
        
        // Get form elements
        const firstName = document.getElementById('firstName');
        const lastName = document.getElementById('lastName');
        const email = document.getElementById('email');
        const subject = document.getElementById('subject');
        const message = document.getElementById('message');
        const submitButton = document.getElementById('submitButton');
        
        // Basic validation
        if (!validateForm(firstName, lastName, email, message)) {
            return;
        }
        
        // Disable submit button and show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Sending...
        `;
        
        // Simulate form submission with a delay
        setTimeout(() => {
            // In a real application, you would send the data to your backend here
            
            // Reset form and show success message
            contactForm.reset();
            
            // Hide the form and show success message
            contactForm.style.display = 'none';
            if (formSuccess) {
                formSuccess.classList.remove('hidden');
            }
            
            // Reset form after 5 seconds
            setTimeout(() => {
                contactForm.style.display = 'block';
                if (formSuccess) {
                    formSuccess.classList.add('hidden');
                }
                submitButton.disabled = false;
                submitButton.innerHTML = 'Send Message';
            }, 5000);
        }, 1500);
    }

    function validateForm(firstName, lastName, email, message) {
        // Reset previous error states
        clearErrors();
        
        let isValid = true;
        
        // Validate email
        if (!email.value || !isValidEmail(email.value)) {
            showError(email, 'Please enter a valid email address');
            isValid = false;
        }
        
        // Validate message
        if (!message.value.trim()) {
            showError(message, 'Please enter a message');
            isValid = false;
        }
        
        return isValid;
    }

    function showError(input, message) {
        input.classList.add('border-red-500');
        
        const errorElement = document.createElement('p');
        errorElement.className = 'text-red-500 text-xs mt-1 error-message';
        errorElement.textContent = message;
        
        input.parentNode.appendChild(errorElement);
    }

    function clearErrors() {
        // Remove all error classes
        const inputs = contactForm.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.classList.remove('border-red-500');
        });
        
        // Remove all error messages
        const errorMessages = contactForm.querySelectorAll('.error-message');
        errorMessages.forEach(error => {
            error.remove();
        });
    }

    function isValidEmail(email) {
        const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return regex.test(email);
    }
});
