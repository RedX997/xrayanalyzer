document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const loadingContainer = document.getElementById('loadingContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const resultModal = document.getElementById('resultModal');
    const closeModal = document.getElementById('closeModal');
    const newScanBtn = document.getElementById('newScanBtn');
    const resultText = document.getElementById('resultText');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const navLinks = document.querySelectorAll('.footer-links a');

    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Drag and Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);

    // File Input Handler
    fileInput.addEventListener('change', handleFileSelect);

    // Button Handlers
    analyzeBtn.addEventListener('click', analyzeImage);
    cancelBtn.addEventListener('click', resetUpload);
    closeModal.addEventListener('click', () => resultModal.style.display = 'none');
    newScanBtn.addEventListener('click', () => {
        resultModal.style.display = 'none';
        resetUpload();
    });

    // Touch Event Handlers
    let touchStartY = 0;
    let touchEndY = 0;

    dropZone.addEventListener('touchstart', (e) => {
        touchStartY = e.touches[0].clientY;
    }, false);

    dropZone.addEventListener('touchmove', (e) => {
        touchEndY = e.touches[0].clientY;
    }, false);

    dropZone.addEventListener('touchend', (e) => {
        const touchDiff = touchStartY - touchEndY;
        if (Math.abs(touchDiff) < 10) { // If it's a tap rather than a scroll
            document.getElementById('fileInput').click();
        }
    }, false);

    // Prevent zoom on double tap
    document.addEventListener('dblclick', (e) => {
        e.preventDefault();
    }, { passive: false });

    // Handle orientation change
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            if (resultModal.style.display === 'block') {
                const modalContent = document.querySelector('.modal-content');
                modalContent.style.maxHeight = `${window.innerHeight * 0.9}px`;
            }
        }, 100);
    });

    // Prevent default drag behaviors
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when dragging over it
    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    // Handle dropped files
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle file selection
    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                // For mobile devices, ensure the file is properly oriented
                if (file.type.startsWith('image/')) {
                    const img = new Image();
                    img.onload = function() {
                        displayPreview(file);
                    };
                    img.src = URL.createObjectURL(file);
                } else {
                    displayPreview(file);
                }
            }
        }
    }

    // Validate file type and size
    function validateFile(file) {
        const validTypes = ['image/jpeg', 'image/png'];
        const maxSize = 5 * 1024 * 1024; // 5MB

        if (!validTypes.includes(file.type)) {
            showError('Please upload a JPEG or PNG image.');
            return false;
        }

        if (file.size > maxSize) {
            showError('File size should be less than 5MB.');
            return false;
        }

        return true;
    }

    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }

    // Reset upload state
    function resetUpload() {
        fileInput.value = '';
        imagePreview.src = '';
        dropZone.style.display = 'block';
        previewContainer.style.display = 'none';
        loadingContainer.style.display = 'none';
    }

    // Show error message
    function showError(message) {
        // You can implement a toast notification here
        alert(message);
    }

    // Analyze image
    async function analyzeImage() {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            previewContainer.style.display = 'none';
            loadingContainer.style.display = 'block';

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResults(data);
            } else {
                throw new Error(data.error || 'An error occurred during analysis');
            }
        } catch (error) {
            showError(error.message);
            previewContainer.style.display = 'block';
        } finally {
            loadingContainer.style.display = 'none';
        }
    }

    // Display results in modal
    function displayResults(data) {
        const confidence = data.confidence;
        const prediction = data.prediction;
        
        resultText.textContent = `Diagnosis: ${prediction}`;
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `Confidence: ${confidence}%`;

        const color = prediction.toLowerCase() === 'normal' ? 
            'var(--success-color)' : 'var(--danger-color)';
        confidenceBar.style.backgroundColor = color;

        resultModal.style.display = 'block';
        resultModal.style.opacity = '0';
        
        // Adjust modal position for mobile
        const modalContent = document.querySelector('.modal-content');
        modalContent.style.maxHeight = `${window.innerHeight * 0.9}px`;
        
        setTimeout(() => {
            resultModal.style.opacity = '1';
        }, 50);
    }

    // Add animation to feature cards on scroll
    const featureCards = document.querySelectorAll('.feature-card');
    const observerOptions = {
        threshold: 0.2
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    featureCards.forEach(card => {
        observer.observe(card);
    });

    // Add touch feedback
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('touchstart', () => {
            button.style.opacity = '0.7';
        });
        
        button.addEventListener('touchend', () => {
            button.style.opacity = '1';
        });
    });
});
