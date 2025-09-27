// ========================================
// GRIHA Green Rating Predictor - JavaScript
// Core functionality preserved
// ========================================

let currentSessionId = null;
let chatbotOpen = false;

// Add SVG gradient definition for progress ring
window.addEventListener('DOMContentLoaded', () => {
    const svg = document.querySelector('.progress-ring');
    if (svg) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.setAttribute('id', 'greenGradient');
        gradient.setAttribute('x1', '0%');
        gradient.setAttribute('y1', '0%');
        gradient.setAttribute('x2', '100%');
        gradient.setAttribute('y2', '0%');
        
        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('style', 'stop-color:#00D084;stop-opacity:1');
        
        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('style', 'stop-color:#00A86B;stop-opacity:1');
        
        gradient.appendChild(stop1);
        gradient.appendChild(stop2);
        defs.appendChild(gradient);
        svg.insertBefore(defs, svg.firstChild);
    }
});

// Form submission with negative value prevention
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Validate for negative values
    const formData = new FormData(this);
    for (let [key, value] of formData.entries()) {
        const input = document.getElementById(key);
        if (input && input.type === 'number' && parseFloat(value) < 0) {
            showAlert(`${key.replace('_', ' ').title()} cannot be negative. Please enter a positive value.`, 'warning');
            input.focus();
            return;
        }
    }
    
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnLoading = document.getElementById('btnLoading');
    const results = document.getElementById('results');
    const alerts = document.getElementById('alerts');
    const placeholder = document.getElementById('placeholder');
    const greenybotSection = document.getElementById('greenybotSection');
    
    // Show loading state
    predictBtn.disabled = true;
    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    alerts.innerHTML = '';
    greenybotSection.classList.add('hidden');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.warning) {
            alerts.innerHTML = `<div class="alert alert-warning fade-in">
                <i class="fas fa-exclamation-triangle"></i> ${data.message}
            </div>`;
            results.classList.add('hidden');
            placeholder.style.display = 'block';
            currentSessionId = null;
        } else if (data.success) {
            currentSessionId = data.session_id;
            
            // Show results
            const rating = parseInt(data.prediction);
            const stars = '⭐'.repeat(rating);
            
            document.getElementById('predictionText').textContent = rating;
            document.getElementById('starsDisplay').textContent = stars;
            
            // Animate progress circle
            const confidence = (data.confidence * 100).toFixed(1);
            const circumference = 2 * Math.PI * 85;
            const offset = circumference - (confidence / 100) * circumference;
            
            const progressCircle = document.getElementById('progressCircle');
            if (progressCircle) {
                progressCircle.style.strokeDashoffset = offset;
            }
            
            document.getElementById('confidenceBar').style.width = confidence + '%';
            document.getElementById('confidenceText').textContent = `${confidence}%`;
            
            // Show probability breakdown
            const probabilityList = document.getElementById('probabilityList');
            probabilityList.innerHTML = '';
            
            data.probabilities.forEach(prob => {
                const percentage = (prob.probability * 100).toFixed(1);
                const stars = '⭐'.repeat(parseInt(prob.label));
                
                probabilityList.innerHTML += `
                    <div class="probability-item">
                        <span>${stars} ${prob.label} Stars</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${percentage}%"></div>
                        </div>
                        <span><strong>${percentage}%</strong></span>
                    </div>
                `;
            });
            
            results.classList.remove('hidden');
            results.classList.add('fade-in');
            placeholder.style.display = 'none';
            
            // Scroll to results
            setTimeout(() => {
                results.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);
            
        } else if (data.error) {
            alerts.innerHTML = `<div class="alert alert-error fade-in">
                <i class="fas fa-times-circle"></i> ${data.error}
            </div>`;
            results.classList.add('hidden');
            placeholder.style.display = 'block';
            currentSessionId = null;
        }
    } catch (error) {
        alerts.innerHTML = `<div class="alert alert-error fade-in">
            <i class="fas fa-times-circle"></i> Network error: ${error.message}
        </div>`;
        results.classList.add('hidden');
        placeholder.style.display = 'block';
        currentSessionId = null;
    }
    
    // Reset button state
    predictBtn.disabled = false;
    btnText.classList.remove('hidden');
    btnLoading.classList.add('hidden');
});

// Get GreenyBot recommendations
async function getRecommendations() {
    if (!currentSessionId) {
        showAlert('Please get a prediction first!', 'warning');
        return;
    }
    
    const btn = document.getElementById('getRecommendationsBtn');
    const greenybotSection = document.getElementById('greenybotSection');
    const assessmentContent = document.getElementById('assessmentContent');
    
    btn.innerHTML = '<span class="spinner"></span> Getting Assessment...';
    btn.disabled = true;
    
    // Show the GreenyBot section
    greenybotSection.classList.remove('hidden');
    greenybotSection.classList.add('fade-in');
    
    // Scroll to the assessment section
    setTimeout(() => {
        greenybotSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
    
    try {
        const response = await fetch('/get_initial_assessment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const cleanedAssessment = cleanMarkdownText(data.assessment);
            
            assessmentContent.innerHTML = `
                <div class="assessment-item">
                    <h3><i class="fas fa-lightbulb"></i> Why This Rating?</h3>
                    <div class="content">${cleanedAssessment}</div>
                </div>
            `;
        } else {
            assessmentContent.innerHTML = `
                <div class="assessment-item">
                    <h3><i class="fas fa-exclamation-triangle"></i> Assessment Error</h3>
                    <div class="content">Unable to generate assessment. Please try again later.</div>
                </div>
            `;
        }
    } catch (error) {
        assessmentContent.innerHTML = `
            <div class="assessment-item">
                <h3><i class="fas fa-times-circle"></i> Network Error</h3>
                <div class="content">Failed to get assessment: ${error.message}</div>
            </div>
        `;
    }
    
    btn.innerHTML = '<i class="fas fa-check-circle"></i> <span>Assessment Complete</span>';
    btn.disabled = false;
}

// Get section details
async function getSection(sectionType) {
    if (!currentSessionId) {
        showAlert('Please get a prediction first!', 'warning');
        return;
    }
    
    const buttons = document.querySelectorAll('.action-btn');
    const clickedButton = event.target.closest('.action-btn');
    const originalText = clickedButton.innerHTML;
    
    clickedButton.innerHTML = '<span class="spinner"></span> Loading...';
    clickedButton.disabled = true;
    
    // Disable all buttons during loading
    buttons.forEach(btn => btn.disabled = true);
    
    try {
        const response = await fetch('/get_section', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                session_id: currentSessionId,
                section_type: sectionType 
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const sectionTitles = {
                'strengths': 'Key Strengths',
                'improvements': 'Areas for Improvement', 
                'benefits': 'Benefits of Improvement',
                'next_steps': 'Next Steps'
            };
            
            const sectionIcons = {
                'strengths': 'fas fa-trophy',
                'improvements': 'fas fa-tools',
                'benefits': 'fas fa-gem',
                'next_steps': 'fas fa-route'
            };
            
            const cleanedContent = cleanMarkdownText(data.content);
            
            // Add or update the section in assessment content
            const assessmentContent = document.getElementById('assessmentContent');
            const existingSection = document.getElementById(`section-${sectionType}`);
            
            const newSectionHTML = `
                <div class="assessment-item fade-in" id="section-${sectionType}">
                    <h3><i class="${sectionIcons[sectionType]}"></i> ${sectionTitles[sectionType]}</h3>
                    <div class="content">${cleanedContent}</div>
                </div>
            `;
            
            if (existingSection) {
                existingSection.outerHTML = newSectionHTML;
            } else {
                assessmentContent.innerHTML += newSectionHTML;
            }
            
            // Scroll to the new section
            setTimeout(() => {
                const section = document.getElementById(`section-${sectionType}`);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
            
        } else {
            showAlert('Failed to get section details. Please try again.', 'error');
        }
    } catch (error) {
        showAlert('Error getting section details: ' + error.message, 'error');
    }
    
    // Reset all buttons
    buttons.forEach(btn => {
        btn.disabled = false;
        if (btn === clickedButton) {
            btn.innerHTML = originalText;
        }
    });
}

// Clean markdown text and format properly
function cleanMarkdownText(text) {
    if (!text) return '';
    
    let cleaned = text
        .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/###\s*([^\n]+)/g, '<h4>$1</h4>')
        .replace(/##\s*([^\n]+)/g, '<h3>$1</h3>')
        .replace(/#\s*([^\n]+)/g, '<h2>$1</h2>')
        .replace(/^\d+\.\s+(.+)$/gm, '<p><strong>$1</strong></p>')
        .replace(/^-\s+(.+)$/gm, '<p>• $1</p>')
        .replace(/^\*\s+(.+)$/gm, '<p>• $1</p>')
        .replace(/\n\n+/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .trim();
    
    // Wrap in paragraph tags if not already wrapped
    if (!cleaned.startsWith('<')) {
        cleaned = '<p>' + cleaned + '</p>';
    }
    
    // Clean up empty paragraphs and extra breaks
    cleaned = cleaned
        .replace(/<p><\/p>/g, '')
        .replace(/<p><br>/g, '<p>')
        .replace(/<br><\/p>/g, '</p>')
        .replace(/<br><br>/g, '<br>');
    
    return cleaned;
}

// Chatbot functions for additional queries
function toggleChatbot() {
    const widget = document.getElementById('chatbotWidget');
    chatbotOpen = !chatbotOpen;
    widget.style.display = chatbotOpen ? 'flex' : 'none';
    
    if (chatbotOpen) {
        document.getElementById('chatbotInput').focus();
    }
}

function handleChatbotKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

async function sendChatMessage() {
    if (!currentSessionId) {
        addBotMessage("Error", "Please get a GRIHA rating prediction first before asking questions!");
        return;
    }

    const input = document.getElementById('chatbotInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    input.value = '';
    
    // Add user message
    addUserMessage(question);
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                question: question
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (data.success) {
            const cleanedResponse = cleanMarkdownText(data.response);
            addBotMessage("GreenyBot", cleanedResponse);
        } else {
            addBotMessage("Error", data.error || "Sorry, I couldn't process your question.");
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addBotMessage("Error", "Network error: " + error.message);
    }
}

function addUserMessage(message) {
    const messagesContainer = document.getElementById('chatbotMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    
    const now = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-content">${escapeHtml(message)}</div>
        <div class="message-time">${now}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addBotMessage(title, content) {
    const messagesContainer = document.getElementById('chatbotMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    
    const now = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <strong>${title}</strong>
            ${content}
        </div>
        <div class="message-time">${now}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chatbotMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typing-indicator-' + Date.now();
    
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <span class="spinner"></span> GreenyBot is typing...
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return typingDiv.id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// Prevent negative values in number inputs
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        if (parseFloat(this.value) < 0) {
            this.value = 0;
            this.style.borderColor = '#EF4444';
            setTimeout(() => {
                this.style.borderColor = '';
            }, 1000);
        }
    });
});

// Utility function to show alerts
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} fade-in`;
    
    const icons = {
        warning: 'fas fa-exclamation-triangle',
        error: 'fas fa-times-circle',
        success: 'fas fa-check-circle',
        info: 'fas fa-info-circle'
    };
    
    alertDiv.innerHTML = `
        <i class="${icons[type] || icons.info}"></i>
        <span>${message}</span>
    `;
    
    alertsContainer.innerHTML = '';
    alertsContainer.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        alertDiv.style.opacity = '0';
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 300);
    }, 5000);
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add smooth scroll behavior to all internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add intersection observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
        }
    });
}, observerOptions);

// Observe all glass cards
document.querySelectorAll('.glass-card').forEach(card => {
    observer.observe(card);
});

// Initialize tooltips for form inputs
document.querySelectorAll('.form-group label').forEach(label => {
    label.setAttribute('title', 'Enter the ' + label.textContent.toLowerCase());
});

// Add loading state to all buttons
document.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', function() {
        if (!this.disabled) {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 100);
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    }
    
    // Escape to close chatbot
    if (e.key === 'Escape' && chatbotOpen) {
        toggleChatbot();
    }
});

// Add service worker for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment to enable service worker
        // navigator.serviceWorker.register('/sw.js')
        //     .then(registration => console.log('SW registered'))
        //     .catch(err => console.log('SW registration failed'));
    });
}

// Performance monitoring
window.addEventListener('load', () => {
    if (window.performance) {
        const perfData = window.performance.timing;
        const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
        console.log(`Page loaded in ${pageLoadTime}ms`);
    }
});

// Console welcome message
console.log('%c🌿 GRIHA Green Rating Predictor', 'color: #00D084; font-size: 24px; font-weight: bold;');
console.log('%cPowered by AI for Sustainable Building Intelligence', 'color: #666; font-size: 14px;');
console.log('%c⚡ Ready for predictions!', 'color: #0066FF; font-size: 16px; font-weight: bold;');