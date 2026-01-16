// Malware Detection System - Application Logic

// Global state
let features = [];
let numericFeatures = [];
let categoricalFeatures = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Fetch feature information
    fetchFeatures();
    
    // Attach event listeners
    document.getElementById('prediction-form').addEventListener('submit', handlePrediction);
    document.getElementById('load-goodware-btn').addEventListener('click', () => loadSampleData('goodware'));
    document.getElementById('load-malware-btn').addEventListener('click', () => loadSampleData('malware'));

    // Attach upload handler if present
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // Only attach random button if it exists
    const randomBtn = document.getElementById('load-random-btn');
    if (randomBtn) {
        randomBtn.addEventListener('click', () => loadSampleData('random'));
    }
});

// Fetch features from API
function fetchFeatures() {
    fetch('/api/features')
        .then(response => response.json())
        .then(data => {
            features = data.all_features;
            numericFeatures = data.numeric_features;
            categoricalFeatures = data.categorical_features;
            
            document.getElementById('feature-count').textContent = features.length;
            
            renderForm();
        })
        .catch(error => {
            showAlert('Error loading features from API', 'danger');
        });
}

// Render form dynamically with a responsive grid
function renderForm() {
    const container = document.getElementById('features-container');
    container.innerHTML = '';
    container.className = 'features-grid';

    // Numeric features header
    if (numericFeatures.length > 0) {
        const header = document.createElement('div');
        header.innerHTML = '<h6 class="feature-category">📊 Numeric Features</h6>';
        container.appendChild(header);

        numericFeatures.forEach((feat) => {
            const item = document.createElement('div');
            item.innerHTML = createInputField(feat, 'number');
            container.appendChild(item);
        });
    }

    // Categorical features header
    if (categoricalFeatures.length > 0) {
        const header = document.createElement('div');
        header.innerHTML = '<h6 class="feature-category">🏷️ Categorical Features</h6>';
        container.appendChild(header);

        categoricalFeatures.forEach((feat) => {
            const item = document.createElement('div');
            item.innerHTML = createSelectField(feat);
            container.appendChild(item);
        });
    }
}

// Create input field
function createInputField(featureName, type = 'number') {
    return `
        <div class="form-group">
            <label for="${featureName}" class="form-label">${featureName}</label>
            <input 
                type="${type}" 
                class="form-control" 
                id="${featureName}" 
                name="${featureName}" 
                placeholder="Enter value"
                step="any"
            >
        </div>
    `;
}

// Create select field
function createSelectField(featureName) {
    return `
        <div class="form-group">
            <label for="${featureName}" class="form-label">${featureName}</label>
            <input 
                type="number" 
                class="form-control" 
                id="${featureName}" 
                name="${featureName}" 
                placeholder="Enter numeric value"
                step="1"
            >
        </div>
    `;
}

// Handle prediction form submission
function handlePrediction(event) {
    event.preventDefault();
    
    // Show loading state
    const submitBtn = document.querySelector('button[type="submit"]');
    const submitText = document.getElementById('submit-text');
    const spinner = document.getElementById('loading-spinner');
    
    submitBtn.disabled = true;
    submitText.textContent = 'Analyzing...';
    spinner.style.display = 'inline-block';
    
    // Collect form data
    const formData = new FormData(document.getElementById('prediction-form'));
    const inputData = Object.fromEntries(formData);
    
    // Validate all required features are present
    const missingFeatures = features.filter(f => !(f in inputData));
    if (missingFeatures.length > 0) {
        showAlert(`❌ Please fill in all fields. Missing: ${missingFeatures.join(', ')}`, 'danger');
        submitBtn.disabled = false;
        submitText.textContent = '🔬 Analyze File';
        spinner.style.display = 'none';
        return;
    }
    
    // Convert numeric strings to numbers
    for (let key in inputData) {
        if (numericFeatures.includes(key)) {
            inputData[key] = parseFloat(inputData[key]);
        } else if (categoricalFeatures.includes(key)) {
            inputData[key] = parseInt(inputData[key]);
        }
    }
    
    // Send prediction request
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Prediction failed');
            });
        }
        return response.json();
    })
    .then(result => {
        displayResult(result);
        showAlert('✅ Analysis complete!', 'success');
    })
    .catch(error => {
        showAlert(`❌ ${error.message}`, 'danger');
    })
    .finally(() => {
        submitBtn.disabled = false;
        submitText.textContent = '🔬 Analyze File';
        spinner.style.display = 'none';
    });
}

// Display prediction result
function displayResult(result) {
    const card = document.getElementById('results-card');
    const header = document.getElementById('results-header');
    const alert = document.getElementById('result-alert');
    
    // Set header color
    header.className = `card-header bg-${result.color} text-white`;
    
    // Set alert class
    alert.className = `alert alert-${result.color}`;
    
    // Update result content
    document.getElementById('classification').textContent = result.classification;
    document.getElementById('status').textContent = result.status;
    const confidence = result.confidence;
    document.getElementById('confidence').textContent = confidence.toFixed(2) + '%';
    document.getElementById('prob-goodware').textContent = 
        (result.probability_goodware * 100).toFixed(2) + '%';
    document.getElementById('prob-malware').textContent = 
        (result.probability_malware * 100).toFixed(2) + '%';
    
    // Update confidence bar
    const confidenceBar = document.getElementById('confidence-bar');
    confidenceBar.style.width = confidence + '%';
    confidenceBar.textContent = confidence.toFixed(1) + '%';
    
    // Show card
    card.style.display = 'block';
    
    // Scroll to result
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Load sample data
function loadSampleData(sampleType) {
    // Show loading state
    const btn = document.getElementById(`load-${sampleType}-btn`);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    
    // Fetch sample data from API
    fetch(`/api/sample/${sampleType}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || `Failed to load sample (Status: ${response.status})`);
                });
            }
            return response.json();
        })
        .then(result => {
            // Fill form with sample data
            const sample = result.sample;
            let filledCount = 0;
            
            for (let [key, value] of Object.entries(sample)) {
                const element = document.getElementById(key);
                if (element) {
                    // Handle INPUT elements (all fields are now number inputs)
                    if (element.tagName === 'INPUT') {
                        if (element.type === 'number') {
                            // For integer fields (categorical), don't add decimals
                            if (categoricalFeatures.includes(key)) {
                                element.value = parseInt(value) || 0;
                            } else {
                                // For numeric fields, keep 2 decimal places
                                element.value = parseFloat(value).toFixed(2);
                            }
                        } else {
                            element.value = value;
                        }
                    }
                    filledCount++;
                }
            }
            
            // Scroll to form
            const formContainer = document.getElementById('features-container');
            if (formContainer) {
                formContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            // Show success message with sample name
            showAlert(`✓ ${result.sample_name} loaded (${filledCount}/${result.features} fields filled)`, 'success');
        })
        .catch(error => {
            showAlert(`❌ Error: ${error.message}`, 'danger');
        })
        .finally(() => {
            btn.disabled = false;
            btn.innerHTML = originalText;
        });
}

// Show alert message
function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}
