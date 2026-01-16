// Malware Detection System - Application Logic

// Global state
let features = [];
let numericFeatures = [];
let categoricalFeatures = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Loading malware detection app...');
    
    // Fetch feature information
    fetchFeatures();
    
    // Attach event listeners
    document.getElementById('prediction-form').addEventListener('submit', handlePrediction);
    document.getElementById('load-goodware-btn').addEventListener('click', () => loadSampleData('goodware'));
    document.getElementById('load-malware-btn').addEventListener('click', () => loadSampleData('malware'));
    document.getElementById('load-random-btn').addEventListener('click', () => loadSampleData('random'));
});

// Fetch features from API
function fetchFeatures() {
    fetch('/api/features')
        .then(response => response.json())
        .then(data => {
            features = data.all_features;
            numericFeatures = data.numeric_features;
            categoricalFeatures = data.categorical_features;
            
            console.log(`Loaded ${features.length} features`);
            document.getElementById('feature-count').textContent = features.length;
            
            renderForm();
        })
        .catch(error => {
            console.error('Error fetching features:', error);
            showAlert('Error loading features from API', 'danger');
        });
}

// Render form dynamically
function renderForm() {
    const container = document.getElementById('features-container');
    container.innerHTML = '';
    
    // Numeric features
    if (numericFeatures.length > 0) {
        container.innerHTML += '<div class="col-12"><h6 class="feature-category">📊 Numeric Features</h6></div>';
        
        for (let i = 0; i < numericFeatures.length; i += 2) {
            const col = document.createElement('div');
            col.className = 'col-md-6';
            
            col.innerHTML += createInputField(numericFeatures[i], 'number');
            
            if (i + 1 < numericFeatures.length) {
                col.innerHTML += createInputField(numericFeatures[i + 1], 'number');
            }
            
            container.appendChild(col);
        }
    }
    
    // Categorical features
    if (categoricalFeatures.length > 0) {
        container.innerHTML += '<div class="col-12"><h6 class="feature-category">🏷️ Categorical Features</h6></div>';
        
        for (let i = 0; i < categoricalFeatures.length; i += 2) {
            const col = document.createElement('div');
            col.className = 'col-md-6';
            
            col.innerHTML += createSelectField(categoricalFeatures[i]);
            
            if (i + 1 < categoricalFeatures.length) {
                col.innerHTML += createSelectField(categoricalFeatures[i + 1]);
            }
            
            container.appendChild(col);
        }
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
                required
            >
        </div>
    `;
}

// Create select field
function createSelectField(featureName) {
    return `
        <div class="form-group">
            <label for="${featureName}" class="form-label">${featureName}</label>
            <select 
                class="form-control" 
                id="${featureName}" 
                name="${featureName}"
                required
            >
                <option value="">-- Select --</option>
                <option value="0">0</option>
                <option value="1">1</option>
            </select>
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
    
    // Convert numeric strings to numbers
    for (let key in inputData) {
        if (numericFeatures.includes(key)) {
            inputData[key] = parseFloat(inputData[key]);
        } else if (categoricalFeatures.includes(key)) {
            inputData[key] = parseInt(inputData[key]);
        }
    }
    
    console.log('Sending prediction request:', inputData);
    
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
        console.log('Prediction result:', result);
        displayResult(result);
        showAlert('✅ Prediction successful!', 'success');
    })
    .catch(error => {
        console.error('Prediction error:', error);
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
    document.getElementById('confidence').textContent = result.confidence.toFixed(2) + '%';
    document.getElementById('prob-goodware').textContent = 
        (result.probability_goodware * 100).toFixed(2) + '%';
    document.getElementById('prob-malware').textContent = 
        (result.probability_malware * 100).toFixed(2) + '%';
    
    // Show card
    card.style.display = 'block';
    
    // Scroll to result
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Load sample data
function loadSampleData(sampleType) {
    console.log(`Loading ${sampleType} sample data...`);
    
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
                    throw new Error(data.error || 'Failed to load sample');
                });
            }
            return response.json();
        })
        .then(result => {
            console.log('Sample data loaded:', result);
            
            // Fill form with sample data
            const sample = result.sample;
            for (let [key, value] of Object.entries(sample)) {
                const element = document.getElementById(key);
                if (element) {
                    element.value = value;
                }
            }
            
            // Show success message with sample name
            showAlert(`✓ ${result.sample_name} loaded (${result.features} features)`, 'success');
        })
        .catch(error => {
            console.error('Error loading sample:', error);
            showAlert(`❌ ${error.message}`, 'danger');
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

// API Health Check
function checkHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('Health check:', data);
            if (!data.model_loaded || !data.preprocessor_loaded) {
                showAlert('⚠️ Warning: Model or preprocessor not loaded', 'warning');
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
        });
}

// Check health on load
window.addEventListener('load', checkHealth);
