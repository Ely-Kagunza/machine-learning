// Malware Detection System - Application Logic

// Global state
let features = [];
let numericFeatures = [];
let categoricalFeatures = [];

// Sample data for testing
const SAMPLE_GOODWARE = {
    'API_Count': 145,
    'String_Count': 89,
    'Entropy': 6.2,
    'Legitimate_API': 1,
    'Suspicious_String': 0,
    'Packed': 0,
    'Debug_Info': 1,
    'Import_Table': 1,
    'Export_Table': 0,
    'Resource_Section': 1,
    'Text_Section': 1,
    'Data_Section': 1,
    'Section_Count': 5,
    'Virtual_Size': 245760,
    'Raw_Size': 258048,
    'Timestamp': 1356998400,
    'Machine_Type': 332,
    'Subsystem': 3,
    'Characteristics': 258,
    'Header_Size': 512,
    'Code_Section_Entropy': 5.8,
    'Data_Section_Entropy': 4.2,
    'Imported_DLLs': 8,
    'Exported_Functions': 0
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Loading malware detection app...');
    
    // Fetch feature information
    fetchFeatures();
    
    // Attach event listeners
    document.getElementById('prediction-form').addEventListener('submit', handlePrediction);
    document.getElementById('load-sample-btn').addEventListener('click', loadSampleData);
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
function loadSampleData() {
    console.log('Loading sample goodware data...');
    
    for (let [key, value] of Object.entries(SAMPLE_GOODWARE)) {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
        }
    }
    
    showAlert('✓ Sample goodware data loaded', 'info');
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
