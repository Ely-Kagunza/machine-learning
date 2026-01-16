// File Upload Handler

// Handle file upload form submission
function handleFileUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('csv-file');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('❌ Please select a file', 'danger');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        showAlert('❌ Please select a CSV file', 'danger');
        return;
    }
    
    // Show loading state
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    // Upload file
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Upload failed');
            });
        }
        return response.json();
    })
    .then(result => {
        displayBatchResults(result);
        showAlert('✅ File uploaded and processed successfully!', 'success');
    })
    .catch(error => {
        showAlert(`❌ ${error.message}`, 'danger');
    })
    .finally(() => {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
        fileInput.value = ''; // Clear file input
    });
}

// Display batch results
function displayBatchResults(result) {
    const card = document.getElementById('batch-results-card');
    
    // Update summary
    document.getElementById('total-predictions').textContent = result.total_predictions;
    document.getElementById('goodware-count').textContent = result.goodware_count;
    document.getElementById('malware-count').textContent = result.malware_count;
    
    // Display metrics if available
    if (result.metrics) {
        const metricsSection = document.getElementById('metrics-section');
        metricsSection.style.display = 'block';
        
        document.getElementById('metric-auc').textContent = result.metrics.auc.toFixed(4);
        document.getElementById('metric-accuracy').textContent = (result.metrics.accuracy * 100).toFixed(2) + '%';
        document.getElementById('metric-precision').textContent = (result.metrics.precision * 100).toFixed(2) + '%';
        document.getElementById('metric-recall').textContent = (result.metrics.recall * 100).toFixed(2) + '%';
        
        // Confusion matrix
        const cm = result.metrics.confusion_matrix;
        document.getElementById('cm-tn').textContent = cm.true_negatives;
        document.getElementById('cm-fp').textContent = cm.false_positives;
        document.getElementById('cm-fn').textContent = cm.false_negatives;
        document.getElementById('cm-tp').textContent = cm.true_positives;
    } else {
        document.getElementById('metrics-section').style.display = 'none';
    }
    
    // Populate predictions table
    const tbody = document.getElementById('predictions-tbody');
    tbody.innerHTML = '';
    
    result.predictions.forEach(pred => {
        const row = document.createElement('tr');
        const badgeColor = pred.prediction === 1 ? 'danger' : 'success';
        const badgeText = pred.class;
        
        row.innerHTML = `
            <td>${pred.row}</td>
            <td><span class="badge bg-${badgeColor}">${badgeText}</span></td>
            <td>${pred.goodware_prob.toFixed(2)}%</td>
            <td>${pred.malware_prob.toFixed(2)}%</td>
            <td><strong>${pred.confidence.toFixed(2)}%</strong></td>
        `;
        tbody.appendChild(row);
    });
    
    // Show results card and scroll to it
    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
