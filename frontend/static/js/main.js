// Function to format timestamps
function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

// Function to update alerts
function updateAlerts() {
    fetch('/alerts')
        .then(response => response.json())
        .then(alerts => {
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';
            
            alerts.forEach(alert => {
                const alertElement = document.createElement('div');
                alertElement.className = 'alert-item';
                alertElement.innerHTML = `
                    <div class="alert-content">
                        <strong>${alert.name || 'Unknown Event'}</strong>
                        <div class="alert-timestamp">${formatTimestamp(alert.timestamp)}</div>
                        <div>Similarity: ${alert.similarity || ''} Liveness: ${alert.liveness || ''}</div>
                    </div>
                `;
                container.appendChild(alertElement);
            });
        })
        .catch(error => console.error('Error fetching alerts:', error));
}

// Function to update captured frames
function updateCapturedFrames() {
    fetch('/captured_frames')
        .then(response => response.json())
        .then(frames => {
            const container = document.getElementById('captured-frames');
            container.innerHTML = '';
            
            frames.forEach(frame => {
                const frameElement = document.createElement('div');
                frameElement.className = 'col-md-3 captured-frame';
                frameElement.innerHTML = `
                    <img src="${frame.url}" 
                         alt="${frame.filename}"
                         onclick="showFrameModal('${frame.url}', '${frame.filename}', '${frame.timestamp}')">
                    <div class="captured-frame-info">
                        <div>${frame.filename}</div>
                        <small class="text-muted">${formatTimestamp(frame.timestamp)}</small>
                    </div>
                `;
                container.appendChild(frameElement);
            });
        })
        .catch(error => console.error('Error fetching captured frames:', error));
}

// Function to show frame in modal
function showFrameModal(url, filename, timestamp) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('frameModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'frameModal';
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title"></h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <img class="img-fluid" alt="">
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Update modal content
    modal.querySelector('.modal-title').textContent = filename;
    modal.querySelector('img').src = url;
    modal.querySelector('img').alt = filename;

    // Show modal
    new bootstrap.Modal(modal).show();
}

// Update data periodically
setInterval(updateAlerts, 5000);  // Every 5 seconds
setInterval(updateCapturedFrames, 10000);  // Every 10 seconds

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    updateAlerts();
    updateCapturedFrames();
}); 