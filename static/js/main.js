// let socket = io();
// let streaming = false;
// let statsDiv = document.getElementById('stats');
// let video = document.getElementById('video');
// let startBtn = document.getElementById('startBtn');
// let mediaStream = null;
// let connectionStatus = document.createElement('div');
// let frameInterval = null;
//
// // Initialize connection status indicator
// connectionStatus.id = 'connection-status';
// connectionStatus.style.cssText = `
//     position: fixed;
//     top: 10px;
//     right: 10px;
//     padding: 8px 16px;
//     border-radius: 4px;
//     font-weight: bold;
//     z-index: 1000;
//     transition: all 0.3s ease;
// `;
// document.body.appendChild(connectionStatus);
//
// // Socket connection handlers
// socket.on('connect', function() {
//     console.log('Connected to server');
//     updateConnectionStatus('Connected', 'green');
// });
//
// socket.on('disconnect', function() {
//     console.log('Disconnected from server');
//     updateConnectionStatus('Disconnected', 'red');
//     if (streaming) {
//         stopStreaming();
//         startBtn.innerText = 'Start AI Detection';
//         streaming = false;
//     }
// });
//
// socket.on('connect_error', function(error) {
//     console.error('Connection error:', error);
//     updateConnectionStatus('Connection Error', 'orange');
// });
//
// // Update connection status indicator
// function updateConnectionStatus(status, color) {
//     connectionStatus.textContent = status;
//     connectionStatus.style.backgroundColor = color;
//     connectionStatus.style.color = 'white';
// }
//
// function toggleStreaming() {
//     if (!socket.connected) {
//         alert('Please wait for server connection before starting detection.');
//         return;
//     }
//
//     if (!streaming) {
//         startStreaming();
//     } else {
//         stopStreaming();
//     }
// }
//
// function startStreaming() {
//     navigator.mediaDevices.getUserMedia({
//         video: {
//             width: { ideal: 640 },
//             height: { ideal: 480 },
//             facingMode: 'user'
//         },
//         audio: false
//     })
//     .then(stream => {
//         video.srcObject = stream;
//         mediaStream = stream;
//         streaming = true;
//         startBtn.innerText = 'Stop AI Detection';
//
//         // Wait for video to load before starting frame capture
//         video.addEventListener('loadedmetadata', () => {
//             console.log('Video loaded, starting frame capture');
//             frameInterval = setInterval(sendFrame, 1000);
//         });
//
//         updateStats('Starting detection...', '?', 'Initializing', 'OK', 'OK');
//     })
//     .catch(error => {
//         console.error('Error accessing camera:', error);
//         let errorMessage = 'Camera access denied or not available';
//
//         if (error.name === 'NotFoundError') {
//             errorMessage = 'No camera found on this device';
//         } else if (error.name === 'NotAllowedError') {
//             errorMessage = 'Camera permission denied. Please allow camera access and try again.';
//         } else if (error.name === 'NotReadableError') {
//             errorMessage = 'Camera is being used by another application';
//         } else if (error.name === 'OverconstrainedError') {
//             errorMessage = 'Camera does not meet the required constraints';
//         }
//
//         alert(errorMessage);
//         updateStats('Camera Error', '?', 'Error', 'Error', 'Error');
//     });
// }
//
// function stopStreaming() {
//     streaming = false;
//     startBtn.innerText = 'Start AI Detection';
//
//     if (frameInterval) {
//         clearInterval(frameInterval);
//         frameInterval = null;
//     }
//
//     if (mediaStream) {
//         mediaStream.getTracks().forEach(track => {
//             track.stop();
//             console.log('Camera track stopped');
//         });
//         mediaStream = null;
//     }
//
//     if (video.srcObject) {
//         video.srcObject = null;
//     }
//
//     updateStats('Detection Stopped', '?', 'Stopped', 'Stopped', 'Stopped');
// }
//
// function sendFrame() {
//     if (!streaming || !socket.connected) {
//         console.log('Not sending frame: streaming =', streaming, 'connected =', socket.connected);
//         return;
//     }
//
//     try {
//         const canvas = document.createElement('canvas');
//         canvas.width = video.videoWidth || 640;
//         canvas.height = video.videoHeight || 480;
//
//         const ctx = canvas.getContext('2d');
//         ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
//
//         let imageData = canvas.toDataURL('image/jpeg', 0.8);
//         socket.emit('frame', { 'image': imageData });
//
//         console.log('Frame sent to server');
//
//     } catch (error) {
//         console.error('Error capturing frame:', error);
//         updateStats('Frame Capture Error', '?', 'Error', 'Error', 'Error');
//     }
// }
//
// // Handle AI detection results
// socket.on('ai_result', function(data) {
//     console.log('Received AI result:', data);
//
//     if (data.error) {
//         console.error('Detection error:', data.error);
//         updateStats('Detection Error', data.error, 'Error', 'Error', 'Error');
//         return;
//     }
//
//     // Update UI with detection results
//     const suspicionPercent = Math.round((data.suspicion || 0) * 100);
//     const detectedClass = data.detected_class || '?';
//     const headPose = data.head_pose_flag ? '⚠️ Warning' : '✅ OK';
//     const audio = data.audio_flag ? '⚠️ Warning' : '✅ OK';
//     const deepfake = data.deepfake_flag ? '⚠️ Warning' : '✅ OK';
//
//     updateStats(suspicionPercent + '%', detectedClass, headPose, audio, deepfake);
// });
//
// // Helper function to update stats display
// function updateStats(suspicion, aiGuess, headPose, audio, deepfake) {
//     statsDiv.innerHTML = `
//         <div style="display: grid; gap: 8px; font-family: monospace;">
//             <div><strong>Suspicion:</strong> <span style="color: ${getSuspicionColor(suspicion)}">${suspicion}</span></div>
//             <div><strong>AI Top Guess:</strong> ${aiGuess}</div>
//             <div><strong>Head Pose:</strong> ${headPose}</div>
//             <div><strong>Audio:</strong> ${audio}</div>
//             <div><strong>Deepfake:</strong> ${deepfake}</div>
//         </div>
//     `;
// }
//
// // Get color based on suspicion level
// function getSuspicionColor(suspicion) {
//     if (typeof suspicion === 'string') return '#666';
//
//     const percent = parseInt(suspicion);
//     if (percent >= 70) return '#ff4444';      // High suspicion - red
//     if (percent >= 40) return '#ff8800';      // Medium suspicion - orange
//     return '#44ff44';                         // Low suspicion - green
// }
//
// // Handle page visibility changes
// document.addEventListener('visibilitychange', function() {
//     if (document.hidden && streaming) {
//         console.log('Page hidden, pausing detection');
//         if (frameInterval) {
//             clearInterval(frameInterval);
//             frameInterval = null;
//         }
//     } else if (!document.hidden && streaming && !frameInterval) {
//         console.log('Page visible, resuming detection');
//         frameInterval = setInterval(sendFrame, 1000);
//     }
// });
//
// // Handle page unload
// window.addEventListener('beforeunload', function() {
//     if (streaming) {
//         stopStreaming();
//     }
//     socket.disconnect();
// });
//
// // Initialize connection status
// updateConnectionStatus('Connecting...', 'orange');
//
// // Periodically check connection status
// setInterval(() => {
//     if (!socket.connected) {
//         updateConnectionStatus('Disconnected', 'red');
//     }
// }, 5000);




















// static/js/main.js

let socket = io();
let streaming = false;
let statsDiv = document.getElementById('stats');
let video = document.getElementById('video');
let startBtn = document.getElementById('startBtn');
let mediaStream = null;
let connectionStatus = document.createElement('div');
let frameInterval = null;

// Frame send cadence: adjust based on server performance (lower = more FPS)
const SEND_EVERY_MS = 500; // ~2 FPS default for CPU YOLOv8n [web:102]

// Initialize connection status indicator
connectionStatus.id = 'connection-status';
connectionStatus.style.cssText = `
    position: fixed;
    top: 10px;
    right: 10px;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
    z-index: 1000;
    transition: all 0.3s ease;
`;
document.body.appendChild(connectionStatus);

// Socket connection handlers
socket.on('connect', function() {
    console.log('Connected to server');
    updateConnectionStatus('Connected', 'green');
}); // [web:127]

socket.on('disconnect', function() {
    console.log('Disconnected from server');
    updateConnectionStatus('Disconnected', 'red');
    if (streaming) {
        stopStreaming();
        startBtn.innerText = 'Start AI Detection';
        streaming = false;
    }
}); // [web:127]

socket.on('connect_error', function(error) {
    console.error('Connection error:', error);
    updateConnectionStatus('Connection Error', 'orange');
}); // [web:127]

// Update connection status indicator
function updateConnectionStatus(status, color) {
    connectionStatus.textContent = status;
    connectionStatus.style.backgroundColor = color;
    connectionStatus.style.color = 'white';
} // [web:127]

// Hook to a button onclick in HTML: <button id="startBtn" onclick="toggleStreaming()">...</button>
function toggleStreaming() {
    if (!socket.connected) {
        alert('Please wait for server connection before starting detection.');
        return;
    }
    if (!streaming) {
        startStreaming();
    } else {
        stopStreaming();
    }
} // [web:127]

function startStreaming() {
    navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
        },
        audio: false
    })
    .then(stream => {
        video.srcObject = stream;
        mediaStream = stream;
        streaming = true;
        startBtn.innerText = 'Stop AI Detection';

        // Wait for video to load before starting frame capture
        video.addEventListener('loadedmetadata', () => {
            console.log('Video loaded, starting frame capture');
            frameInterval = setInterval(sendFrame, SEND_EVERY_MS);
        }, { once: true });

        updateStats('Starting detection...', '?', 'Initializing', 'OK', 'OK', [], [], '?');
    })
    .catch(error => {
        console.error('Error accessing camera:', error);
        let errorMessage = 'Camera access denied or not available';

        if (error.name === 'NotFoundError') {
            errorMessage = 'No camera found on this device';
        } else if (error.name === 'NotAllowedError') {
            errorMessage = 'Camera permission denied. Please allow camera access and try again.';
        } else if (error.name === 'NotReadableError') {
            errorMessage = 'Camera is being used by another application';
        } else if (error.name === 'OverconstrainedError') {
            errorMessage = 'Camera does not meet the required constraints';
        }

        alert(errorMessage);
        updateStats('Camera Error', '?', 'Error', 'Error', 'Error', [], [], '?');
    });
} // [web:102]

function stopStreaming() {
    streaming = false;
    startBtn.innerText = 'Start AI Detection';

    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            track.stop();
            console.log('Camera track stopped');
        });
        mediaStream = null;
    }

    if (video.srcObject) {
        video.srcObject = null;
    }

    updateStats('Detection Stopped', '?', 'Stopped', 'Stopped', 'Stopped', [], [], '?');
} // [web:102]

// Capture current video frame and send as base64 JPEG to server via Socket.IO
function sendFrame() {
    if (!streaming || !socket.connected) {
        return;
    }

    try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        const ctx = canvas.getContext('2d');
        // Optional: small border fill to reduce edge artifacts in JPEG
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Use moderate JPEG quality to keep bandwidth reasonable
        let imageData = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('frame', { 'image': imageData });
    } catch (error) {
        console.error('Error capturing frame:', error);
        updateStats('Frame Capture Error', '?', 'Error', 'Error', 'Error', [], [], '?');
    }
} // [web:102]

// Handle AI detection results
socket.on('ai_result', function(data) {
    console.log('Received AI result:', data);

    if (data.error) {
        console.error('Detection error:', data.error);
        updateStats('Detection Error', data.error, 'Error', 'Error', 'Error', [], [], '?');
        return;
    }

    // Legacy/topline fields
    const suspicionPercent = Math.round((data.suspicion || 0) * 100);
    const detectedClass = data.detected_class || '?';
    const headPose = data.head_pose_flag ? '⚠️ Warning' : '✅ OK';
    const audio = data.audio_flag ? '⚠️ Warning' : '✅ OK';
    const deepfake = data.deepfake_flag ? '⚠️ Warning' : '✅ OK';

    // Extended fields from YOLO pipeline
    const persons = Array.isArray(data.persons) ? data.persons : [];
    const devices = Array.isArray(data.devices) ? data.devices : [];
    const audioRms = (typeof data.audio_rms === 'number') ? data.audio_rms.toFixed(4) : '?';

    updateStats(suspicionPercent + '%', detectedClass, headPose, audio, deepfake, persons, devices, audioRms);
}); // [web:127]

// Helper to build readable summaries for people/devices
function summarizePeople(persons) {
    if (!persons.length) return 'None';
    return persons.map(p => {
        const id = (p.id !== null && p.id !== undefined) ? `#${p.id} ` : '';
        const state = p.state || 'NA';
        const conf = (typeof p.conf === 'number') ? ` (${(p.conf*100).toFixed(1)}%)` : '';
        return `${id}${state}${conf}`;
    }).join(', ');
} // [web:127]

function summarizeDevices(devices) {
    if (!devices.length) return 'None';
    return devices.map(d => {
        const name = d.label || 'device';
        const conf = (typeof d.conf === 'number') ? ` (${(d.conf*100).toFixed(1)}%)` : '';
        return `${name}${conf}`;
    }).join(', ');
} // [web:127]

// Helper function to update stats display
function updateStats(suspicion, aiGuess, headPose, audio, deepfake, persons, devices, audioRms) {
    const peopleInfo = summarizePeople(persons || []);
    const deviceInfo = summarizeDevices(devices || []);

    statsDiv.innerHTML = `
        <div style="display: grid; gap: 8px; font-family: monospace;">
            <div><strong>Suspicion:</strong> <span style="color: ${getSuspicionColor(suspicion)}">${suspicion}</span></div>
            <div><strong>AI Top Guess:</strong> ${aiGuess}</div>
            <div><strong>Head Pose:</strong> ${headPose}</div>
            <div><strong>Audio:</strong> ${audio}</div>
            <div><strong>Deepfake:</strong> ${deepfake}</div>
            <div style="margin-top:8px;"><strong>People:</strong> ${peopleInfo}</div>
            <div><strong>Devices:</strong> ${deviceInfo}</div>
            <div><strong>Audio RMS:</strong> ${audioRms}</div>
        </div>
    `;
} // [web:127]

// Get color based on suspicion level
function getSuspicionColor(suspicion) {
    if (typeof suspicion === 'string') {
        const num = parseInt(suspicion);
        if (!isNaN(num)) {
            if (num >= 70) return '#ff4444';
            if (num >= 40) return '#ff8800';
            return '#44ff44';
        }
        return '#666';
    }

    const percent = parseInt(suspicion);
    if (percent >= 70) return '#ff4444';
    if (percent >= 40) return '#ff8800';
    return '#44ff44';
} // [web:127]

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden && streaming) {
        if (frameInterval) {
            clearInterval(frameInterval);
            frameInterval = null;
        }
    } else if (!document.hidden && streaming && !frameInterval) {
        frameInterval = setInterval(sendFrame, SEND_EVERY_MS);
    }
}); // [web:102]

// Handle page unload
window.addEventListener('beforeunload', function() {
    if (streaming) {
        stopStreaming();
    }
    socket.disconnect();
}); // [web:127]

// Initialize connection status
updateConnectionStatus('Connecting...', 'orange'); // [web:127]

// Periodically check connection status
setInterval(() => {
    if (!socket.connected) {
        updateConnectionStatus('Disconnected', 'red');
    }
}, 5000); // [web:127]

// Expose toggle for button
window.toggleStreaming = toggleStreaming; // [web:127]
