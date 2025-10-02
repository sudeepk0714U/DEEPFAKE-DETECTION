// Initialize Socket.IO connection only on assessment page after login
let socket = io();
let streaming = false;
let statsDiv = document.getElementById('stats');
let video = document.getElementById('video');
let startBtn = document.getElementById('startBtn');
let mediaStream = null;
let connectionStatus = document.createElement('div');
let frameInterval = null;

const SEND_EVERY_MS = 500;

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

function updateConnectionStatus(status, color) {
    connectionStatus.textContent = status;
    connectionStatus.style.backgroundColor = color;
    connectionStatus.style.color = 'white';
}
updateConnectionStatus('Connecting...', 'orange');

// Require authenticated session; server will refuse unauthenticated connects
socket.on('connect', function() {
    updateConnectionStatus('Connected', 'green');
});

socket.on('disconnect', function(reason) {
    updateConnectionStatus('Disconnected', 'red');
    if (streaming) {
        streaming = false;
        startBtn.innerText = 'Start AI Detection';
        startBtn.disabled = false;
    }
});

socket.on('connect_error', function(error) {
    updateConnectionStatus('Connection Error', 'orange');
});

socket.on('connection_confirmed', function(data) {
    // Connected and authorized
});

socket.on('ai_result', function(data) {
    if (data.error) {
        updateStats('Detection Error', data.error, 'Error', 'Error', 'Error', [], [], '?');
        return;
    }

    const suspicionPercent = Math.round((data.suspicion || 0) * 100);
    const detectedClass = data.detected_class || '?';
    const headPose = data.head_pose_flag ? '⚠️ Warning' : '✅ OK';
    const audio = data.audio_flag ? '⚠️ Warning' : '✅ OK';
    const deepfake = data.deepfake_flag ? '⚠️ Warning' : '✅ OK';
    const persons = Array.isArray(data.persons) ? data.persons : [];
    const devices = Array.isArray(data.devices) ? data.devices : [];
    const audioRms = (typeof data.audio_rms === 'number') ? data.audio_rms.toFixed(4) : '?';

    updateStats(suspicionPercent + '%', detectedClass, headPose, audio, deepfake, persons, devices, audioRms);
});

// Server-driven redirect to report
socket.on('redirect_to_report', function(data) {
    const url = (data && data.url) ? data.url : '/report';

    streaming = false;
    if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (video && video.srcObject) video.srcObject = null;

    try { window.location.assign(url); } catch (e) {}
    setTimeout(() => { try { window.location.replace(url); } catch (e) {} }, 150);
    setTimeout(() => { try { window.location.href = url; } catch (e) {} }, 300);
    setTimeout(() => { try { location.reload(true); } catch (e) {} }, 1500);
});

function toggleStreaming() {
    if (!socket.connected) {
        alert('Please wait for server connection before starting detection.');
        return;
    }
    if (!streaming) startStreaming();
    else stopStreaming();
}

function startStreaming() {
    navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
    })
    .then(stream => {
        video.srcObject = stream;
        mediaStream = stream;
        streaming = true;
        startBtn.innerText = 'Stop AI Detection';

        video.addEventListener('loadedmetadata', () => {
            frameInterval = setInterval(sendFrame, SEND_EVERY_MS);
        }, { once: true });

        updateStats('Starting detection...', '?', 'Initializing', 'OK', 'OK', [], [], '?');
    })
    .catch(error => {
        let errorMessage = 'Camera access denied or not available';
        if (error.name === 'NotFoundError') errorMessage = 'No camera found on this device';
        else if (error.name === 'NotAllowedError') errorMessage = 'Camera permission denied. Please allow camera access and try again.';
        else if (error.name === 'NotReadableError') errorMessage = 'Camera is being used by another application';
        else if (error.name === 'OverconstrainedError') errorMessage = 'Camera does not meet the required constraints';

        alert(errorMessage);
        updateStats('Camera Error', '?', 'Error', 'Error', 'Error', [], [], '?');
    });
}

function stopStreaming() {
    streaming = false;
    startBtn.innerText = 'Generating Report...';
    startBtn.disabled = true;

    if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; }
    if (video && video.srcObject) video.srcObject = null;

    updateStats('Processing...', 'Generating report', 'Finalizing', 'Finalizing', 'Finalizing', [], [], '?');

    socket.emit('stop_detection', { sid: socket.id });
}

function sendFrame() {
    if (!streaming || !socket.connected) return;

    try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('frame', { image: imageData });
    } catch (error) {
        updateStats('Frame Capture Error', '?', 'Error', 'Error', 'Error', [], [], '?');
    }
}

function summarizePeople(persons) {
    if (!persons.length) return 'None';
    return persons.map(p => {
        const id = (p.id !== null && p.id !== undefined) ? `#${p.id} ` : '';
        const state = p.state || 'NA';
        const conf = (typeof p.conf === 'number') ? ` (${(p.conf*100).toFixed(1)}%)` : '';
        return `${id}${state}${conf}`;
    }).join(', ');
}

function summarizeDevices(devices) {
    if (!devices.length) return 'None';
    return devices.map(d => {
        const name = d.label || 'device';
        const conf = (typeof d.conf === 'number') ? ` (${(d.conf*100).toFixed(1)}%)` : '';
        return `${name}${conf}`;
    }).join(', ');
}

function updateStats(suspicion, aiGuess, headPose, audio, deepfake, persons, devices, audioRms) {
    const peopleInfo = summarizePeople(persons || []);
    const deviceInfo = summarizeDevices(devices || []);

    if (!statsDiv) return;
    statsDiv.innerHTML = `
        <div style="display: grid; gap: 8px; font-family: monospace;">
            <div><span>Suspicion:</span> <span style="color: ${getSuspicionColor(suspicion)}">${suspicion}</span></div>
            <div><span>AI Top Guess:</span> ${aiGuess}</div>
            <div><span>Head Pose:</span> ${headPose}</div>
            <div><span>Audio:</span> ${audio}</div>
            <div><span>Deepfake:</span> ${deepfake}</div>
            <div style="margin-top:8px;"><span>People:</span> ${peopleInfo}</div>
            <div><span>Devices:</span> ${deviceInfo}</div>
            <div><span>Audio RMS:</span> ${audioRms}</div>
        </div>
    `;
}

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
}

window.toggleStreaming = toggleStreaming;
