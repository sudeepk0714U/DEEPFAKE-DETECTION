from flask import Flask, render_template, session, url_for, request, redirect
from flask_socketio import SocketIO, emit
from ai_detection import AIDetectionEngine
from datetime import datetime
from flask_session import Session  # pip install Flask-Session
import ssl
import uuid

# For macOS SSL in some environments
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Server-side session storage to make WebSocket -> HTTP visibility reliable
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
# Cookie flags for local dev; adjust for HTTPS/cross-site if needed
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # set to 'None' if cross-site/iframe with HTTPS
app.config['SESSION_COOKIE_SECURE'] = False       # True if HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True

Session(app)  # initialize Flask-Session [web:195][web:199]

socketio = SocketIO(app, cors_allowed_origins="*", manage_session=True, logger=True, engineio_logger=False)

# In-memory live store (replace with Redis/DB in production)
detection_sessions = {}

# Report snapshots by ID (fallback when session cookie is missing)
report_snapshots = {}

# Initialize detector once
try:
    detector = AIDetectionEngine(use_tracking=True, imgsz=640, conf=0.25, iou=0.45)
    print("✓ AI Detector initialized")
except Exception as e:
    detector = None
    init_error = f"Detector init failed: {e}"
    print(init_error)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    # Touch session so cookie is set via HTTP
    session['initialized'] = True
    session.modified = True
    return render_template('assessment.html')

@app.route('/report')
def report():
    # 1) If a report_id is provided, prefer loading from server snapshot dict
    rid = request.args.get('report_id')
    data = None
    if rid and rid in report_snapshots:
        data = report_snapshots.get(rid)

    # 2) Else, try live data based on the linked session id
    if not data:
        sid = session.get('detection_session_id')
        if sid and sid in detection_sessions:
            data = detection_sessions[sid]

    # 3) Else, fallback to snapshot saved in Flask session
    if not data:
        snap = session.get('report_snapshot')
        if snap:
            data = snap

    # 4) Final fallback: render minimal data to avoid template errors
    if not data:
        data = {
            'session_id': 'unknown',
            'start_time': '-',
            'end_time': '-',
            'total_frames': 0,
            'suspicious_frames': 0,
            'total_suspicion_score': 0.0,
            'max_suspicion': 0.0,
            'avg_suspicion': 0.0,
            'suspicion_rate': 0.0,
            'head_pose_warnings': 0,
            'audio_warnings': 0,
            'device_detections': {'laptop': 0, 'cell phone': 0},
            'unique_persons_count': 0,
            'frame_data': []
        }

    # Normalize unique persons count if a set is present
    if isinstance(data.get('unique_persons'), set):
        data['unique_persons_count'] = len(data['unique_persons'])

    return render_template('report.html', data=data)

@socketio.on('connect')
def on_connect():
    sid = request.sid
    # Link this socket to an HTTP-visible session id
    session['detection_session_id'] = sid
    session.modified = True

    detection_sessions[sid] = {
        'session_id': sid,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': None,
        'total_frames': 0,
        'suspicious_frames': 0,
        'total_suspicion_score': 0.0,
        'max_suspicion': 0.0,
        'avg_suspicion': 0.0,
        'suspicion_rate': 0.0,
        'head_pose_warnings': 0,
        'audio_warnings': 0,
        'device_detections': {'laptop': 0, 'cell phone': 0},
        'unique_persons': set(),
        'unique_persons_count': 0,
        'frame_data': []
    }

    emit('connection_confirmed', {'status': 'connected', 'session_id': sid[:12]})

    if detector is None:
        emit('ai_result', {
            'suspicion': 0,
            'detected_class': init_error if 'init_error' in globals() else 'Detector not available',
            'head_pose_flag': False,
            'audio_flag': False,
            'deepfake_flag': False,
            'persons': [],
            'devices': [],
            'audio_rms': 0.0
        })

@socketio.on('frame')
def handle_frame(data):
    try:
        if detector is None:
            raise RuntimeError("Detector not initialized")

        sid = session.get('detection_session_id') or request.sid
        if sid not in detection_sessions:
            detection_sessions[sid] = {
                'session_id': sid,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': None,
                'total_frames': 0,
                'suspicious_frames': 0,
                'total_suspicion_score': 0.0,
                'max_suspicion': 0.0,
                'avg_suspicion': 0.0,
                'suspicion_rate': 0.0,
                'head_pose_warnings': 0,
                'audio_warnings': 0,
                'device_detections': {'laptop': 0, 'cell phone': 0},
                'unique_persons': set(),
                'unique_persons_count': 0,
                'frame_data': []
            }

        img_b64 = data.get('image', '')
        result = detector.check_frame(img_b64)

        sd = detection_sessions[sid]
        sd['total_frames'] += 1
        suspicion = result.get('suspicion', 0.0)
        sd['total_suspicion_score'] += suspicion
        sd['max_suspicion'] = max(sd['max_suspicion'], suspicion)

        if suspicion >= 0.4:
            sd['suspicious_frames'] += 1
        if result.get('head_pose_flag'):
            sd['head_pose_warnings'] += 1
        if result.get('audio_flag'):
            sd['audio_warnings'] += 1

        for device in result.get('devices', []):
            label = device.get('label', '')
            if label in sd['device_detections']:
                sd['device_detections'][label] += 1

        for person in result.get('persons', []):
            pid = person.get('id')
            if pid is not None:
                sd['unique_persons'].add(pid)

        sd['frame_data'].append({
            'frame_num': sd['total_frames'],
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'suspicion': suspicion,
            'detected_class': result.get('detected_class', '?'),
            'persons_count': len(result.get('persons', [])),
            'devices_count': len(result.get('devices', []))
        })
        if len(sd['frame_data']) > 100:
            sd['frame_data'].pop(0)

        emit('ai_result', result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        emit('ai_result', {
            'suspicion': 0,
            'detected_class': f'Error: {str(e)}',
            'head_pose_flag': False,
            'audio_flag': False,
            'deepfake_flag': False,
            'persons': [],
            'devices': [],
            'audio_rms': 0.0
        })

@socketio.on('stop_detection')
def handle_stop_detection(payload=None):
    # Identify caller and session
    client_sid = (payload or {}).get('sid') or request.sid
    sid = session.get('detection_session_id') or client_sid
    session['detection_session_id'] = sid
    session.modified = True

    # Finalize stats and persist snapshot both in session and in a server dict by report_id
    if sid in detection_sessions:
        sd = detection_sessions[sid]
        sd['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if sd['total_frames'] > 0:
            sd['avg_suspicion'] = sd['total_suspicion_score'] / sd['total_frames']
            sd['suspicion_rate'] = (sd['suspicious_frames'] / sd['total_frames']) * 100.0
        else:
            sd['avg_suspicion'] = 0.0
            sd['suspicion_rate'] = 0.0

        # Build snapshot payload
        snapshot = {
            'session_id': sd['session_id'],
            'start_time': sd['start_time'],
            'end_time': sd['end_time'],
            'total_frames': sd['total_frames'],
            'suspicious_frames': sd['suspicious_frames'],
            'total_suspicion_score': sd['total_suspicion_score'],
            'max_suspicion': sd['max_suspicion'],
            'avg_suspicion': sd['avg_suspicion'],
            'suspicion_rate': sd['suspicion_rate'],
            'head_pose_warnings': sd['head_pose_warnings'],
            'audio_warnings': sd['audio_warnings'],
            'device_detections': sd['device_detections'],
            'unique_persons_count': len(sd['unique_persons']),
            'frame_data': sd['frame_data'][-100:]
        }

        # Save snapshot in HTTP-visible session (cookie-based or server-side storage)
        session['report_snapshot'] = snapshot
        session.modified = True

        # Also save a server-side report snapshot with an ID to bypass cookie issues
        report_id = str(uuid.uuid4())
        report_snapshots[report_id] = snapshot

        # Redirect with absolute URL and report_id query
        report_url = url_for('report', _external=True, report_id=report_id)
        emit('redirect_to_report', {'url': report_url}, room=client_sid)
    else:
        # No data; still redirect to report (will render minimal report)
        emit('redirect_to_report', {'url': url_for('report', _external=True)}, room=client_sid)

@socketio.on('disconnect')
def on_disconnect(reason=None):
    # Do NOT delete data here; report page may still need it shortly after
    pass

@socketio.on_error_default
def default_error_handler(e):
    import traceback
    print("Socket.IO Error:", e, "Event:", getattr(request, 'event', None))
    traceback.print_exc()

if __name__ == "__main__":
    print("Assessment: http://localhost:5001/assessment")
    print("Report:     http://localhost:5001/report")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
