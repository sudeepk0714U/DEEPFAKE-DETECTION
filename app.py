# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# from ai_detection import AIDetectionEngine
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'demo-key'
# socketio = SocketIO(app, cors_allowed_origins="*")
# detector = AIDetectionEngine()
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/assessment')
# def assessment():
#     return render_template('assessment.html')
#
# @socketio.on('frame')
# def handle_frame(data):
#     try:
#         result = detector.check_frame(data.get('image', b''))
#         emit('ai_result', result)
#     except Exception as e:
#         emit('ai_result', {
#             'suspicion': 0,
#             'detected_class': f'Error: {str(e)}',
#             'head_pose_flag': False,
#             'audio_flag': False,
#             'deepfake_flag': False
#         })
#
# if __name__ == "__main__":
#     socketio.run(app, debug=True, host='0.0.0.0', port=5001)
















from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ai_detection import AIDetectionEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize detector once; enable tracking for stable IDs
try:
    detector = AIDetectionEngine(use_tracking=True, imgsz=640, conf=0.25, iou=0.45)
except Exception as e:
    detector = None
    init_error = f"Detector init failed: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@socketio.on('connect')
def on_connect():
    if detector is None:
        emit('ai_result', {
            'suspicion': 0,
            'detected_class': init_error if 'init_error' in globals() else 'Detector not available',
            'head_pose_flag': False,
            'audio_flag': False,
            'deepfake_flag': False
        })

@socketio.on('frame')
def handle_frame(data):
    try:
        if detector is None:
            raise RuntimeError("Detector not initialized")
        image_b64 = data.get('image', '')
        result = detector.check_frame(image_b64)
        emit('ai_result', result)
    except Exception as e:
        emit('ai_result', {
            'suspicion': 0,
            'detected_class': f'Error: {str(e)}',
            'head_pose_flag': False,
            'audio_flag': False,
            'deepfake_flag': False
        })

if __name__ == "__main__":
    # In production, set debug=False
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)

