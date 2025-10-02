import ssl
import uuid
import secrets
import string
from datetime import datetime

from flask import Flask, render_template, session, url_for, request, redirect, flash
from flask_session import Session
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# For macOS SSL in some environments
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'change-this-secret-in-production'

# Server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
Session(app)  # server-side session storage [web:3]

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)  # ORM [web:22]

# Login manager
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)  # Flask-Login [web:3]

# Socket.IO with session integration
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=True, logger=True, engineio_logger=False)
# manage_session=True makes Flask session visible in Socket.IO handlers; current_user works post login_user [web:6][web:24]

# =======================
# Models
# =======================
class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')  # 'company' or 'user'
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=True)
    company = db.relationship('Company', backref='users')

class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    join_code = db.Column(db.String(16), unique=True, nullable=False)
    start_at = db.Column(db.DateTime, nullable=True)
    end_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='open')  # draft, open, closed
    company = db.relationship('Company', backref='tests')

class TestAccess(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role = db.Column(db.String(20), default='candidate')  # host, candidate
    test = db.relationship('Test', backref='access_list')
    user = db.relationship('User', backref='test_access')

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    payload = db.Column(db.JSON, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))  # Flask-Login user loader [web:3]

# =======================
# Utilities
# =======================
def gen_code(n=8):
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(n))

def ensure_db():
    with app.app_context():
        db.create_all()

# =======================
# AI detector import (existing)
# =======================
from ai_detection import AIDetectionEngine

# In-memory live store (replace with Redis/DB in production)
detection_sessions = {}
report_snapshots = {}

# Initialize detector once
try:
    detector = AIDetectionEngine(use_tracking=True, imgsz=640, conf=0.25, iou=0.45)
    print("✓ AI Detector initialized")
except Exception as e:
    detector = None
    init_error = f"Detector init failed: {e}"
    print(init_error)

# =======================
# Routes: auth
# =======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        pwd = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, pwd):
            login_user(user)
            next_url = request.args.get('next')
            return redirect(next_url or url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')  # uses minimal login.html [web:2]

@app.route('/register/company', methods=['GET', 'POST'])
def register_company():
    if request.method == 'POST':
        company_name = request.form.get('company_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        pwd = request.form.get('password', '')
        if not company_name or not email or not pwd:
            flash('All fields required')
            return redirect(url_for('register_company'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register_company'))
        c = Company(name=company_name)
        db.session.add(c); db.session.flush()
        u = User(email=email, password_hash=generate_password_hash(pwd), role='company', company_id=c.id)
        db.session.add(u); db.session.commit()
        login_user(u)
        return redirect(url_for('company_dashboard'))
    return render_template('register_company.html')

@app.route('/register/user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        pwd = request.form.get('password', '')
        if not email or not pwd:
            flash('All fields required')
            return redirect(url_for('register_user'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register_user'))
        u = User(name=name,email=email, password_hash=generate_password_hash(pwd), role='user')
        db.session.add(u); db.session.commit()
        login_user(u)
        return redirect(url_for('join_test'))
    return render_template('register_user.html')

@app.route('/logout')
@login_required
def logout():
    # Clear test bindings to avoid reuse
    session.pop('active_test_id', None)
    session.pop('detection_session_id', None)
    session.modified = True
    logout_user()
    return redirect(url_for('login'))  # return to login page [web:2][web:104]

# =======================
# Routes: pages
# =======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'company':
        return redirect(url_for('company_dashboard'))
    return redirect(url_for('join_test'))

@app.route('/company/dashboard', methods=['GET'])
@login_required
def company_dashboard():
    if current_user.role != 'company':
        return "Unauthorized", 403
    # Only show OPEN tests
    tests = Test.query.filter_by(company_id=current_user.company_id, status='open').all()
    return render_template('company_dashboard.html', tests=tests)


@app.route('/company/tests/create', methods=['POST'])
@login_required
def create_test():
    if current_user.role != 'company':
        return "Unauthorized", 403
    name = request.form.get('name', 'Untitled Test')
    t = Test(company_id=current_user.company_id, name=name, join_code=gen_code(), status='open')
    db.session.add(t); db.session.commit()
    db.session.add(TestAccess(test_id=t.id, user_id=current_user.id, role='host')); db.session.commit()
    flash(f'Test created. Join code: {t.join_code}')
    return redirect(url_for('company_dashboard'))

@app.route('/company/tests/<int:test_id>/close', methods=['POST'])
@login_required
def close_test(test_id):
    if current_user.role != 'company':
        return "Unauthorized", 403
    t = Test.query.get_or_404(test_id)
    if t.company_id != current_user.company_id:
        return "Unauthorized", 403
    t.status = 'closed'; db.session.commit()
    flash('Test closed')
    return redirect(url_for('company_dashboard'))

@app.route('/company/tests/<int:test_id>/reports')
@login_required
def company_test_reports(test_id):
    if current_user.role != 'company':
        return "Unauthorized", 403
    t = Test.query.get_or_404(test_id)
    if t.company_id != current_user.company_id:
        return "Unauthorized", 403
    reports = Report.query.filter_by(test_id=test_id, company_id=current_user.company_id)\
                          .order_by(Report.created_at.desc()).all()
    # fetch users for display names
    user_ids = {r.user_id for r in reports}
    users = {u.id: u for u in User.query.filter(User.id.in_(user_ids)).all()}
    return render_template('company_reports.html', test=t, reports=reports, users=users)


@app.route('/join', methods=['GET', 'POST'])
@login_required
def join_test():
    if request.method == 'POST':
        code = (request.form.get('code') or '').strip().upper()
        t = Test.query.filter_by(join_code=code).first()
        if not t or t.status != 'open':
            # Stay on page with error; do not set active_test_id
            return render_template('join.html', error='Invalid or closed code')
        if current_user.role == 'user':
            exists = TestAccess.query.filter_by(test_id=t.id, user_id=current_user.id).first()
            if not exists:
                db.session.add(TestAccess(test_id=t.id, user_id=current_user.id, role='candidate'))
                db.session.commit()
        session['active_test_id'] = t.id
        session.modified = True
        return redirect(url_for('assessment'))
    return render_template('join.html')

@app.route('/assessment')
@login_required
def assessment():
    test_id = session.get('active_test_id') or request.args.get('test_id', type=int)
    if current_user.role == 'company':
        if not test_id:
            return redirect(url_for('company_dashboard'))
        t = Test.query.get_or_404(test_id)
        if t.company_id != current_user.company_id:
            return "Unauthorized", 403
        session['active_test_id'] = t.id; session.modified = True
    else:
        if not test_id:
            return redirect(url_for('join_test'))
        ta = TestAccess.query.filter_by(test_id=test_id, user_id=current_user.id).first()
        if not ta:
            session.pop('active_test_id', None)
            session.modified = True
            return redirect(url_for('join_test'))
    return render_template('assessment.html')

@app.route('/report')
@login_required
def report():
    rid = request.args.get('rid', type=int)
    data = None

    if rid:
        r = Report.query.get_or_404(rid)
        if current_user.role == 'company':
            if r.company_id != current_user.company_id:
                return "Unauthorized", 403
        else:
            if r.user_id != current_user.id:
                return "Unauthorized", 403
        data = r.payload  # <- use the persisted snapshot payload

    if not data:
        snap_id = request.args.get('report_id')
        if snap_id and snap_id in report_snapshots:
            data = report_snapshots[snap_id]
    if not data:
        sid = session.get('detection_session_id')
        if sid and sid in detection_sessions:
            data = detection_sessions[sid]
    if not data:
        snap = session.get('report_snapshot')
        if snap:
            data = snap
    if not data:
        data = {
            'session_id': 'unknown', 'start_time': '-', 'end_time': '-',
            'total_frames': 0, 'suspicious_frames': 0, 'total_suspicion_score': 0.0,
            'max_suspicion': 0.0, 'avg_suspicion': 0.0, 'suspicion_rate': 0.0,
            'head_pose_warnings': 0, 'audio_warnings': 0,
            'device_detections': {'laptop': 0, 'cell phone': 0},
            'unique_persons_count': 0, 'frame_data': []
        }

    if isinstance(data.get('unique_persons'), set):
        data['unique_persons_count'] = len(data['unique_persons'])

    # Optional: clear candidate bindings
    if current_user.role == 'user':
        session.pop('active_test_id', None)
        session.pop('detection_session_id', None)
        session.modified = True

    return render_template('report.html', data=data)


# =======================
# Socket.IO events
# =======================
@socketio.on('connect')
def on_connect():
    # Enforce authentication and test membership for streaming [web:6]
    if not current_user.is_authenticated:
        return False
    test_id = session.get('active_test_id')
    if not test_id:
        return False
    t = Test.query.get(test_id)
    if not t:
        return False
    if current_user.role == 'company':
        if t.company_id != current_user.company_id:
            return False
    else:
        ta = TestAccess.query.filter_by(test_id=test_id, user_id=current_user.id).first()
        if not ta:
            return False

    sid = request.sid
    session['detection_session_id'] = sid
    session.modified = True

    detection_sessions[sid] = {
        'session_id': sid,
        'user_id': current_user.id,
        'company_id': t.company_id,
        'test_id': t.id,
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

@socketio.on('frame')
def handle_frame(data):
    try:
        if detector is None:
            raise RuntimeError("Detector not initialized")
        if not current_user.is_authenticated:
            return
        sid = session.get('detection_session_id') or request.sid
        if sid not in detection_sessions:
            return

        result = detector.check_frame(data.get('image', ''))

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
    client_sid = (payload or {}).get('sid') or request.sid
    sid = session.get('detection_session_id') or client_sid
    session['detection_session_id'] = sid
    session.modified = True

    if sid in detection_sessions:
        sd = detection_sessions[sid]
        sd['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if sd['total_frames'] > 0:
            sd['avg_suspicion'] = sd['total_suspicion_score'] / sd['total_frames']
            sd['suspicion_rate'] = (sd['suspicious_frames'] / sd['total_frames']) * 100.0
        else:
            sd['avg_suspicion'] = 0.0
            sd['suspicion_rate'] = 0.0

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

        # Save to session and server snapshot map
        session['report_snapshot'] = snapshot
        session.modified = True
        report_id = str(uuid.uuid4())
        report_snapshots[report_id] = snapshot

        # Persist report for company visibility
        try:
            r = Report(
                test_id=sd.get('test_id'),
                company_id=sd.get('company_id'),
                user_id=sd.get('user_id'),
                payload=snapshot
            )
            db.session.add(r); db.session.commit()
        except Exception:
            db.session.rollback()

        # Choose post-test flow:
        # Option A: Show report, then user may logout manually or via meta-refresh
        report_url = url_for('report', _external=True, report_id=report_id)
        emit('redirect_to_report', {'url': report_url}, room=client_sid)

        # Option B (force immediate logout instead of showing report):
        # emit('redirect_to_report', {'url': url_for('logout', _external=True)}, room=client_sid)

    else:
        emit('redirect_to_report', {'url': url_for('report', _external=True)}, room=client_sid)

@socketio.on('disconnect')
def on_disconnect(reason=None):
    pass

@socketio.on_error_default
def default_error_handler(e):
    import traceback
    print("Socket.IO Error:", e, "Event:", getattr(request, 'event', None))
    traceback.print_exc()

if __name__ == "__main__":
    ensure_db()
    print("Assessment: http://localhost:5001/assessment")
    print("Report:     http://localhost:5001/report")
    print("Login:      http://localhost:5001/login")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
