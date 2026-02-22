import os
import json
import re
import atexit
import shutil
import subprocess
import threading
from collections import Counter
from datetime import datetime
from functools import wraps
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

from models import Answer, Form, Question, Response, User, db


def is_placeholder_key(value):
    return (value or '').strip().lower().startswith('your_')


def load_local_env(paths=None):
    if paths is None:
        paths = ['.env', 'example.env', 'ecample.env']
    for env_path in paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, 'r', encoding='utf-8') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if not key:
                        continue
                    current = os.environ.get(key, '')
                    if (not current) or is_placeholder_key(current):
                        os.environ[key] = value
        except OSError:
            continue


load_local_env()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'feedback-secret-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PREFERRED_URL_SCHEME'] = os.environ.get('PREFERRED_URL_SCHEME', 'https')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

db.init_app(app)

ALLOWED_QTYPES = {'text', 'longtext', 'rating', 'mcq', 'checkbox'}
DEFAULT_STARTER_QUESTIONS = [
    {'type': 'rating', 'text': 'How satisfied are you with your overall experience?', 'options': ''},
    {
        'type': 'mcq',
        'text': 'Which area needs the most improvement?',
        'options': 'Service, Quality, Pricing, Support',
    },
    {'type': 'longtext', 'text': 'What did you like the most?', 'options': ''},
    {
        'type': 'checkbox',
        'text': 'Which features did you use?',
        'options': 'Website, Mobile App, Customer Support, In-store',
    },
    {'type': 'text', 'text': 'Any additional suggestions?', 'options': ''},
]


cloudflared_proc = None
public_tunnel_url = ''


def ensure_schema():
    inspector = inspect(db.engine)
    table_names = set(inspector.get_table_names())

    if 'form' in table_names:
        form_cols = {col['name'] for col in inspector.get_columns('form')}
        if 'user_id' not in form_cols:
            db.session.execute(text('ALTER TABLE form ADD COLUMN user_id INTEGER'))
            db.session.commit()
        if 'closing_at' not in form_cols:
            db.session.execute(text('ALTER TABLE form ADD COLUMN closing_at DATETIME'))
            db.session.commit()

    if 'response' in table_names:
        response_cols = {col['name'] for col in inspector.get_columns('response')}
        if 'user_id' not in response_cols:
            db.session.execute(text('ALTER TABLE response ADD COLUMN user_id INTEGER'))
            db.session.commit()


with app.app_context():
    db.create_all()
    ensure_schema()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)

    return wrapped_view


def parse_questions_from_request(req):
    questions = req.form.getlist('question')
    types = req.form.getlist('type')
    options = req.form.getlist('options')
    parsed = []

    for q, t, o in zip(questions, types, options):
        text = (q or '').strip()[:300]
        qtype = (t or 'text').strip()
        opts = (o or '').strip()[:500]
        if not text:
            continue
        parsed.append({'text': text, 'qtype': qtype, 'options': opts})

    return parsed


def parse_closing_at(raw_value):
    raw = (raw_value or '').strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def is_form_closed(form):
    return bool(form.closing_at and datetime.utcnow() > form.closing_at)


def sanitize_generated_questions(items):
    sanitized = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        qtype = str(item.get('type', 'text')).strip().lower()
        if qtype not in ALLOWED_QTYPES:
            qtype = 'text'
        text = str(item.get('text', '')).strip()[:300]
        options = str(item.get('options', '')).strip()[:500]
        if not text:
            continue
        if qtype not in ('mcq', 'checkbox'):
            options = ''
        sanitized.append({'type': qtype, 'text': text, 'options': options})
    return sanitized[:8]


def parse_json_from_model_text(raw_text):
    cleaned = (raw_text or '').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def build_ai_prompt(title):
    return (
        f"Form title: {title}\n"
        "Generate 5 concise starter questions tailored to this title.\n"
        "Use this JSON array format exactly:\n"
        "[{\"type\":\"text|longtext|rating|mcq|checkbox\",\"text\":\"...\",\"options\":\"...\"}]\n"
        "Rules:\n"
        "- rating/text/longtext must have empty options.\n"
        "- mcq/checkbox must provide 3-6 comma-separated options.\n"
        "- Keep each question practical and specific."
    )


def generate_ollama_questions_from_title(title):
    model = os.environ.get('OLLAMA_MODEL', 'llama3.2').strip()
    base_url = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').strip().rstrip('/')
    endpoint = f'{base_url}/api/generate'
    prompt = (
        "You create feedback form questions. Return only valid JSON, no markdown.\n"
        + build_ai_prompt(title)
    )

    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
    }
    data = json.dumps(payload).encode('utf-8')
    req = Request(
        endpoint,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urlopen(req, timeout=40) as resp:
            body = resp.read().decode('utf-8')
    except HTTPError as err:
        detail = err.read().decode('utf-8', errors='ignore')
        return None, f'Ollama request failed ({err.code}): {detail[:200]}'
    except URLError as err:
        return None, (
            'Cannot connect to Ollama. Start it with "ollama serve" '
            'and make sure OLLAMA_BASE_URL is correct.'
        )
    except Exception as err:
        return None, f'Ollama request failed: {err}'

    try:
        parsed = json.loads(body)
        content = parsed.get('response', '')
        candidate = parse_json_from_model_text(content)
        questions = sanitize_generated_questions(candidate)
        if questions:
            return questions, None
        return None, 'Ollama returned no valid questions.'
    except Exception as err:
        return None, f'Ollama response parse failed: {err}'


def generate_gemini_questions_from_title(title, api_key):
    configured_model = (
        os.environ.get('GEMINI_MODEL', '').strip()
        or os.environ.get('OPENAI_MODEL', '').strip()
    )
    model_candidates = [
        configured_model,
        'gemini-2.0-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash',
    ]
    model_candidates = [m for m in model_candidates if m]
    prompt = build_ai_prompt(title)
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0.6,
            'responseMimeType': 'application/json',
        },
    }
    data = json.dumps(payload).encode('utf-8')
    last_error = None

    for model in model_candidates:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?"
            + urlencode({'key': api_key})
        )
        req = Request(
            endpoint,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        try:
            with urlopen(req, timeout=25) as resp:
                body = resp.read().decode('utf-8')
        except HTTPError as err:
            detail = err.read().decode('utf-8', errors='ignore')
            last_error = f'Gemini request failed ({err.code}) for model {model}: {detail[:160]}'
            if err.code == 404:
                continue
            return None, last_error
        except URLError as err:
            return None, f'Gemini connection failed: {err.reason}'
        except Exception as err:
            return None, f'Gemini request failed: {err}'

        try:
            parsed = json.loads(body)
            content = (
                parsed.get('candidates', [{}])[0]
                .get('content', {})
                .get('parts', [{}])[0]
                .get('text', '')
            )
            candidate = parse_json_from_model_text(content)
            questions = sanitize_generated_questions(candidate)
            if questions:
                return questions, None
            last_error = f'Gemini returned no valid questions for model {model}.'
        except Exception as err:
            last_error = f'Gemini response parse failed for model {model}: {err}'

    return None, last_error or 'Gemini request failed for all attempted models.'


def generate_openai_questions_from_title(title, api_key):
    model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    system_prompt = (
        "You create feedback form questions. "
        "Return only valid JSON, no markdown."
    )
    user_prompt = build_ai_prompt(title)

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0.6,
        'max_tokens': 500,
    }
    data = json.dumps(payload).encode('utf-8')

    req = Request(
        'https://api.openai.com/v1/chat/completions',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )

    try:
        with urlopen(req, timeout=25) as resp:
            body = resp.read().decode('utf-8')
    except HTTPError as err:
        detail = err.read().decode('utf-8', errors='ignore')
        return None, f'OpenAI request failed ({err.code}): {detail[:160]}'
    except URLError as err:
        return None, f'OpenAI connection failed: {err.reason}'
    except Exception as err:
        return None, f'OpenAI request failed: {err}'

    try:
        parsed = json.loads(body)
        content = parsed['choices'][0]['message']['content']
        candidate = parse_json_from_model_text(content)
        questions = sanitize_generated_questions(candidate)
        if questions:
            return questions, None
        return None, 'OpenAI returned no valid questions.'
    except Exception as err:
        return None, f'OpenAI response parse failed: {err}'


def generate_ai_questions_from_title(title):
    provider = os.environ.get('AI_PROVIDER', 'ollama').strip().lower()
    openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
    gemini_key = os.environ.get('GEMINI_API_KEY', '').strip()

    if openai_key.startswith('AIza') and not gemini_key:
        gemini_key = openai_key

    if provider in ('ollama', 'auto'):
        questions, error = generate_ollama_questions_from_title(title)
        if questions:
            return questions, None
        if provider == 'ollama':
            return None, error

    if provider == 'openai':
        if openai_key:
            return generate_openai_questions_from_title(title, openai_key)
        return None, 'OPENAI_API_KEY is not configured.'

    if provider == 'gemini':
        if gemini_key:
            return generate_gemini_questions_from_title(title, gemini_key)
        return None, 'GEMINI_API_KEY is not configured.'

    if gemini_key:
        return generate_gemini_questions_from_title(title, gemini_key)
    if openai_key:
        return generate_openai_questions_from_title(title, openai_key)
    return None, (
        'No AI provider available. For free local mode use Ollama: '
        'set AI_PROVIDER=ollama, run "ollama serve", and pull a model like "ollama pull llama3.2".'
    )


def read_cloudflared_output(proc):
    global public_tunnel_url
    url_pattern = re.compile(r'https://[a-zA-Z0-9.-]+\.trycloudflare\.com')
    for line in iter(proc.stdout.readline, ''):
        message = line.strip()
        if not message:
            continue
        print(f'[cloudflared] {message}')
        match = url_pattern.search(message)
        if match:
            public_tunnel_url = match.group(0).rstrip('/')
            print(f'Public URL: {public_tunnel_url}')


def build_public_url(endpoint, **values):
    path = url_for(endpoint, **values)
    configured_base = os.environ.get('PUBLIC_BASE_URL', '').strip().rstrip('/')
    base = configured_base or public_tunnel_url
    if base:
        return f'{base}{path}'
    return url_for(endpoint, _external=True, _scheme='https', **values)


def start_quick_tunnel_if_available(port):
    global cloudflared_proc
    # Start quick tunnel by default when cloudflared exists; set AUTO_TUNNEL=0 to disable.
    auto_tunnel = os.environ.get('AUTO_TUNNEL', '1').lower() in ('1', 'true', 'yes', 'on')
    if not auto_tunnel:
        return

    candidates = [
        os.environ.get('CLOUDFLARED_PATH', '').strip(),
        os.path.join(os.getcwd(), 'cloudflared.exe'),
        shutil.which('cloudflared') or '',
    ]
    executable = next((path for path in candidates if path and os.path.exists(path)), '')
    if not executable:
        return

    try:
        cloudflared_proc = subprocess.Popen(
            [executable, 'tunnel', '--url', f'http://localhost:{port}', '--no-autoupdate'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as err:
        print(f'Failed to start cloudflared: {err}')
        return

    def cleanup_cloudflared():
        if cloudflared_proc and cloudflared_proc.poll() is None:
            cloudflared_proc.terminate()

    atexit.register(cleanup_cloudflared)
    threading.Thread(target=read_cloudflared_output, args=(cloudflared_proc,), daemon=True).start()


@app.context_processor
def inject_user():
    return {'current_username': session.get('username')}


@app.route('/', methods=['GET'])
def root():
    if session.get('user_id'):
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))

        flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash('Username already exists.', 'error')
            return render_template('register.html')

        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        flash('Account created. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/home')
@login_required
def index():
    forms = Form.query.filter_by(user_id=session['user_id']).order_by(Form.id.desc()).all()
    responses = Response.query.filter_by(user_id=session['user_id']).all()
    response_counts = Counter(r.form_id for r in responses)
    anonymous_counts = Counter(r.form_id for r in responses if r.username == 'Anonymous')
    total_forms = len(forms)
    total_responses = len(responses)
    total_anonymous = sum(anonymous_counts.values())

    form_cards = []
    for form in forms:
        form_cards.append({
            'form': form,
            'response_count': response_counts.get(form.id, 0),
            'anonymous_count': anonymous_counts.get(form.id, 0),
            'share_url': build_public_url('public_form', form_id=form.id),
            'is_closed': is_form_closed(form),
        })

    return render_template(
        'index.html',
        forms=forms,
        form_cards=form_cards,
        total_forms=total_forms,
        total_responses=total_responses,
        total_anonymous=total_anonymous,
    )


@app.route('/create', methods=['GET', 'POST'])
@login_required
def create_form():
    if request.method == 'POST':
        title = request.form['title'].strip()[:200]
        closing_at = parse_closing_at(request.form.get('closing_at'))
        if request.form.get('closing_at') and not closing_at:
            flash('Invalid closing date format.', 'error')
            return render_template('create_form.html')
        parsed_questions = parse_questions_from_request(request)
        if not parsed_questions:
            flash('Add at least one question before saving.', 'error')
            return render_template('create_form.html')

        form = Form(title=title, user_id=session['user_id'], closing_at=closing_at)
        db.session.add(form)
        db.session.commit()

        for item in parsed_questions:
            db.session.add(
                Question(
                    form_id=form.id,
                    text=item['text'],
                    qtype=item['qtype'],
                    options=item['options'],
                )
            )

        db.session.commit()
        flash('Form created successfully.', 'success')
        return redirect(url_for('index'))

    return render_template('create_form.html')


@app.route('/api/starter-questions', methods=['POST'])
@login_required
def starter_questions():
    payload = request.get_json(silent=True) or {}
    title = (payload.get('title') or '').strip()
    if not title:
        return jsonify({'ok': False, 'error': 'Form title is required.'}), 400

    questions, error = generate_ai_questions_from_title(title)
    if questions:
        return jsonify({'ok': True, 'questions': questions, 'source': 'ai'})

    return jsonify(
        {
            'ok': True,
            'questions': DEFAULT_STARTER_QUESTIONS,
            'source': 'fallback',
            'warning': error or 'AI unavailable. Using default starter questions.',
        }
    )


@app.route('/edit/<int:form_id>', methods=['GET', 'POST'])
@login_required
def edit_form(form_id):
    form = Form.query.filter_by(id=form_id, user_id=session['user_id']).first_or_404()
    existing_questions = Question.query.filter_by(form_id=form_id).order_by(Question.id.asc()).all()

    if request.method == 'POST':
        title = request.form['title'].strip()[:200]
        closing_at = parse_closing_at(request.form.get('closing_at'))
        if request.form.get('closing_at') and not closing_at:
            flash('Invalid closing date format.', 'error')
            return render_template('edit_form.html', form=form, questions=existing_questions)
        parsed_questions = parse_questions_from_request(request)
        if not parsed_questions:
            flash('At least one question is required.', 'error')
            return render_template('edit_form.html', form=form, questions=existing_questions)

        form.title = title
        form.closing_at = closing_at
        Question.query.filter_by(form_id=form_id).delete()
        db.session.commit()

        for item in parsed_questions:
            db.session.add(
                Question(
                    form_id=form.id,
                    text=item['text'],
                    qtype=item['qtype'],
                    options=item['options'],
                )
            )
        db.session.commit()
        flash('Form updated successfully.', 'success')
        return redirect(url_for('index'))

    return render_template('edit_form.html', form=form, questions=existing_questions)


@app.route('/delete/<int:form_id>', methods=['POST'])
@login_required
def delete_form(form_id):
    form = Form.query.filter_by(id=form_id, user_id=session['user_id']).first_or_404()
    responses = Response.query.filter_by(form_id=form.id, user_id=session['user_id']).all()
    response_ids = [r.id for r in responses]

    try:
        if response_ids:
            Answer.query.filter(Answer.response_id.in_(response_ids)).delete(
                synchronize_session=False
            )
        Response.query.filter_by(form_id=form.id, user_id=session['user_id']).delete()
        Question.query.filter_by(form_id=form.id).delete()
        db.session.delete(form)
        db.session.commit()
        flash('Form deleted successfully.', 'success')
    except Exception:
        db.session.rollback()
        flash('Failed to delete form. Please try again.', 'error')

    return redirect(url_for('index'))


@app.route('/form/<int:form_id>', methods=['GET', 'POST'])
@login_required
def fill_form(form_id):
    form = Form.query.filter_by(id=form_id, user_id=session['user_id']).first_or_404()
    questions = Question.query.filter_by(form_id=form_id).all()
    closed = is_form_closed(form)

    if request.method == 'POST':
        if closed:
            flash('This form is closed and no longer accepts responses.', 'error')
            return redirect(url_for('fill_form', form_id=form_id))
        anonymous = request.form.get('anonymous') == 'on'
        typed_name = request.form.get('username', '').strip()
        if anonymous:
            username = 'Anonymous'
        else:
            username = (typed_name or session.get('username', 'Anonymous'))[:100]
        response = Response(form_id=form_id, username=username, user_id=session['user_id'])
        db.session.add(response)
        db.session.commit()

        for q in questions:
            if q.qtype == 'checkbox':
                raw_values = [v.strip() for v in request.form.getlist(f'q_{q.id}') if v.strip()]
                value = ', '.join(raw_values)[:300]
            else:
                value = (request.form.get(f'q_{q.id}') or '').strip()[:300]
            db.session.add(Answer(response_id=response.id, question_id=q.id, value=value))

        db.session.commit()
        flash('Response submitted successfully.', 'success')
        return redirect(url_for('index'))

    return render_template(
        'fill_form.html',
        form=form,
        questions=questions,
        public_mode=False,
        share_url=build_public_url('public_form', form_id=form.id),
        is_closed=closed,
    )


@app.route('/respond/<int:form_id>', methods=['GET', 'POST'])
def public_form(form_id):
    form = Form.query.get_or_404(form_id)
    questions = Question.query.filter_by(form_id=form_id).all()
    closed = is_form_closed(form)

    if request.method == 'POST':
        if closed:
            flash('This form is closed and no longer accepts responses.', 'error')
            return redirect(url_for('public_form', form_id=form_id))
        anonymous = request.form.get('anonymous') == 'on'
        typed_name = request.form.get('username', '').strip()

        if anonymous:
            username = 'Anonymous'
        elif typed_name:
            username = typed_name[:100]
        elif session.get('username'):
            username = session.get('username')
        else:
            username = 'Anonymous'

        response = Response(form_id=form_id, username=username, user_id=form.user_id)
        db.session.add(response)
        db.session.commit()

        for q in questions:
            if q.qtype == 'checkbox':
                raw_values = [v.strip() for v in request.form.getlist(f'q_{q.id}') if v.strip()]
                value = ', '.join(raw_values)[:300]
            else:
                value = (request.form.get(f'q_{q.id}') or '').strip()[:300]
            db.session.add(Answer(response_id=response.id, question_id=q.id, value=value))

        db.session.commit()
        flash('Feedback submitted. Thank you.', 'success')
        return redirect(url_for('public_form', form_id=form_id))

    return render_template(
        'fill_form.html',
        form=form,
        questions=questions,
        public_mode=True,
        share_url=None,
        is_closed=closed,
    )


@app.route('/summary/<int:form_id>')
@login_required
def summary(form_id):
    form = Form.query.filter_by(id=form_id, user_id=session['user_id']).first_or_404()
    questions = Question.query.filter_by(form_id=form_id).all()
    responses = Response.query.filter_by(form_id=form_id, user_id=session['user_id']).order_by(Response.id.asc()).all()
    response_ids = [r.id for r in responses]
    answers = Answer.query.filter(Answer.response_id.in_(response_ids)).all() if response_ids else []

    stats = {}
    for q in questions:
        values = [a.value for a in answers if a.question_id == q.id and a.value]
        if q.qtype in ('rating', 'mcq', 'checkbox'):
            if q.qtype == 'checkbox':
                expanded = []
                for val in values:
                    expanded.extend([v.strip() for v in val.split(',') if v.strip()])
                values = expanded
            stats[q.id] = {
                'mode': 'counts',
                'counts': dict(Counter(values)),
                'total': len(values),
            }
        else:
            stats[q.id] = {
                'mode': 'text',
                'entries': values,
                'total': len(values),
            }

    answer_map = {(a.response_id, a.question_id): a.value for a in answers}
    identified_submissions = []
    anonymous_submissions = []

    for response in responses:
        submitted_by = (response.username or '').strip() or 'Anonymous'
        row = {
            'response_id': response.id,
            'submitted_by': submitted_by,
            'answers': [answer_map.get((response.id, q.id), '') for q in questions],
        }
        if submitted_by.lower() == 'anonymous':
            anonymous_submissions.append(row)
        else:
            identified_submissions.append(row)

    return render_template(
        'summary.html',
        form=form,
        questions=questions,
        stats=stats,
        total_responses=len(response_ids),
        identified_submissions=identified_submissions,
        anonymous_submissions=anonymous_submissions,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
