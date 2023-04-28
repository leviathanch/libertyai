import mimetypes
import os
import time
import requests
import uuid
from tempfile import mkdtemp

from gtts import gTTS
from io import BytesIO

mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.ts')
mimetypes.add_type('text/css', '.css')

from gevent.pywsgi import WSGIServer

from flask import (
    Flask,
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    Response,
    session,
    g,
    make_response,
)

from flask_session import Session

from flask_login import (
    login_required,
    current_user,
    LoginManager,
    login_user,
    logout_user,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.wsgi import wrap_file

from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy

from LibertyAI.liberty_chatbot import initialize_chatbot
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_llm import LibertyLLM

config = get_configuration()
SQLALCHEMY_DATABASE_URI = 'postgresql://'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_USER') + ':'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_PASSWORD') + '@'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_SERVER') + ':'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_SERVER_PORT') + '/'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_DATABASE')

app = Flask(__name__)

#app.secret_key = config.get('DEFAULT', 'SECRET_KEY')
#app.config['SECRET_KEY'] = config.get('DEFAULT', 'SECRET_KEY')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SECRET_KEY"] = os.urandom(24)
#app.config["session_cookie_name"] = 'session_cookie_name'
app.session_cookie_name = "session_cookie_name"

#app.app_context().push()
#SESSION_TYPE = 'memcache'

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(80), unique=False, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), unique=False, nullable=False)
    active = db.Column(db.Boolean, unique=False, nullable=False, default = False)
    admin = db.Column(db.Boolean, unique=False, nullable=False, default = False)
    _authenticated = False

    def is_active(self):
        return True # self.active
    
    def get_id(self):
        return int(self.id)
    
    def is_authenticated(self):
        return self._authenticated

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

sess = Session(app)

# ------------------------
@app.route("/")
def home():
    return redirect("/chatbot", code=302)

@app.route('/login', methods=['POST'])
def login_post():
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
    except:
        flash('Please check your login details and try again.')
        return redirect(url_for('login')) # if the user doesn't exist or password is wrong, reload the page

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password.strip()):
        # if the above check passes, then we know the user has the right credentials
        login_user(user, remember=remember)
        session["user_id"] = user.id
        return redirect(url_for('chatbot'))
    else:
        flash('Please check your login details and try again.')
        return redirect(url_for('login')) # if the user doesn't exist or password is wrong, reload the page

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/signup', methods=['POST'])
def signup_post():
    # code to validate and add user to database goes here
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

    if user: # if a user is found, we want to redirect back to signup page so user can try again
        return redirect(url_for('signup'))

    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    pw = generate_password_hash(password.strip())
    new_user = User(email=email, name=name, password=pw)

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('login'))

# ----------------------------
@app.route('/profile')
@login_required
def profile():
    return render_template("profile.html", name=current_user.name, email=current_user.email)

@app.route('/profile/username', methods=['POST'])
@login_required
def post_profile_username():
    name = request.form.get('name')
    name = name.replace("'",'').replace(";",'').replace('"','')
    if len(name) > 0:
        current_user.name = name;
    db.session.commit()
    return redirect(url_for('profile'))

@app.route('/profile/avatar', methods=['POST'])
@login_required
def post_profile_avatar():
    #name = request.form.get('name')
    #name = name.replace("'",'').replace(";",'').replace('"','')
    #if len(name) > 0:
    #    current_user.name = name;
    #db.session.commit()
    return redirect(url_for('profile'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('chatbot'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=session["user_id"]).first()

# ------------------------
@app.route("/voicechat")
@login_required
def voicechat():
    return render_template("voicechat.html", username=current_user.name)

current_audio_files = {}
current_input_audio_files = {}

@app.route('/voicechat/submit', methods=['POST'])
@login_required
def post_voicechat_submit():
    if 'audio_data' in request.files:
        uid = str(uuid.uuid1())
        f = request.files['audio_data']
        current_input_audio_files[uid] = f.read()
        return uid
    return "Error"

@app.route('/voicechat/stream')
@login_required
def voicechat_stream():
    try:
        uid = request.args.get('uuid')
    except:
        return Response('data: [ERROR]\n\n', mimetype="text/event-stream")

    def eventStream(bot, uid):
        response = requests.post(
            config.get('API', 'WHISPER_ENDPOINT'),
            files = {'file': current_input_audio_files[uid]},
        )
        language = 'en'
        text = ''
        reply = response.json()
        if 'results' in reply:
            for result in reply['results']:
                language = result['language']
                text = result['text']
            print("TTS: "+text)
            #bot = current_chatbot(session["user_id"])
            text = bot(text)
            onsei = gTTS(text=text, lang=language)
            buf = BytesIO()
            onsei.write_to_fp(buf)
            current_audio_files[uid] = buf.getvalue()
            buf.close()
            del current_input_audio_files[uid]
            yield 'data: [DONE]\n\n'

        yield 'data: [ERROR]\n\n'

    return Response(eventStream(current_chatbot(session["user_id"]), uid), mimetype="text/event-stream")

@app.route('/voicechat/audio')
@login_required
def voicechat_audio():
    try:
        uid = request.args.get('uuid')
    except:
        return "ERROR!"

    response = make_response(current_audio_files[uid])
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
    del current_audio_files[uid]

    return response

# ------------------------
@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", username=current_user.name)

@app.route("/chatbot/get_chat_history")
@login_required
def chatbot_get_chat_history():
    return current_chatbot(session["user_id"]).chat_history()

@app.route("/chatbot/start_generation")
@login_required
def chatbot_start_generation():
    try:
        message = request.args.get('msg')
    except:
        return ""

    uid = current_chatbot(session["user_id"]).start_generations(message)

    return uid if uid else ""

@app.route('/chatbot/stream')
@login_required
def chatbot_stream():
    try:
        uid = request.args.get('uuid')
    except:
        return Response('data: [DONE]\n\n', mimetype="text/event-stream")

    def eventStream(bot):
        index = 0
        token = ""
        while token != "[DONE]":
            print(token)
            token = bot.get_part(uid, index)
            if token == "[BUSY]":
                time.sleep(0.1)
            else:
                index += 1
                yield 'data: {}\n\n'.format(token)

    return Response(eventStream(current_chatbot(session["user_id"])), mimetype="text/event-stream")

active_chatbots = {}
def current_chatbot(user_id):
    global active_chatbots
    current_user = load_user(user_id)
    if current_user.is_authenticated:
        if current_user.id not in active_chatbots:
            active_chatbots[current_user.id] = initialize_chatbot(
                name = current_user.name,
                email = current_user.email,
                llm = LibertyLLM(endpoint = config.get('API', 'GENERATION_ENDPOINT')),
                emb = LibertyEmbeddings(endpoint = config.get('API', 'EMBEDDING_ENDPOINT')),
                sqlstring = SQLALCHEMY_DATABASE_URI,
            )
        return active_chatbots[current_user.id]
    return None

# --------

if __name__ == "__main__":
    http_server = WSGIServer(('',5000), app)
    http_server.serve_forever()
