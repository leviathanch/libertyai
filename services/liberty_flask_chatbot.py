import mimetypes
import os
import time

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
)

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

from flask_sqlalchemy import SQLAlchemy

from LibertyAI.liberty_chatbot import initialize_chatbot
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_llm import LibertyLLM

from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

config = get_configuration()
SQLALCHEMY_DATABASE_URI = 'postgresql://'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_USER') + ':'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_PASSWORD') + '@'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_SERVER') + ':'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_SERVER_PORT') + '/'
SQLALCHEMY_DATABASE_URI += config.get('DATABASE', 'PGSQL_DATABASE')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-goes-here'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI

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
        active_bots[user.id] = initialize_chatbot(
            name=user.name,
            email=user.email,
            llm = llm,
            emb = emb,
        )
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
    return render_template("profile.html", name=current_user.name)

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
    if current_user.id in active_bots:
        del active_bots[current_user.id]
    logout_user()
    return redirect(url_for('chatbot'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", username=current_user.name)

# ------------------------

@app.route("/chatbot/start_generation")
@login_required
def chatbot_start_generation():
    try:
        message = request.args.get('msg')
    except:
        return ""

    if current_user.id not in active_bots:
        active_bots[current_user.id] = initialize_chatbot(
            name = current_user.name,
            email = current_user.email,
            llm = llm,
            emb = emb,
        )

    uuid = active_bots[current_user.id].start_generations(message)

    return uuid if uuid else ""

@app.route('/chatbot/stream')
@login_required
def chatbot_stream():
    try:
        uuid = request.args.get('uuid')
    except:
        return Response('data: [DONE]\n\n', mimetype="text/event-stream")

    def eventStream(bot):
        index = 0
        token = ""
        keep_alive_count = 0
        while token != "[DONE]":
            token = bot.get_part(uuid, index)
            if token == "[BUSY]":
                if keep_alive_count < 30:
                    time.sleep(1)
                    keep_alive_count += 1
                else:
                    keep_alive_count = 0
                    yield 'data: [KEEP_ALIVE]\n\n'
            else:
                index += 1
                yield 'data: {}\n\n'.format(token)

    return Response(eventStream(active_bots[current_user.id]), mimetype="text/event-stream")


def liberty_llm():
    return LibertyLLM(
        endpoint = config.get('API', 'GENERATION_ENDPOINT'),
    )

def liberty_embedding():
    return LibertyEmbeddings(
        endpoint = config.get('API', 'EMBEDDING_ENDPOINT'),
    )

if __name__ == "__main__":
    llm = liberty_llm()
    emb = liberty_embedding()
    active_bots = {}
    active_conversations = {}
    http_server = WSGIServer(('',5000), app)
    http_server.serve_forever()
