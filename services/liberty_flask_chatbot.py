from gevent.pywsgi import WSGIServer

from flask import (
    Flask,
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
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

from LibertyAI import LibertyChatBot
from LibertyAI.liberty_config import get_configuration

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

@app.route('/index')
def index():
    return render_template("index.html")

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
        active_bots[user.id] = LibertyChatBot()
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

@app.route('/logout')
@login_required
def logout():
    if current_user.id in active_bots:
        del active_bots[current_user.id]
    logout_user()
    return redirect(url_for('index'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")

# ------------------------

@app.route("/chatbot/get")
@login_required
def get_bot_response():
    message = request.args.get('msg')
    #docs_with_score: List[Tuple[Document, float]] = db.similarity_search_with_score(message)
    #docs_with_score = sorted(docs_with_score, key=lambda x: x[1], reverse=True)
    #reply = docs_with_score[0][0].page_content
    #result = bot.chat(message)
    #reply = result["answer"]
    if current_user.id not in active_bots:
        active_bots[current_user.id] = LibertyChatBot()

    reply = active_bots[current_user.id].chat(message)

    #reply = conversation.run(input=message, context="", stop=['Human:'])

    return reply


if __name__ == "__main__":
    active_bots = {}
    app.run(host='0.0.0.0', port=5000)
    #http_server = WSGIServer(('', int(config.get('DEFAULT', 'ModelServicePort'))), app)
    #http_server.serve_forever()

'''
    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)
'''

