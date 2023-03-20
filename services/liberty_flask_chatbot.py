from gevent.pywsgi import WSGIServer

from flask import (
    Flask,
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
)

from flask_login import (
    login_required,
    current_user,
    LoginManager,
)

from werkzeug.security import generate_password_hash

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
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    def __repr__(self):
        return '<User %r>' % self.name

db.init_app(app)
#db.create_all()


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
    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('login'))

# ----------------------------
@app.route('/profile')
@login_required
def profile():
    return render_template("profile.html")

@app.route('/logout')
@login_required
def logout():
    return 'Logout'

@login_manager.user_loader
def load_user(user_id):
    return user_id

@app.route("/chatbot")
#@login_required
def chatbot():
    return render_template("chatbot.html")

# ------------------------

@app.route("/chatbot/get")
#@login_required
def get_bot_response():
    message = request.args.get('msg')
    #docs_with_score: List[Tuple[Document, float]] = db.similarity_search_with_score(message)
    #docs_with_score = sorted(docs_with_score, key=lambda x: x[1], reverse=True)
    #reply = docs_with_score[0][0].page_content
    #result = bot.chat(message)
    #reply = result["answer"]
    reply = bot.chat(message)

    #reply = conversation.run(input=message, context="", stop=['Human:'])

    return reply


if __name__ == "__main__":
    bot = LibertyChatBot()
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

