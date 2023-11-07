from flask import Flask, redirect, url_for, render_template, request, session, flash
import os
import os.path
import datetime
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import yaml

def create_app():
    app = Flask(__name__)
    return app

app = create_app()
app.app_context().push()

db_path = os.path.join(os.path.dirname(__file__), 'users.db')
db_uri = 'sqlite:///'.format(db_path)


# import yaml
# config_filename = "local.yaml"

# base_directory = path = os.path.dirname(os.path.realpath(__file__))

# with open(base_directory + "/config/" + config_filename) as config_file:
#     config = yaml.load(config_file)

app.secret_key = 'magic_stuff'
app.permanent_session_lifetime = timedelta(minutes=1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# db_config = config['database']
db = SQLAlchemy(app)
db.app = app

# def clear_the_template_cache():
#     app.jinja_env.cache = {}

# app.before_request(clear_the_template_cache)
db = SQLAlchemy(app)

def load_user_email(email):
    from models import users
    return users.query.filter_by(email=email).first()

def load_user_name(name):
    from models import users
    return users.query.filter_by(name=name).first()

@app.route("/")
def home():
    return redirect(url_for("login"))
    #return render_template("index.html")

@app.route("/test")
def test():
    return render_template("new.html")

@app.route("/login", methods =["POST", "GET"])
def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form["nm"]
        session["user"] = user
        found_user = load_user_name(user)
        if found_user:
            session["email"] = found_user.email
        else:
            from models import users
            usr = users(user, "")
            db.session.add(usr)
            db.session.commit()
        flash("login successful", "info")
        return redirect(url_for("user"))
    else:
        if "user" in session:
            flash("already logged in", "info")
            return redirect(url_for("user"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    if "user" in session:
        user = session["user"]
        flash(f"You have been logged out, {user} ", "info")
    session.pop("user", None)
    session.pop("email", None)
    return redirect(url_for("login"))

@app.route("/user", methods = ["POST", "GET"])
def user():
    email = None
    if "user" in session:
        user = session["user"]
        if request.method == "POST":
            email = request.form["email"]
            session["email"] = email
            found_user = load_user_name(user)
            found_user.email = email
            db.session.commit()
            flash("email was saved")
        else:
            if email in session:
                email = session["email"]
        return render_template("user.html", email = email)
    else:
        flash("you are not logged in")
        return redirect(url_for("login"))
    
if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)