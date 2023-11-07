from flask import Flask, redirect, url_for, render_template, request, session
from flask_caching import Cache
import os
import datetime
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import cluster_it
import pickle
from PIL import Image

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ndagnklda_ggdaa'
    return app

app = create_app()
#app.secret_key = 'ndagnklda_ggdaa'

app.app_context().push()
app.permanent_session_lifetime = timedelta(minutes=300)
cache = Cache(app)

#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
#app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
#db = SQLAlchemy(app)

root_path = app.root_path
os.chdir(root_path)
print(os.getcwd())

# class users(db.Model):
#     _id = db.Column("id", db.Integer, primary_key=True)
#     name = db.Column("name", db.String(100))
#     email = db.Column("email", db.String(100))
    
#     def __init__(self, name, email):
#         self.name = name
#         self.email = email

@app.route("/")
def home():
    session.clear()
    return redirect(url_for("variables"))

@app.route("/variables", methods = ["POST", "GET"])
def variables():
    if request.method == "POST":
        if request.form['submit'] == 'Submit':
            color = request.form.getlist('Colorcheck')
            session["colorscaling"] = color[0]
            type = request.form.getlist('Typecheck')
            session["typescaling"] = type[0]
            text = request.form.getlist('Textcheck')
            session["textscaling"] = text[0]
            deck = request.form.getlist('Deckcheck')
            session["deckscaling"] = deck[0]
            formats = request.form.getlist('Formatcheck')
            session["formats"] = formats
            format_string = ''
            for each in formats:
                format_string = format_string + each[:2]
            colors = request.form.getlist('WhichColorcheck')
            for count,each in enumerate(colors):
                if each == "White":
                    colors[count] = "W"
                if each == "Blue":
                    colors[count] = "U"
                if each == "Black":
                    colors[count] = "B"
                if each == "Red":
                    colors[count] = "R"
                if each == "Green":
                    colors[count] = "G"
            session["colors"] = colors
            color_string = ''
            for each in colors:
                color_string = color_string + each[:2]
            rarity = request.form.getlist('Raritycheck')
            session["rarities"] = rarity
            rarity_string = ''
            for each in rarity:
                rarity_string = rarity_string + each[:2]
            session['code_string'] = f"{color[0]}{type[0]}{text[0]}{deck[0]}{format_string}{color_string}{rarity_string}"
            return redirect(url_for("results"))
    return render_template("variable_setup.html")
    
# @app.route("/variables_update", methods = ["POST", "GET"])
# def variables_update():
#     color = request.form.getlist('Colorcheck')
#     session["color"] = color[0]
#     return redirect(url_for("results"))

@app.route("/results", methods = ["POST", "GET"])
def results():
    infocolor = session["colorscaling"]
    infotype = session["typescaling"]
    infotext = session["textscaling"]
    infodeck = session["deckscaling"]
    infoformat = session["formats"]
    infocolors = session["colors"]
    inforarity = session["rarities"]
    cluster_it._cluster_this(float(infocolor),float(infotype),float(infotext),float(infodeck),infoformat,infocolors,inforarity,session['code_string'])
    return render_template('cluster.html',variable = session['code_string'])

@app.route("/graph", methods = ["POST", "GET"])
def graph():
    string = os.getcwd()
    with open(f"{string}\\static\\images\\{session['code_string']}_list", 'rb') as pickle_file:
        values = pickle.load(pickle_file)
    if request.method == "POST":
        if request.form['submit'] == 'Submit':
            card_1 = request.form["card"]
            if(card_1 in values[1]):          
                length_to_save = 5
                cluster_it.Create_graph(values[3], -1,card_1, session['code_string'], length_to_save)
                cluster_it.Create_graph(values[3], -1,card_1, session['code_string'], length_to_save)         
                created = True
                return render_template('graph.html',variable = session['code_string'], names = values[1], card1 = card_1, evaluated = created)
            else:
                return("invalid card_name")
    else:
        created = False
        return render_template('graph.html',variable = session['code_string'], names = values[1], card1 ='', evaluated = created)

@app.route('/search', methods=['POST','GET'])    
def search():  # Get the query parameter
    string = os.getcwd()
    if request.method == 'POST':
        if request.form['search'] == 'search':
            try:  
                with open(f"{string}\\static\\images\\{session['code_string']}_list", 'rb') as pickle_file:
                    values = pickle.load(pickle_file)
                    query = request.form['searchbox']
                    print(type(query))
                    print("this is the query:" + query)
                    results = [item for item in values[1] if item.lower().startswith(query.lower())]
                    #print('got_here')
                    return render_template('card_search.html', results=results) 
            except Exception as e:  
               return render_template("variable_setup.html")
        
@app.route('/image_route')
def image_route():
    string = os.getcwd()
    image_id = request.args.get('image_id', default=None, type=str)
    with open(f"{string}\\static\\images\\{session['code_string']}_list", 'rb') as pickle_file:
        values = pickle.load(pickle_file)
    relevant = values[3][image_id]
    # Now you can use image_id, for example to retrieve some data to pass to your template
    return render_template('image_route.html', image_id=image_id, relevant = relevant)

@app.route("/comparison", methods = ["POST", "GET"])
def comparison():
    string = os.getcwd()
    with open(f"{string}\\static\\images\\{session['code_string']}_list", 'rb') as pickle_file:
        values = pickle.load(pickle_file)
    if request.method == "POST":
        if request.form['submit'] == 'Submit':
            card_1 = request.form["card1"]
            info1 = values[3][card_1]
            #cluster_it._return_image(info1[6],card_1,info1[5])
            card_2 = request.form["card2"]
            info2 = values[3][card_2]
            #cluster_it._return_image(info2[6],card_2,info2[5])
            vector = request.form["vector_to_choose"]
            if(card_1 in values[3]):
                created = True 
                result_vector = cluster_it._Compare_cards(values[3],card_1,card_2,int(vector))
                return render_template('comparison.html',variable = session['code_string'], names = values[1], card_1 = card_1, card_2 = card_2, result = result_vector, evaluated = created)
            else:
                return("invalid card_name")
    else:
        created = False
        return render_template('comparison.html',variable = session['code_string'], names = values[1], card_1 = '', card_2 = '', result = [],evaluated = created)

@app.route("/comparison_all", methods = ["POST", "GET"])
def comparison_all():
    string = os.getcwd()
    with open(f"{string}\\static\\images\\{session['code_string']}_list", 'rb') as pickle_file:
        values = pickle.load(pickle_file)
    if request.method == "POST":
        if request.form['submit'] == 'Submit':
            card_1 = request.form["card1"]
            info1 = values[3][card_1]
            vector = request.form["vector_to_choose"]
            if(card_1 in values[1]):
                created = True 
                result_vector = cluster_it._Compare_all_cards(values[3],card_1,int(vector))
                return render_template('comparison_all.html',variable = session['code_string'], names = values[1], card_1 = card_1, list = result_vector, evaluated = created)
            else:
                return("invalid card_name")
    else:
        created = False
        return render_template('comparison_all.html',variable = session['code_string'], names = values[1], card_1 = '', card_2 = '', list = [], evaluated = created)


if __name__ == "__main__":
    app.run(debug=True)
    