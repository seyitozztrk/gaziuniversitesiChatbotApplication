#imports
from flask import Flask, render_template, request
import main


app = Flask(__name__)



#define app routes
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(main.chat(userText))
if __name__ == "__main__":
    app.run()