from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the flask API"

@app.route("/about")
def about():
    return "This is a simple route Example to About Page"

@app.route("/contact")
def contact():
    return "<h2>Contact Us</h2><p>hasnainmalik925@gmail.com</p>"

if __name__ == "__main__":
        app.run(debug=True, use_reloader=False)
