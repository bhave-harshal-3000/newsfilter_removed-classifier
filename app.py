from flask import Flask, render_template, request, redirect, url_for, flash
from news_ai_agent import process_and_send

app = Flask(__name__)
app.secret_key = "_secret_key"  # Change this for production

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form.get("email")
        region = request.form.get("region")
        content_type = request.form.get("content_type")
        top_n = int(request.form.get("top_n", 10))  # <-- Add this line
        sources = request.form.getlist("sources")
        status = process_and_send(email, region, content_type, top_n, sources)  # <-- Pass top_n
        flash(status)
        return redirect(url_for("index"))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)