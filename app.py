from flask import Flask, render_template, request, redirect, url_for, flash
from news_ai_agent import process_and_send

app = Flask(__name__)
app.secret_key = "_secret_key"  # Change this for production

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form.get("email")
        category = request.form.get("category", "general")
        
        region = None
        if category == "higher_ed":
            region = request.form.get("higher_ed_region")
        elif category == "entertainment":
            region = request.form.get("entertainment_region")
        elif category == "sports":
            region = request.form.get("sports_region")
        elif category == "business_and_finance":
            region = request.form.get("business_finance_region")
        elif category == "environment":
            region = request.form.get("region")
        elif category == "industry":
            region = request.form.get("region")
        else: # general
            region = request.form.get("region")

        content_type = request.form.get("content_type")
        top_n = int(request.form.get("top_n", 10))
        sources = request.form.getlist("sources")
        status = process_and_send(email, category, region, content_type, top_n, sources)
        flash(status)
        return redirect(url_for("index"))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)