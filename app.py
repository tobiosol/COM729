import os
import sys
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

from fundus_v2 import fundus_model_manager

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return redirect(request.url)
        
        selected_model = request.form.get("model")
        predicted_class, prediction_matrix = fundus_model_manager.FundusModelManager.predict(file, selected_model)
        print(f"Predicted class: {predicted_class}")
        print(f"Prediction matrix: {prediction_matrix}")

        
        return render_template("result.html", predicted_class=predicted_class,
                                prediction_matrix=prediction_matrix, selected_model=selected_model)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)