from flask import Flask, render_template, request, redirect, url_for
import subprocess

from data_collection import capture_images, split_dataset  # Import your functions from data_collection.py

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Retrieve form data
        name = request.form.get("name")
        user_id = request.form.get("user_id")

        # Call the functions from data_collection.py
        try:
            user_images_path = capture_images(name, user_id)
            split_dataset(user_images_path)
        except Exception as e:
            return f"Error during data collection: {str(e)}", 500

        # Redirect to a success page
        return redirect(url_for("success", name=name, id=user_id))

    # Render the form on GET request
    return render_template("index.html")

@app.route("/start_attendance", methods=["POST"])
def start_attendance():
    try:
        # Invoke the prediction.py script
        subprocess.Popen(["python", "prediction.py"])
        return "Attendance system started. Please check the window displaying real-time recognition."
    except Exception as e:
        return f"Failed to start attendance: {str(e)}", 500

@app.route("/success")
def success():
    name = request.args.get("name")
    user_id = request.args.get("id")
    return f"Data collection successful for {name} (ID: {user_id})!"

if __name__ == "__main__":
    app.run(debug=True)
