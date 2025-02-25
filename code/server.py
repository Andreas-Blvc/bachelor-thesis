from flask import Flask, request, jsonify, send_from_directory
import os
import json
from flask_cors import CORS

app = Flask(__name__, static_folder="frontend/dist")  # Serve React frontend
CORS(app, origins=["http://localhost:5173"])  # Enable CORS for frontend-backend communication

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)


@app.route("/save_road", methods=["POST"])
def save_road():
    """Save road data as JSON."""
    data = request.json
    filename = data.get("filename", "road") + ".json"
    filepath = os.path.join(DATA_FOLDER, filename)

    with open(filepath, "w") as file:
        json.dump(data, file)

    return jsonify({"message": "Road saved successfully!", "file": filename})


@app.route("/load_road", methods=["GET"])
def load_road():
    """Load road data from JSON."""
    filename = request.args.get("filename", "road.json")
    filepath = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    with open(filepath, "r") as file:
        data = json.load(file)

    return jsonify(data)


@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1337)
