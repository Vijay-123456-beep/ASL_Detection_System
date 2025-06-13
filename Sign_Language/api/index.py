from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "ASL Detection API is running!"})

# Add your model loading and prediction endpoints here

if __name__ == "__main__":
    app.run()
