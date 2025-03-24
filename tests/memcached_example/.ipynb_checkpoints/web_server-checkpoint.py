from flask import Flask, request
import pylibmc

app = Flask(__name__)
mc = pylibmc.Client(["127.0.0.1"], binary=True)

@app.route("/get")
def get():
    key = request.args.get("key", "hello")
    value = mc.get(key)
    return value if value else "N/A"

@app.route("/set")
def set_key():
    key = request.args.get("key", "hello")
    val = request.args.get("val", "world")
    mc.set(key, val)
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
