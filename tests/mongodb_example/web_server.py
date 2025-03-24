'''
# web_server.py
from flask import Flask, request, jsonify
from pymongo import MongoClient
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, filename='flask.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
mongo_port = os.getenv("MONGO_PORT", "27017")
flask_port = int(os.getenv("FLASK_PORT", "5000"))  # 通过环境变量设置端口

try:
    client = MongoClient(f"mongodb://127.0.0.1:{mongo_port}/", serverSelectionTimeoutMS=10000)
    client.server_info()  # 测试连接
    logging.info(f"Connected to MongoDB on port {mongo_port}")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB on port {mongo_port}: {e}")
    raise

db = client["test_db"]
collection = db["test_collection"]

@app.route("/set", methods=['GET'])
def set_key():
    key = request.args.get("key")
    val = request.args.get("val")
    if not key or not val:
        logging.error("Missing key or val parameter")
        return jsonify({"error": "Missing key or val parameter"}), 400
    try:
        collection.update_one({"key": key}, {"$set": {"value": val}}, upsert=True)
        logging.info(f"Set key={key}, value={val}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error(f"Error in set_key: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get", methods=['GET'])
def get_key():
    key = request.args.get("key")
    if not key:
        logging.error("Missing key parameter")
        return jsonify({"error": "Missing key parameter"}), 400
    try:
        val = collection.find_one({"key": key})
        result = val["value"] if val else "null"
        logging.info(f"Get key={key}, value={result}")
        return jsonify({"key": key, "value": result}), 200
    except Exception as e:
        logging.error(f"Error in get_key: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info(f"Starting Flask server on port {flask_port}")
    app.run(host="127.0.0.1", port=flask_port, debug=False)
'''
from flask import Flask, request
from pymongo import MongoClient
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, filename='/tmp/flask.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
mongo_port = os.getenv("MONGO_PORT", "27017")
try:
    client = MongoClient(f"mongodb://127.0.0.1:{mongo_port}/", serverSelectionTimeoutMS=5000)
    client.server_info()  # 测试连接
    logging.info(f"Connected to MongoDB on port {mongo_port}")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB on port {mongo_port}: {e}")
    raise

db = client["test_db"]
collection = db["test_collection"]

@app.route("/set")
def set_key():
    key = request.args.get("key")
    val = request.args.get("val")
    try:
        collection.update_one({"key": key}, {"$set": {"value": val}}, upsert=True)
        logging.info(f"Set key={key}, value={val}")
        return "OK"
    except Exception as e:
        logging.error(f"Error in set_key: {e}")
        return "Error", 500

@app.route("/get")
def get_key():
    key = request.args.get("key")
    try:
        val = collection.find_one({"key": key})
        result = val["value"] if val else "null"
        logging.info(f"Get key={key}, value={result}")
        return result
    except Exception as e:
        logging.error(f"Error in get_key: {e}")
        return "Error", 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
