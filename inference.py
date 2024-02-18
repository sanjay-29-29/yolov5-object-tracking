from main import detect
from flask import Flask, Response

app = Flask(__name__)

@app.route('/webcam')
def webcam_display():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)