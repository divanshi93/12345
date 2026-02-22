from flask import Flask, render_template, request
import os
from organic_avocado_sales_forecasting import run_forecast

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    file = request.files['file']
    path = os.path.join("static", "uploaded.csv")
    file.save(path)

    result = run_forecast(path)  # Call your forecasting function
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
