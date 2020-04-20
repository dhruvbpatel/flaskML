import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'RFModel.pckl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        # Extract the input
        sepal_length = flask.request.form['sepal_length']
        sepal_width = flask.request.form['sepal_width']
        petal_length = flask.request.form['petal_length']
        petal_width = flask.request.form['petal_width']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[sepal_length,sepal_width, petal_length,petal_width]],
                                       columns=['sepal_length', 'sepal_width', 'petal_length','petal_width'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's `prediction
        prediction = model.predict(input_variables)[0]

        # Render the form again, but add in the pred`iction and remind user
        # of the values they input before
        return flask.render_template('index.html',
                                     original_input={'sepal_length':sepal_length,
                                                     'sepal_width':sepal_width,
                                                     'petal_length':petal_length,
                                                     'petal_width':petal_width},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(debug=True)
