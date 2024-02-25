import webbrowser

from flask import Flask, render_template
from modules import *

app = Flask(__name__)

# Dummy URL for the API - replace with the actual URL
MODEL_PATH = get_resource(r"lib\models\amazon_reviews_5k_model.pkl")
PRODUCTS_PATH = get_resource(r"lib\datasets\amazon_reviews_5k.csv")
FEATURES_PATH = get_resource(r"lib\datasets\amazon_reviews_5k.pkl")
NO_IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac' \
               '/No_image_available.svg/300px-No_image_available.svg.png'

@app.route('/')
def home():
    # uncomment to evaluate top proucts on the fly
    # top_products = get_top_products(model_path=MODEL_PATH,
    #                         data_path=PRODUCTS_PATH,
    #                                 features_path=FEATURES_PATH,
    #                         n=5)
    top_products = get_top_products_display()
    top_products['image'] = top_products['image'].apply(lambda x: x[2:-2] if
    x[1:-1] else NO_IMAGE_URL)
    top_products = top_products.to_dict(orient='records')[::-1]
    return render_template(r'index.html',
                           products=top_products)

def main():
    """
    Main function to run the app
    :return:
    """
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(host="127.0.0.1", port=5000)

if __name__ == '__main__':
    main()
    # app.run(debug=True)
