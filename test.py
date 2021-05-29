import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from card_extractor import CardExtractor
from card_determiner import CardDeterminer
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
cd = CardDeterminer()
ce = CardExtractor()
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


# Save query image
img = Image.open('test.jpg')  # PIL image
uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + 'test.jpg'
img.save(uploaded_img_path)

determine = cd.detect_text(uploaded_img_path);
card = ce.extract(uploaded_img_path)

card = Image.open(uploaded_img_path)
# Run search
query = fe.extract(card)
dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
ids = np.argsort(dists)[:30]  # Top 30 results
scores = [(dists[id], img_paths[id]) for id in ids]
