# Drape

From the project directory, serve the frontend in one terminal.

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

In another terminal, use the existing backend environment. Install the BMNet
and recommendation-model dependencies once. Extract the trained Random Forest
from your project's trusted `check fml.zip` into `models/`. Download the DeepLabV3
segmentation weights to `models/torch-checkpoints/` with the virtual environment's
certificate bundle.

```bash
cd backend
.venv/bin/python -m pip install -r body_shape/requirements.txt -r requirements-recommendations.txt
unzip -jn "$HOME/Downloads/check fml.zip" 'check fml/compatibility_class_model.joblib' -d ../models
SSL_CERT_FILE="$(.venv/bin/python -m certifi)" .venv/bin/python -c 'from body_shape.service import get_person_segmenter; get_person_segmenter()'
.venv/bin/python -c 'import os, getpass, uvicorn; os.environ["ROBOFLOW_API_KEY"] = getpass.getpass("Roboflow API key: "); uvicorn.run("main:app", host="127.0.0.1", port=8001)'
```

Open [Drape Studio](http://127.0.0.1:8000/#studio). Use a front-facing face photo,
a head-to-toe body photo, and choose the men's or women's collection and a design level.
The selection is sent to both BMNet and the catalogue audience filter.
When body inference is unavailable, Drape shows that status and matches all body
classes consistent with the available face, skin, and style profile.

## Source files

| File | Role |
| --- | --- |
| `frontend/index.html`, `frontend/studio.js` | Collect photos and preferences, then show the profile and results. |
| `backend/main.py` | Validate uploads and pass the model outputs to the recommender. |
| `backend/body_shape/service.py` and `weights/body_shape_front.pt` in the same package | Segment the person, run the supplied BMNet model, and classify the resulting measurements. |
| `backend/ml_recommendations.py` | Match the profile and design level to a class, filter by catalogue audience, and rank up to five distinct listings. |
| `data/classes_data.csv`, `data/final_dataset.csv`, `models/compatibility_class_model.joblib` | Supply the 150 class definitions, 5,000 products, and the trained Random Forest. |
| `evaluation/recommendation_model_metrics.csv` | Preserve the supplied Random Forest evaluation on its noisy labels. |

The analysis endpoint calls
`recommend_outfits([skin_color, face_shape, body_shape, design_level, gender])`.
The selected collection filters the `gender` column in `data/final_dataset.csv`.
Catalogue labels come from product metadata, with `unknown` used for insufficient evidence.
At startup, Drape reads the Random Forest's exported global weights from the saved
bundle. A request filters by the profile's matching class and chosen collection.
Eligible products are ranked by their final score, calculated as
W1 × skin compatibility + W2 × face compatibility + W3 × body compatibility.
Design preference is used to match the class. When the body estimate is
unavailable, Drape uses W1 and W2 and matches all eligible body classes.
