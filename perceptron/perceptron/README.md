# Perceptron Digit Classifier

Small FastAPI web app for predicting whether an uploaded PNG digit is `0` or `1`.

## Project Structure

```text
percep/
  app/
    main.py                 # FastAPI routes and app setup
    services/
      classifier.py         # Model loading, image preprocessing, prediction
    models/
      model.pkl             # Saved perceptron model
    templates/
      index.html            # Web page
    static/
      style.css             # UI styling
    samples/
      image.png             # Test/sample images
      images.png
  classifirer.py            # Backward-compatible import wrapper
  requirements_fastapi.txt  # Web app dependencies
```

## Run

```bash
.venv/bin/python -m uvicorn percep.app.main:app --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001`.
