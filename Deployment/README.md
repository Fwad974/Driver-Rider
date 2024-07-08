# Completion Prediction Deployment Co
This is Ride Completion Prediction Deployment codes



## Usage

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r req.txt
```

for using fastapi application, send your requests to 0.0.0.0:8000/model/predict by using post method, and your description of the object should be named as description and your request should be sth like:
```python
{
    "dest": 11
    "origin": 810
    "Time": 770
    "Income": 10000
    "Comment": "ABC"
    "Comment": "This is a sample comment.",
    "Created_at": "2023-07-08T12:30:00"  # Example datetime string in ISO 8601 format
}
```

