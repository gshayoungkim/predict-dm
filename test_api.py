import requests

url = "http://localhost:8000/predict"
data = {
    "nPregnancies": 6,
    "GlucoseConcentration": 148,
    "BP": 72,
    "SkinThickness": 35,
    "SerumInsulin": 125,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}

try:
    response = requests.post(url, json=data)
    print("응답 결과:")
    print(response.json())
except Exception as e:
    print(f"에러 발생: {e}")
