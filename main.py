from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import uvicorn

# FastAPI 앱 초기화
app = FastAPI(
    title="당뇨병 예측 API",
    description="환자 정보를 입력하면 당뇨병 발생 가능성을 예측합니다",
    version="1.0.0"
)

# 모델과 스케일러 로드
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("모델이 성공적으로 로드되었습니다!")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None
    scaler = None

# 요청 데이터 모델 정의
class DiabetesInput(BaseModel):
    nPregnancies: int = Field(..., description="임신 횟수", ge=0)
    GlucoseConcentration: float = Field(..., description="포도당 농도", ge=0)
    BP: float = Field(..., description="혈압", ge=0)
    SkinThickness: float = Field(..., description="피부 두께", ge=0)
    SerumInsulin: float = Field(..., description="인슐린", ge=0)
    BMI: float = Field(..., description="체질량지수", ge=0)
    DiabetesPedigreeFunction: float = Field(..., description="당뇨 가족력", ge=0)
    Age: int = Field(..., description="나이", ge=0)

# 응답 데이터 모델
class PredictionResponse(BaseModel):
    prediction: int  # 0 또는 1
    probability: float  # 당뇨병 발생 확률
    risk_level: str  # 위험도


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>당뇨병 예측 시스템</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; text-align: center; }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #666;
                font-weight: bold;
            }
            input {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                padding: 12px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover { background: #45a049; }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .success { background: #d4edda; color: #155724; }
            .warning { background: #fff3cd; color: #856404; }
            .danger { background: #f8d7da; color: #721c24; }
            .links {
                text-align: center;
                margin-top: 20px;
            }
            .links a {
                color: #4CAF50;
                text-decoration: none;
                margin: 0 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 당뇨병 예측 시스템</h1>
            <form id="predictionForm">
                <div class="form-group">
                    <label>임신 횟수</label>
                    <input type="number" id="nPregnancies" value="6" required>
                </div>
                <div class="form-group">
                    <label>포도당 농도</label>
                    <input type="number" step="0.1" id="GlucoseConcentration" value="148" required>
                </div>
                <div class="form-group">
                    <label>혈압</label>
                    <input type="number" step="0.1" id="BP" value="72" required>
                </div>
                <div class="form-group">
                    <label>피부 두께</label>
                    <input type="number" step="0.1" id="SkinThickness" value="35" required>
                </div>
                <div class="form-group">
                    <label>인슐린</label>
                    <input type="number" step="0.1" id="SerumInsulin" value="125" required>
                </div>
                <div class="form-group">
                    <label>체질량지수 (BMI)</label>
                    <input type="number" step="0.1" id="BMI" value="33.6" required>
                </div>
                <div class="form-group">
                    <label>당뇨 가족력</label>
                    <input type="number" step="0.001" id="DiabetesPedigreeFunction" value="0.627" required>
                </div>
                <div class="form-group">
                    <label>나이</label>
                    <input type="number" id="Age" value="50" required>
                </div>
                <button type="submit">예측하기</button>
            </form>
            
            <div id="result"></div>
            
            <div class="links">
                <a href="/docs" target="_blank">📚 API 문서</a>
                <a href="/health" target="_blank">🏥 상태 확인</a>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const data = {
                    nPregnancies: parseInt(document.getElementById('nPregnancies').value),
                    GlucoseConcentration: parseFloat(document.getElementById('GlucoseConcentration').value),
                    BP: parseFloat(document.getElementById('BP').value),
                    SkinThickness: parseFloat(document.getElementById('SkinThickness').value),
                    SerumInsulin: parseFloat(document.getElementById('SerumInsulin').value),
                    BMI: parseFloat(document.getElementById('BMI').value),
                    DiabetesPedigreeFunction: parseFloat(document.getElementById('DiabetesPedigreeFunction').value),
                    Age: parseInt(document.getElementById('Age').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    let className = 'success';
                    if (result.risk_level === '중간') className = 'warning';
                    if (result.risk_level === '높음') className = 'danger';
                    
                    resultDiv.className = className;
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `
                        <h3>예측 결과</h3>
                        <p><strong>당뇨병 여부:</strong> ${result.prediction === 1 ? '있음' : '없음'}</p>
                        <p><strong>발생 확률:</strong> ${(result.probability * 100).toFixed(2)}%</p>
                        <p><strong>위험도:</strong> ${result.risk_level}</p>
                    `;
                } catch (error) {
                    alert('예측 중 오류가 발생했습니다: ' + error);
                }
            });
        </script>
    </body>
    </html>
    """

    
# 헬스 체크
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

# 예측 엔드포인트
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: DiabetesInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    try:
        # 입력 데이터를 배열로 변환
        input_data = np.array([[
            data.nPregnancies,
            data.GlucoseConcentration,
            data.BP,
            data.SkinThickness,
            data.SerumInsulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])
        
        # 스케일링
        scaled_data = scaler.transform(input_data)
        
        # 예측
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        # 위험도 판정
        if probability < 0.3:
            risk_level = "낮음"
        elif probability < 0.7:
            risk_level = "중간"
        else:
            risk_level = "높음"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=round(float(probability), 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

# 배치 예측
@app.post("/predict/batch")
async def predict_batch(data_list: List[DiabetesInput]):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    results = []
    for data in data_list:
        input_data = np.array([[
            data.nPregnancies, data.GlucoseConcentration, data.BP,
            data.SkinThickness, data.SerumInsulin, data.BMI,
            data.DiabetesPedigreeFunction, data.Age
        ]])
        
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        results.append({
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        })
    
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
