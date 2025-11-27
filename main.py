from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import uvicorn

app = FastAPI(
    title="당뇨병 예측 API",
    description="환자 정보를 입력하면 당뇨병 발생 가능성을 예측합니다",
    version="1.0.0"
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("모델이 성공적으로 로드되었습니다!")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None
    scaler = None

class DiabetesInput(BaseModel):
    nPregnancies: int = Field(..., description="임신 횟수", ge=0)
    GlucoseConcentration: float = Field(..., description="포도당 농도", ge=0)
    BP: float = Field(..., description="혈압", ge=0)
    SkinThickness: float = Field(..., description="피부 두께", ge=0)
    SerumInsulin: float = Field(..., description="인슐린", ge=0)
    BMI: float = Field(..., description="체질량지수", ge=0)
    DiabetesPedigreeFunction: float = Field(..., description="당뇨 가족력", ge=0)
    Age: int = Field(..., description="나이", ge=0)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str

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
            button:disabled { background: #ccc; cursor: not-allowed; }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .success { background: #d4edda; color: #155724; }
            .warning { background: #fff3cd; color: #856404; }
            .danger { background: #f8d7da; color: #721c24; }
            .error { background: #f8d7da; color: #721c24; }
            .links {
                text-align: center;
                margin-top: 20px;
            }
            .links a {
                color: #4CAF50;
                text-decoration: none;
                margin: 0 10px;
            }
            .loading {
                text-align: center;
                color: #666;
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
                <button type="submit" id="submitBtn">예측하기</button>
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
                
                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');
                
                // 버튼 비활성화
                submitBtn.disabled = true;
                submitBtn.textContent = '예측 중...';
                
                // 로딩 표시
                resultDiv.style.display = 'block';
                resultDiv.className = 'loading';
                resultDiv.innerHTML = '<p>⏳ 예측 중입니다...</p>';
                
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
                
                console.log('전송 데이터:', data);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    console.log('응답 상태:', response.status);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    console.log('응답 데이터:', result);
                    
                    // 데이터 검증
                    if (result.prediction === undefined || result.probability === undefined || result.risk_level === undefined) {
                        throw new Error('서버 응답이 올바르지 않습니다');
                    }
                    
                    let className = 'success';
                    if (result.risk_level === '중간') className = 'warning';
                    if (result.risk_level === '높음') className = 'danger';
                    
                    resultDiv.className = className;
                    resultDiv.innerHTML = `
                        <h3>예측 결과</h3>
                        <p><strong>당뇨병 여부:</strong> ${result.prediction === 1 ? '있음 ⚠️' : '없음 ✅'}</p>
                        <p><strong>발생 확률:</strong> ${(result.probability * 100).toFixed(2)}%</p>
                        <p><strong>위험도:</strong> ${result.risk_level}</p>
                    `;
                    
                } catch (error) {
                    console.error('에러 발생:', error);
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `
                        <h3>오류 발생</h3>
                        <p>예측 중 오류가 발생했습니다: ${error.message}</p>
                        <p>브라우저 콘솔(F12)을 확인하세요.</p>
                    `;
                } finally {
                    // 버튼 다시 활성화
                    submitBtn.disabled = false;
                    submitBtn.textContent = '예측하기';
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: DiabetesInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    try:
        # 입력 데이터 로깅
        print(f"받은 데이터: {data}")
        
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
        
        print(f"변환된 배열: {input_data}")
        
        scaled_data = scaler.transform(input_data)
        print(f"스케일링된 데이터: {scaled_data}")
        
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        print(f"예측 결과 - prediction: {prediction}, probability: {probability}")
        
        if probability < 0.3:
            risk_level = "낮음"
        elif probability < 0.7:
            risk_level = "중간"
        else:
            risk_level = "높음"
        
        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
        
        print(f"최종 응답: {response}")
        return response
    
    except Exception as e:
        print(f"예측 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
