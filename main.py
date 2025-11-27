from fastapi import FastAPI, HTTPException
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
