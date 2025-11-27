import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

print(f"현재 작업 디렉토리: {os.getcwd()}")

try:
    # 데이터 로드
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
    column_names = ['nPregnancies', 'GlucoseConcentration', 'BP', 'SkinThickness',
                    'SerumInsulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    print("데이터 로드 중...")
    df = pd.read_csv(url, names=column_names)
    
    # 0값을 NaN으로 변경
    zero_cols = ['GlucoseConcentration', 'BP', 'SkinThickness', 'SerumInsulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
    
    # Median으로 결측치 채우기
    df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())
    
    # 특성 분리
    X = df[['nPregnancies', 'GlucoseConcentration', 'BP', 'SkinThickness',
            'SerumInsulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df['Outcome']
    
    # StandardScaler 생성 및 학습
    print("Scaler 학습 중...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 모델 학습
    print("모델 학습 중...")
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # 저장
    print("파일 저장 중...")
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(model, 'diabetes_model.pkl')
    
    # 확인
    if os.path.exists('scaler.pkl'):
        print(f"✓ scaler.pkl 생성 완료! (크기: {os.path.getsize('scaler.pkl')} bytes)")
    else:
        print("✗ scaler.pkl 생성 실패!")
        
    if os.path.exists('diabetes_model.pkl'):
        print(f"✓ diabetes_model.pkl 생성 완료! (크기: {os.path.getsize('diabetes_model.pkl')} bytes)")
    else:
        print("✗ diabetes_model.pkl 생성 실패!")
    
    # 로드 테스트
    print("\n파일 로드 테스트...")
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_model = joblib.load('diabetes_model.pkl')
    print("✓ 모든 파일이 정상적으로 로드됩니다!")
    
except Exception as e:
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()
