import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

print("모델 학습을 시작합니다...")

# 데이터 로드
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
column_names = ['nPregnancies', 'GlucoseConcentration', 'BP', 'SkinThickness',
                'SerumInsulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# 0값을 NaN으로 변경
zero_cols = ['GlucoseConcentration', 'BP', 'SkinThickness', 'SerumInsulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

# Median으로 결측치 채우기
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

# 특성과 타겟 분리
X = df[['nPregnancies', 'GlucoseConcentration', 'BP', 'SkinThickness',
        'SerumInsulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

# StandardScaler 학습
print("데이터 스케일링 중...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 최종 모델 학습 (당신의 노트북에서 가장 좋았던 모델)
print("Gradient Boosting 모델 학습 중...")
model = GradientBoostingClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    random_state=42
)
model.fit(X_scaled, y)

# 모델 성능 확인
train_score = model.score(X_scaled, y)
print(f"학습 정확도: {train_score:.4f}")

# 모델 및 스케일러 저장
print("모델 저장 중...")
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✓ 모델과 스케일러가 성공적으로 저장되었습니다!")
print("✓ diabetes_model.pkl")
print("✓ scaler.pkl")
