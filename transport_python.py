import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #선형 회귀 모델

csv_path = '20251126213153.csv'
try:
	df = pd.read_csv(csv_path, encoding='utf-8-sig')
	print(f"Loaded '{csv_path}' with encoding='utf-8-sig'")
except UnicodeDecodeError:
	# 한국 환경에서 생성된 CSV는 cp949로 인코딩된 경우가 많습니다.
	print(f"utf-8-sig 로 읽는 데 실패하여 'cp949'로 재시도합니다: {csv_path}")
	df = pd.read_csv(csv_path, encoding='cp949')

# seed 값 설정
seed = 0
np.random.seed(seed)

dataSet = df.values
X = dataSet[:, 1:2]
Y = dataSet[:, 3]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = seed)
transportModel = LinearRegression(fit_intercept=True)
model = transportModel.fit(X_train, Y_train)

print("정확도(R^2) =", model.score(X_train, Y_train))
w = model.coef_
b = model.intercept_
print("가중치:", w)
print("편향:", b)

Y_prediction = model.predict(X_test).flatten()
# 테스트 샘플이 10개 미만일 수 있으므로 안전하게 출력 개수를 제한
count_show = min(10, len(Y_test), len(Y_prediction))
for i in range(count_show):
		label = Y_test[i]
		prediction = Y_prediction[i]
		print("실제 수송량: {:.3f}, 학습후 예측수송량: {:.3f}".format(label, prediction))