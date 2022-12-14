#%%
import pandas
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 데이터 정의
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25, random_state=0)


# %%
# 잠재요인 협업필터링 수행
# 순서 : 알고리즘 정의 > 학습 > 예측(test: 전체 데이터셋, predict: 개별사용자)
# 잠재요인은 알 수 없는 값임

# 알고리즘 정의 및 학습
algo = SVD()
algo.fit(trainset)

#%%
# 전체 예측 (test)

preds = algo.test(testset) # 테스트 세트 전체에 대한 추천 영화 평점 데이터 생성
print("prediction type:", type(preds), "size:", len(preds)) # 2만5천개인 이유는 10만개 데이터중 25%
print("prediction 결과의 최초 5개 추출")
print(preds[:5], '\n')
# uid = 유저아이디
# iid = 아이템아이디
# r_ui = 실제평점
# est = 예측 평점 (새롭게 만든 예측 데이터의 값)

# 반환된 객체 값 가져오기
print([(pred.uid, pred.iid, pred.est) for pred in preds[:3]])

# # print([(pred.uid=='196', pred.iid=='302', pred.est) for pred in preds])
# for pred in preds:
#     if (pred.uid=='196') & (pred.iid=='302'):
#         print(pred.r_ui)

#%%
# 개별 예측 (predict)
# 개별 사용자에 대한 아이템 예측 평점을 반환
# 사용자 아이디와 아이템 아이디를 입력해야 하는데 문자열 형태로 입력해야 함
# predict(uid(문자열), iid(문자열), r_ui(선택사항))

uid = str(196) # 196번 사용자
iid = str(302) # 302번 아이템

pred = algo.predict(uid, iid)
print(pred)
# %%
# 정확도평가 : 실제평점과 예측평점간의 차이를 검사
# (test로 반환된 예측평점)

accuracy.rmse(preds)

# %%

# 주요 API
# 1. Dataset.load_builtin : 데이터 내려받기
# 2. Dataset.load_from_file(file_path, reader) : os파일에서 데이터를 로딩
#     - file_path : os파일명
#     - reader : 파일의 포맷
# 3. Dataset.load_from_df(df, reader) : 판다스 데이터프레임에서 데이터 로딩
#     - 로우 레벨 데이터(pivot_table로 사용자, 아이템, 레이팅과 같은 형태로 나타낼 수 있는 dataframe)만을 취급(컬럼의 순서도 중요함)
#     - df : 데이터프레임(userId, itemId, rating 순으로 컬럼이 정해져 있어야 함)
#     - reader : 파일의 포맷

# %%
# os파일 데이터 surprise데이터 세트로 로딩
import pandas as pd

# 데이터 불러오기
ratings = pd.read_csv(r'C:\sbbigdata\Restaurant_Recommendation\data\grouples_ratings.csv')

# 컬럼명 제거 후 데이터 저장
ratings.to_csv(r'C:\sbbigdata\Restaurant_Recommendation\data\ratings_noh.csv', index=False, header=False)

# %%
# os파일 불러오기
# os파일을 불러오기 위해서는 불러오려는 데이터프레임의 컬럼, 평점 관련 정보를 입력해야함
from surprise import Reader

reader = Reader(line_format="user item rating timestamp", sep=',', rating_scale=(0.5, 5))
data = Dataset.load_from_file(r"C:\sbbigdata\Restaurant_Recommendation\data\ratings_noh.csv", reader=reader)

trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD()
algo.fit(trainset)
preds = algo.test(testset)

accuracy.rmse(preds)

# %%
# 판다스 데이터프레임에서 Surprise세트로 로딩

import pandas as pd
from surprise import Reader

ratings = pd.read_csv(r'C:\sbbigdata\surprise_recomendation\data\grouples_ratings.csv')

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)

trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD()
algo.fit(trainset)
preds = algo.test(testset)

accuracy.rmse(preds)

# %%
# svd parameter
# n_factors = 잠재요인개수 (개수 달라져도 성능 향상 거의 없음. -> 그냥 안만지면 됨)
# n_epochs = SGD수행시 반복 횟수
# biased = 베이스라인 사용자 편향 적용 여부, 디폴트는 True

#%%
# BaseLine평점 -> 사용자의 성향을 반영하여 예측 평점 계산
#   - 전체 평균 평점 + 사용자 편향 점수 + 아이템 편향 점수
#   - 전체 평균 평점 : 모든 사용자의 아이템에 대한 평점을 평균한 값 (데이터 프레임 전체의 평점의 평균)
#   - 사용자 편향 점수 : 사용자별 아이템 평점 평균 값 - 전체 평균 평점
#   - 아이템 편향 점수 : 아이템별 사용자들의 평점 평균 - 전체 평균 평점
#   - 예시 :
#        - 전체 평균 평점(DataFrame의 rating값 평균) : 3.5
#        - 어벤져스(아이템)의 평균 평점 : 4.2
#        - 윤대호의 평균 평점 : 3.0
#        - 윤대호는 어벤져스에 몇 점을 줄까? : 3.5 + (3.0 - 3.5) + (4.2 - 3.5) = 3.7

#%%
# Surprise 교차검증 및 하이퍼파라미터 튜닝 cross_validation

import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate

ratings = pd.read_csv(r"C:\sbbigdata\surprise_recomendation\data\grouples_ratings.csv")
reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)

algo = SVD()

cvDict = cross_validate(algo, data = data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
cvDictDf = pd.DataFrame(cvDict)
cvDictDf
# %%
cvDictDf1 = cvDictDf.T
cvDictDf1.columns = ['Fold1','Fold2','Fold3','Fold4','Fold5']
cvDictDf1
# %%
# GridSearchCV
# epoch = 20, 40, 60
# factor = 50, 100, 200

import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import GridSearchCV

ratings = pd.read_csv(r"C:\sbbigdata\surprise_recomendation\data\grouples_ratings.csv")
reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)

# algo = SVD()
paramGrid = {'n_epochs' : [20, 40, 60], 'n_factors' : [50, 100, 200]}

svdGrid = GridSearchCV(algo_class=SVD, param_grid=paramGrid, cv=3, measures=['RMSE', 'MAE'])

svdGrid.fit(data=data)
# %%
print(svdGrid.best_params)
print(svdGrid.best_score)

gridSearchResultDf = pd.DataFrame(svdGrid.cv_results)[['params', 'rank_test_rmse', 'rank_test_mae']]
gridSearchResultDf.head()
# %%
