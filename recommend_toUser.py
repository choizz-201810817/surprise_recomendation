# 특정 유저가 보지 않은 영화들의 평점을 예측하여 예측 평점이 높은 순서대로 top10의 영화를 추천
# surprise 라이브러리를 활용한 잠재요인 협업 필터링 추천 시스템
#%%
import pandas as pd
import numpy as np

from surprise.dataset import DatasetAutoFolds
from surprise import Dataset
from surprise import Reader
from surprise import SVD

#%%
# train dataset 정의
# DatasetAutoFolds는 전체 데이터를 train dataset으로 만들기 위해서 사용함
reader = Reader(line_format="user item rating", sep=',', rating_scale=(0.5, 5))
dataFolds = DatasetAutoFolds(ratings_file=r'C:\sbbigdata\surprise_recomendation\data\ratings_noh.csv', reader=reader)
trainset = dataFolds.build_full_trainset()

# 알고리즘 정의
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

# %%
# 9번 사용자에게 영화를 추천
# 9번 사용자가 평점을 매기지 않은 영화를 추천
# 9번 사용자가 평점을 매기지 않은 영화를 추출
# 9번 사용자가 본 영화를 확인
moviesDf = pd.read_csv(r"C:\sbbigdata\surprise_recomendation\data\grouplens_movies.csv")
ratingsDf = pd.read_csv(r"C:\sbbigdata\surprise_recomendation\data\grouples_ratings.csv")
print("movies shape :", moviesDf.shape)
print("ratings shape :", ratingsDf.shape)

# 9번 사용자가 보지 않은 영화 추출
def unseenMovies(df, user_id):
    userSeen = df[df['userId']==user_id]['movieId'].values

    # 9번 사용자가 평점을 매기지 않은 영화들
    userNotSeen = moviesDf[~moviesDf['movieId'].isin(userSeen)]['movieId'].values
    print('userId :', user_id)
    print(f"평점 매긴 영화 수 :", len(userSeen), "전체 영화 수 :", len(moviesDf), "추천해야 할 영화 수 :", len(userNotSeen))

    return userNotSeen.tolist()


# 해당 유저가 안 본 영화에 대해 높은 평점 순으로 영화 추천
## 영화추천
def recTop10ByUser(notSeen, algo, uid):
    notSeenList = []
    ests = []
    # movieRankByUser = []

    for i in notSeen:
        iid = str(i)
        notSeenList.append(algo.predict(uid, iid))

    for notSeen in notSeenList:
        ests.append((notSeen.est, notSeen.iid))

    ests.sort(reverse=True) # est기준 내림차순 정렬

    recomend = ests[:10]
    
    return recomend

def recommendResult(allDf, userTop10):
    movieIdList = []
    etcList = []
    
    for tu in userTop10:
        etcList.append(tu[0])
        movieIdList.append(int(tu[1]))

    top10Dict = {'movieId' : movieIdList, 'rate' : etcList}
    etcRateDf = pd.DataFrame(top10Dict)

    top10Df = allDf[allDf.movieId.isin(movieIdList)].drop('genres', axis=1)

    recoDf = pd.merge(top10Df, etcRateDf, on='movieId')
    
    return recoDf

# %%
# 9번 유저에게 맞는 top10 영화 추천
notSeen_9 = unseenMovies(ratingsDf, 9)
user9Top10 = recTop10ByUser(notSeen_9, algo, 9)

recommendResult(moviesDf, user9Top10)
