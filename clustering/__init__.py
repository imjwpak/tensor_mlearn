import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_regression(n_neighbors=1) # 점들을 한 곳으로 회귀 시켜 군집을 만듬. 가까운 곳으로 모은다고 보면 됨.
#mglearn.plots.plot_knn_classification() # 산점도 등에서 카테고리화하여 군집 만듬.


# 이웃한 값(nn)이 1일 때 알고리즘
plt.show()



