class RamPrice:
    def __init__(self):
        pass

    def execute(self):
        # 여기에 import한 것은 그냥 공부하기 위한 것이니까 여기다 넣었음
        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import mglearn
        import numpy as np

        ram_price = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
        # plt.semilogy(ram_price.date, ram_price.price)
        # plt.xlabel("년")
        # plt.ylabel("가격")
        # plt.show()

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression

        data_train = ram_price[ram_price['date'] < 2000]  # 2000년 기준으로 2000년 이전 것으로 train 시킴
        data_test = ram_price[ram_price['date'] >= 2000]  # 2000년 이후로 test

        x_train = data_train['date'][:, np.newaxis]  # train data를 1열로 만든다
        y_train = np.log(data_train['price'])

        tree = DecisionTreeRegressor().fit(x_train, y_train)
        lr = LinearRegression().fit(x_train, y_train)

        # test는 모든 데이터에 대해 적용한다
        x_all = ram_price['date'].values.reshape(-1, 1)  # x_all을 1열로 만든다
        pred_tree = tree.predict(x_all)
        price_tree = np.exp(pred_tree)  # log 값 되돌리기
        pred_lr = lr.predict(x_all)
        price_lr = np.exp(pred_lr)  # log 값 되돌리기

        plt.semilogy(ram_price['date'], pred_tree,
                     label="TREE PREDICT", ls='-', dashes=(2, 1))
        plt.semilogy(ram_price['date'], pred_lr,
                     label="LINEAR REGRESSION", ls='-', dashes=(2, 1))
        plt.semilogy(data_train['date'], data_train['price'], label="TRAIN DATA", alpha=0.4)
        plt.semilogy(data_test['date'], data_test['price'], label="TEST DATA")

        plt.legend(loc=1)
        plt.xlabel('year', size=15)
        plt.ylabel('price', size=15)
        plt.show()





