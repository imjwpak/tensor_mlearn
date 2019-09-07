class Mammal:
    def __init__(self):
        pass

    @staticmethod
    def execute():
        import tensorflow as tf
        import numpy as np
        # 원핫 인코딩 예제
        # [털, 날개] -> [기타, 포유류, 조류]
        '''
            [[0, 0],    -> [1, 0, 0] 둘다 없으면 기타 
             [1, 0],    -> [0, 1, 0] 포유류
             [1, 1],    -> [0, 0, 1] 조류
             [0, 0],    -> [1, 0, 0] 기타 
             [0, 0],    -> [1, 0, 0] 기타
             [0, 1]     -> [0, 0, 1] 조류
             ]
        '''
        x_data = np.array(
            [[0, 0],
             [1, 0],
             [1, 1],
             [0, 0],
             [0, 0],
             [0, 1]
             ]
        )
        # 답을 무조건 하나만 찍는 것임
        y_data = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]
             ]
        )
        # 외부에서 받는 값 placeholder
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)

        W = tf.Variable(tf.random_uniform([2, 3], -1, 1.)) # 정규분포에 맞는 값을 랜덤으로 뽑으라는 의미.
        # -1은 all
        # 신경망 neural network 앞으로는 nn으로 표기
        # nn은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정합니다
        b = tf.Variable(tf.zeros([3])) # 초기값은 0
        
        # b는 편향 bias
        # W는 가중치 weight
        # b는 각 레이어의 아웃풋 갯수로 설정함
        # b는 최종 결과값의 분류갯수인 3으로 설정함

        # 이게 뉴런 하나 만든 것
        L = tf.add(tf.matmul(X, W), b) # y = WX + b

        # 가중치와 편향을 이용해 계산한 결과 값에
        # relu 활성함수
        # 뉴런 하나를 만들어서 신경망에 넣어준 것
        # nn은 다층 신경망을 통과 시켰다고 이해하면 됨
        L = tf.nn.relu(L)
        model = tf.nn.softmax(L)

        # 외부값이 들어왔을 때 기타, 조류, 포유류 중 어떤게 가장 확률이 높다고 알려주는 모델을 만든 것
        '''
        softmax 소프트맥스 함수는 다음처럼 결과값을 전체 합이 1인 확률로 만들어 주는 함수
        예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23]
        '''
        print('-------- 모델 내부 보기 --------')
        print(model)
        # 여기까지 모델 만들고
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # 비용함수를 최소화 시키면(=경사도를 0으로 만들면) 그 값이 최적화된 값이다.
        train_op = optimizer.minimize(cost) # 기울기가 0이 된 것을 찾는 것
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for step in range(100):
            sess.run(train_op, {X: x_data, Y: y_data})
            if (step + 1) % 10 == 0:
                print(step + 1, sess.run(cost, {X: x_data, Y: y_data}))
        # 여기까지 학습

        # 결과 확인
        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y, 1)
        print('예측값: ', sess.run(prediction, {X: x_data}))
        print('실제값: ', sess.run(target, {Y: y_data}))
        
        # tf.argmax : 예측값과 실제값의 행렬에서 tf.argmax를 이용해 가장 큰 값의 위치값(인덱스)을 가져옴
        # 예) [[0, 1, 1][1, 0, 0]] -> [1, 0]
        # [[0.2, 0.7, 0.1][0.9, 0.1, 0.]] -> [1, 0]
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도 : %.2f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))


