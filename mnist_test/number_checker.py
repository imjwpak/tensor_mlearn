class NumberChecker:
    def __init__(self):
        pass

    def execute(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from tensorflow import keras

        mnist = keras.datasets.mnist
        (train_images, train_lables), (test_images, test_lables) = mnist.load_data()

        '''
        print("훈련 이미지: ", train_images.shape)
        print("훈련 레이블: ", train_lables.shape)
        print("테스트 이미지: ", test_images.shape)
        print("테스트 레이블: ", test_lables.shape)
        print("\n")
        '''

        plt.figure(figsize=(5, 5))
        image = train_images[100]
        plt.imshow(image)
        plt.show()






