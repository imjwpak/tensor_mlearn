from mnist_test.number_checker import NumberChecker
from mnist_test.fashion_checker import FashionChecker

if __name__ == '__main__':
    #nc = NumberChecker()
    #nc.execute()

    fc = FashionChecker()
    fc.create_model()
