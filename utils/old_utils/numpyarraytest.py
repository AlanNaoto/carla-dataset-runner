import numpy as np


def test_array(input1, input2):
    input1 = input1.ravel()  # Found the solution... Might just need to get this data correct formatted later on
    input2 = input2.ravel()
    try:
        print('input1 shape: {0} input2 shape: {1}'.format(input1.shape, input2.shape))
        bb = np.array([input1, input2])
    except:
        return "Fail"
    return 'Success'


if __name__ == "__main__":
    bb_walkers = np.array([-1, -1, -1, -1])

    # case1: deu boa
    bb_vehicles = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(test_array(bb_walkers, bb_vehicles))

    # case2: deu ruim
    bb_vehicles = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(test_array(bb_walkers, bb_vehicles))

    # case3: deu boa
    bb_vehicles = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(test_array(bb_walkers, bb_vehicles))

    # case4: deu boa
    bb_vehicles = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    print(test_array(bb_walkers, bb_vehicles))

