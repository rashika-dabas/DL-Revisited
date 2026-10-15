import rmse

def test_rmse():
    predictions = [1, 2, 3, 4, 5]
    targets = [1, 2, 3, 4, 5]

    assert rmse.rmse(predictions, targets) == 0

    predictions = [2, 3, 4, 5, 6]
    targets = [1, 2, 3, 4, 5]

    assert rmse.rmse(predictions, targets) == 1

# To run tests, run this file from the command line with `python rmse_test.py` after uncommenting the lines below
# or use the command `pytest rmse_test.py` to run the tests.

# if __name__ == "__main__":
#     test_rmse()