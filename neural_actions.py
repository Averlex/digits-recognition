# A module containing the operations with neural network and its data

from neural import *
import numpy as np


# A process of training the network (long version)
def train_nn(n, fpath):
    # Setting the variable for time measurements
    est_time = lambda t: datetime.now() - t

    # Reading the train file data
    try:
        training_data_file = open(fpath, 'r')
    except OSError:
        print("An error occurred during reading the file. Please, try again or change the file name.\n")
        return

    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # Additional variables: epoch and total time spent
    cur_time = datetime.now()
    first_time = datetime.now()

    # Training the net
    for e in range(n.get_epochs()):
        print("Epoch", e + 1, "out of", n.get_epochs(), "is processing...")
        cur_time = datetime.now()
        for record in training_data_list:
            # Reorganizing data to lists
            all_values = record.split(',')
            # Scaling the matrix values to 0.01 : 1.00 range
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # Defining output nodes as 0's adjusted to 0.01
            targets = np.zeros(n.get_onodes()) + 0.01

            # Defining an output node for the current number as 1 with adjustment
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
        print("Epoch", e + 1, "is done.", n.get_epochs() - e - 1, "more to go. Required time:",
              round(est_time(cur_time).total_seconds(), 2), "s")

    # Printing the neural network parameters
    minutes, seconds = divmod(est_time(first_time).total_seconds(), 60)
    print("\nThe neural network is ready for use.")
    print("Total number of input nodes:", n.get_inodes())
    print("Total number of hidden nodes:", n.get_hnodes())
    print("Total number of output nodes:", n.get_onodes())
    print("Learning rate:", n.get_lr())
    print("Epochs proceeded:", n.get_epochs())
    print("Total time required:", int(minutes), "minutes", "{:.2f}".format(float(seconds)), "seconds.", "\n")

    return


# A function for testing the network performance
def test_nn(n, fpath):
    # Reading the test file data
    try:
        test_data_file = open(fpath, 'r')
    except OSError:
        print("An error occurred during reading the file. Please, try again or change the file name.\n")
        return
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # An array for result evaluation
    scorecard = []
    efficiency = 0.0

    for record in test_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # Getting the result and its index therefore its value
        result = n.query(inputs)
        label = np.argmax(result)
        correct_label = int(all_values[0])

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = np.asarray(scorecard)
    efficiency = round(scorecard_array.sum() / scorecard_array.size * 100, 2)
    print("The neural network efficiency: ", efficiency, "%\n")

    return


# Manual configuration
def configure(hid_nodes, learn_rate, ep):
    internal_nodes = 0
    lr_coef = 0.0
    epoch_num = 0

    internal_nodes = int(input("Number of nodes in the hidden layer: "))
    lr_coef = float(input("Learning rate: "))
    epoch_num = int(input("Number of epochs: "))
    train_path = input("Path to the training source file(.csv): ")
    test_path = input("Path to the testing source file(.csv): ")

    if internal_nodes <= 0 or lr_coef <= 0.0 or lr_coef >= 1.0 or epoch_num <= 0:
        print("\nSome of the values are incorrect. The default parameters were set.\n")
        return None
    else:
        print("\nThe configuration completed successfully.\n")

    return [internal_nodes, lr_coef, epoch_num, train_path, test_path]


# Reading the random 10 numbers from the test file
def ten_random_queries(n, fpath):
    # Reading the test file data
    test_data_file = open(fpath, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    outputs = []
    # Generating an array of non-repeatable random ints
    rng = np.random.default_rng(seed=datetime.now().second)
    indices = rng.choice(len(test_data_list), size=10, replace=False)

    # Performing queries only for those records that were chosen before
    final_records = []
    counter = 0
    for record in test_data_list:
        if counter in indices:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # Getting the result and its index therefore its value
            result = n.query(inputs)
            outputs.append(result)
            final_records.append(record.split(","))

        counter += 1

    # Processing the results of recognition, reshaping the source record
    final_outputs = []
    for counter in range(len(outputs)):
        final_records[counter] = np.asfarray(final_records[counter][1:]).reshape((28, 28))
        final_outputs.append(np.argmax(outputs[counter]))

    return [final_records, final_outputs]


# Decoding the neural network revere query result (transforming it into an image)
def decode_rev_query(inputs):
    res = np.asfarray((inputs.transpose() - 0.01) / 0.99 * 255).reshape((28, 28))

    return res


# Transforming the digit into a neural network output format
def proc_digit(dig):
    arr = np.zeros(10, dtype=float)

    if dig < 0 or dig > 9:
        return None
    else:
        arr += 0.01
        for i in range(10):
            if i == dig:
                arr[i] = 0.99

    arr.transpose()
    return arr


# Getting a set of 10 digits via reverse querying
def get_some_digits(n):
    # Filling the source array with 0 to 9 digits
    # Transforming each number in a (10, 1) associated matrix matching the neural network outputs
    source = [i for i in range(10)]
    source_mat = [proc_digit(arr) for arr in source]
    res = []
    for i in source_mat:
        # Getting scaled matrix with a shape of 28*28
        tmp = decode_rev_query(n.reverse_query(i))
        res.append(tmp)

    return res


# A query for logging the results of training
def save_logs(n):
    ch = input("Export results of the training to .csv file? (y/n): ")
    if ch == "y" or ch == "Y":
        tmp = n.export_weights()
        wih = tmp[0]
        who = tmp[1]

        np.savetxt("wih.csv", wih, delimiter=",")
        np.savetxt("who.csv", who, delimiter=",")
        print("Successful export.\n")
    else:
        print("No results were exported.\n")

    return


# Optimizing the learning procedure via dumped log
def quick_train(n):
    ch = input("Do you want to upload a preset configuration from logs? (y/n): ")
    if ch == "y" or ch == "Y":
        fname1 = "wih.csv"
        fname2 = "who.csv"
        try:
            wih = np.loadtxt(fname1, delimiter=",", dtype=float)
            who = np.loadtxt(fname2, delimiter=",", dtype=float)
        except OSError:
            print("An error occurred during reading the file. Initializing the default method.\n")
            return False

        n.set_weights(wih, who)
        print("Configuration files loaded successfully.\n")
        return True

    return False
