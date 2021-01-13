import pickle

training_data_file = "./images/training_data.pkl"
test_data_file = "./images/test_data.pkl"


def save_files(training_data, test_data):
    __load_to_file(training_data, training_data_file)
    __load_to_file(test_data, test_data_file)


def read_files():
    training_data = __read_from_file(training_data_file)
    test_data = __read_from_file(test_data_file)
    return training_data, test_data


def __load_to_file(data, file):
    open_file = open(file, "wb")
    pickle.dump(data, open_file)
    open_file.close()


def __read_from_file(file):
    open_file = open(file, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


class ImageData:

    def __init__(self, first_image, second_image, result):
        self.first_image = first_image
        self.second_image = second_image
        self.result = result

    def get_first_img(self):
        return self.first_image

    def set_first_img(self, first_image):
        self.first_image = first_image

    def get_second_img(self):
        return self.second_image

    def set_second_img(self, second_image):
        self.second_image = second_image

    def get_result(self):
        return self.result

    def set_result(self, result):
        self.result = result
