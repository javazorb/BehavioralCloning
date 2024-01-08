import dataset.generate_environment as generate_data
import dataset.dataset


def run():
    generate_data.generate_and_save_environments(num_environments=1000)
    training_data, validation_data, testing_data = dataset.dataset.train_test_val_split(generate_data.load_environments())


if __name__ == '__main__':
    run()
