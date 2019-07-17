import cPickle as pickle
import gzip
import json
with gzip.open('mnist.pkl.gz') as f:
    training_data, validation_data, test_data = pickle.load(f)

training_data = (training_data[0].tolist(), training_data[1].tolist())
validation_data = (validation_data[0].tolist(), validation_data[1].tolist())
test_data = (test_data[0].tolist(), test_data[1].tolist())

with open('training_data.json', 'w') as f:
    print(json.dump(training_data, f))

with open('validation_data.json', 'w') as f:
    print(json.dump(validation_data, f))

with open('test_data.json', 'w') as f:
    print(json.dump(test_data, f))
