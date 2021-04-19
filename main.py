# packages needed
import pandas
import numpy
import sklearn.tree

# release data
train_data = pandas.read_csv('train.csv')
test_data = pandas.read_csv('test.csv')

# point target and key features
target = 'label'
features = train_data.columns[1:]

# get columns
y = train_data['label']
X = train_data[features]

# create model
model = sklearn.tree.DecisionTreeRegressor()
model.fit(X, y)
predictions = model.predict(test_data)
predictions = predictions = numpy.around(predictions).astype(int)

# output
output = pandas.DataFrame({'ImageId': numpy.arange(1, 28001), 'Label': predictions})
output.set_index('ImageId', inplace=True)
output.to_csv('predictions.csv')