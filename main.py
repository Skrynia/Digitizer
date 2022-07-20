import pandas
import numpy
import sklearn.tree


def main():
    train_data = pandas.read_csv('train.csv')
    test_data = pandas.read_csv('test.csv')

    target = train_data['label']
    args = train_data[train_data.columns[1:]]

    model = sklearn.tree.DecisionTreeClassifier(random_state=0)
    model.fit(args, target)
    predictions = model.predict(test_data)

    output = pandas.DataFrame({'ImageId': numpy.arange(1, 28001),
                               'Label': predictions})

    output.set_index('ImageId', inplace=True)
    output.to_csv('predictions.csv')


if __name__ == '__main__':
    main()
