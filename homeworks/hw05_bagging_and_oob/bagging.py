import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = data.shape[0]
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.randint(0, data_length, data_length))
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = [], []
            for i in self.indices_list[bag]:
                data_bag.append(data[i])
                target_bag.append(target[i])
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = []
        for object in data:
            object_predictions = [model.predict(object.reshape(1, -1)) for model in self.models_list]
            predictions.append(np.mean(object_predictions))
        return predictions
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = []
        for i_object in range(self.num_bags):
            models = []
            for i in range(self.num_bags):
                if i_object not in self.indices_list[i]:
                    models.append(self.models_list[i])
            list_of_predictions_lists.append([model.predict(self.data[i_object].reshape(1, -1)) for model in models])
        # Your Code Here

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        def get_elem(list):
            if (len(list)):
                return np.mean(list)
            return None
        self.oob_predictions = [get_elem(predicts) for predicts in self.list_of_predictions_lists]
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        error_list = []
        for prediction, target in zip(self.oob_predictions, self.target):
            if prediction is not None:
                error = (target - prediction) ** 2
                error_list.append(error)
        mean_squared_error = np.mean(error_list)
        return mean_squared_error