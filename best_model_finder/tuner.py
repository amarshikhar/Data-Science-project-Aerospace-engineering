from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Model_Finder:
    """
                This class shall  be used to find the model with best RMSE and R2 score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rf_model = RandomForestRegressor()
        self.xgb_model = XGBRegressor()

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130],
                               "max_depth": [6,8,10,12], "min_samples_split" : [60,70,80,100], "min_samples_leaf" : [30,40,50,60]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rf_model, param_grid=self.param_grid, cv=5,  verbose=3,n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                              max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            # training the mew model
            self.rf_model.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            
            return self.rf_model
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 8, 10, 12],
                'n_estimators': [10, 50, 80, 100, 120]
                "min_samples_split" : [50,70,100,150]
                "min_samples_leaf" : [20,30,40,50]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.xgb_model,self.param_grid_xgboost, verbose=3,cv=5,n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']



            # creating a new model with the best parameters
            self.xgb_model = XGBRegressor(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            # training the mew model
            self.xgb_model.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb_model
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best R2 score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict_proba(test_x) # Predictions using the XGBoost Model

            self.xgboost_score1 = np.sqrt(mean_squared_error(test_y, self.prediction_xgboost1))
            self.logger_object.log(self.file_object, 'RMSE for XGBoost:' + str(self.xgboost_score1)) 

            self.xgboost_score2 = r2_score(test_y, self.prediction_xgboost) 
            self.logger_object.log(self.file_object, 'R2 score for XGBoost:' + str(self.xgboost_score2))

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.=self.random_forest.predict_proba(test_x) # prediction using the Random Forest Algorithm

            self.random_forest_score1 = np.sqrt(mean_squared_error(test_y,self.prediction_random_forest))
            self.logger_object.log(self.file_object, 'RMSE for RF:' + str(self.random_forest_score1))

            self.random_forest_score2 = r2_score(test_y, self.prediction_random_forest)
            self.logger_object.log(self.file_object, 'R2 score for RF:' + str(self.random_forest_score2))

            #comparing the two models
            if(self.random_forest_score2 < self.xgboost_score2):
                return 'XGBoost',self.xgboost
            else:
                return 'RandomForest',self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

