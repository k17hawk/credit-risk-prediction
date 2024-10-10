from cmath import log
import importlib
from pyexpat import model
import numpy as np
import yaml
from scikit_credit_risk import logging
import os
import sys
from collections import namedtuple
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from scikit_credit_risk.entity.config import MetricInfoArtifact,InitializeModelDetails,RandomSearchBestModel,BestModel
from scikit_credit_risk.entity.constant import *

def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6)->MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model
    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    return
    It retured a frozen class 
    MetricInfoArtifact
    """
    try:
        
    
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            print(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")

            y_train_clean = y_train[np.logical_not(np.isnan(y_train))]
            y_test_clean = y_test[np.logical_not(np.isnan(y_test))]
           
            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            print(y_train_clean.shape,"y_train-clean pred shape")
            print(y_test_clean.shape,'y-test-clean pred shape')

            print(y_train_pred.shape,"y_train pred shape")
            print(y_test_pred.shape,'y-test pred shape')
            # print("y_train shape---------")
            # print(y_train_clean.shape)
            # print("the shape of train data after predicting---------- ")
            # print(y_train_pred.shape)


            # # print("y_test shape---------")
            # # print(y_test_clean.shape)
            # print("the shape of test data after predicting---------- ")
            # print(y_test_pred.shape)
            
            
            

            test_acc = accuracy_score(y_test_clean, y_test_pred)
            train_acc = accuracy_score(y_train_clean,y_train_pred)

            print(train_acc,"training accuracy")
            



            #Calculating haarrmonic mean for train data
            # precision_train = precision_score(y_train, y_train_pred)
            # recall_train = recall_score(y_train, y_train_pred)
            # f1_train = f1_score(y_train, y_train_pred)

            #ncludes all the entries in a series using harmonic mean
            # Calculating harmonic mean of train_accuracy and test_accuracy in order to test with base accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            

            #Calculating haarrmonic mean for test data
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)

            
            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
        
            
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.30:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        test_precision=test_precision,
                                                        test_recall = test_recall,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise e

class ModelFinder:
    def __init__(self, model_config_path: str = None,):
        try:
            #read the model.yaml 
            self.config: dict = ModelFinder.read_params(model_config_path)
            
            #for import RandomSearchCV module info random_search
            self.random_search_cv_module: str = self.config[SEARCH_KEY][MODULE_KEY]
            
            #for import RandomSearchCV params class 
            self.random_search_class_name: str = self.config[SEARCH_KEY][CLASS_KEY]
            
            #for import RandomSearchCV params
            self.random_search_property_data: dict = dict(self.config[SEARCH_KEY][PARAM_KEY])

            #get the key from model_selection_key from model.yaml in dict format
            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.random_searched_best_model_list = None

        except Exception as e:
            raise  e
        
    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise e
    
    @staticmethod
    def update_property_of_class(instance_ref:object, property_data: dict):
        """
        this function is created to assign the property to the model object

        """
        try:
            #check first if property_data is type dict  {'fit_intercept' :'True'}
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            #iterate over property_data to get fit_intercept and True
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                #for setting dynamically model_object.fit_intercept  = True we use setattr

                '''
                clas A:
                  pass
                A.a = 2

                print(A.a)

                setattr(A,'b',5)
                print(A.b)  
                '''
                setattr(instance_ref, key, value)
        
            return instance_ref
        except Exception as e:
            raise  e
    
    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        """
        this function is created to import the module 
        """
        try:
            # load the module, will raise ImportError if module cannot be loaded
            
            #importlib is dynamic  module importer
            module = importlib.import_module(module_name)

            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            
            #here getattr will give refrence to  module and class_name
            #module is sklean.linear_module  class_name = LogisticRegression
            class_ref = getattr(module, class_name)

            return class_ref
        except Exception as e:
            raise  e
    def execute_random_search_operation(self, initialized_model: InitializeModelDetails, input_feature,
                                      output_feature) -> RandomSearchBestModel:
        """
        excute_random_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_random: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return RandomSearchOperation object
        """
        try:
            if initialized_model.param_random_search is None:
                logging.info(f"Random search parameters not provided for {initialized_model.model}. Skipping random search.")

                initialized_model.model.fit(input_feature, output_feature)
                if hasattr(initialized_model.model, 'score'):
                    best_score = initialized_model.model.score(input_feature, output_feature)
                else:
                    best_score = None 
                
                # Directly return the model without random search
                return RandomSearchBestModel(
                    model_serial_number=initialized_model.model_serial_number,
                    model=initialized_model.model,
                    best_model=initialized_model.model,
                    best_parameters=None,  # No best params since no random search was done
                    best_score=best_score  # No best score since no random search was done
                )
        
            
           # initialized_model is the model that we have initalize
           # importing RandomSearchCV class and instantiating it
            random_search_cv_ref = ModelFinder.class_for_name(module_name=self.random_search_cv_module,
                                                             class_name=self.random_search_class_name
                                                             )

            random_search_cv = random_search_cv_ref(estimator=initialized_model.model,
                                                param_distributions=initialized_model.param_random_search)

            #add the property to random_search_cv using setattr
            random_search_cv = ModelFinder.update_property_of_class(random_search_cv,
                                                                   self.random_search_property_data)

            
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)

            random_search_cv.fit(input_feature, output_feature)

            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            
            #returning to namedtuple
            random_searched_best_model = RandomSearchBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=random_search_cv.best_estimator_,
                                                             best_parameters=random_search_cv.best_params_,
                                                             best_score=random_search_cv.best_score_
                                                             )
            
            return random_searched_best_model
        except Exception as e:
            raise e
    
    def get_initialized_model_list(self) -> List[InitializeModelDetails]:
        """
        This function will return a list of model details.
        return List[ModelDetail]
        i.e  initialized models class
        model_serial_number", "model", "param_random_search", "model_name"
        """
        try:
            #initialize model list as empty list
            initialized_model_list = []
            #get the model_0 and model_1 info  from models_initialization_config.keys() 
            #and call them model serial number
            """
            module_0:
             class: LinearRegression
             module: sklearn.linear_model
            """
            for model_serial_number in self.models_initialization_config.keys():
                #loop to get each module info
                model_initialization_config = self.models_initialization_config[model_serial_number]
                """
                    class: LinearRegression
                    module: sklearn.linear_model
              
                """
                #now since we have model info ,we have to import the library name.
                #class_for_name will import module dynamically 
                model_obj_ref = ModelFinder.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY]
                                                            )
                #here we created the object of the module
                model = model_obj_ref()
                """
                params:
                 fit_intercept: true
                 search_param_random:
                 fit_intercept:
                  - true
                  - false
                """
                #use constant PARAM_KEY to import the params keys but first we check if it is present
                if PARAM_KEY in model_initialization_config:

                    #create dict from dict keys 
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    print("model object property data",model_obj_property_data)

                    #fit_intercept = True  should be done in logistic regression to set it use update_property_of_class
                    model = ModelFinder.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)

 
                # param_random_search = model_initialization_config[SEARCH_PARAM_RANDOM_KEY]
                
                param_random_search = model_initialization_config.get(SEARCH_PARAM_RANDOM_KEY, None)
                print("param_random_search",param_random_search)



                #Access the model name using class:LogisticRegression and  module: sklearn.linear_model
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                #assign it to the namedtuple
                model_initialization_config = InitializeModelDetails(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_random_search=param_random_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise  e
    
    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializeModelDetails,
                                                             input_feature,
                                                             output_feature) -> RandomSearchBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_random: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a RandomSearchOperation
        """
        try:
            #execute_random_search_operation
            return self.execute_random_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise  e
    
    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializeModelDetails],
                                                              input_feature,
                                                              output_feature) -> List[RandomSearchBestModel]:
        
        """
        this function is written to check  which is the best paramter after 
        search_param_random:
            fit_intercept:
                - true
                - false

        """

        try:
            self.random_searched_best_model_list = []

            # initialized_model_list returns the list of model
            for initialized_model_list in initialized_model_list:
                
                #since we are searching best  params for best models now we will look for best model
                random_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )

                self.random_searched_best_model_list.append(random_searched_best_model)
            return self.random_searched_best_model_list
        except Exception as e:
            raise  e
    
    @staticmethod
    def get_model_detail(model_details: List[InitializeModelDetails],
                         model_serial_number: str) -> InitializeModelDetails:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise  e
    
    @staticmethod
    def get_best_model_from_random_searched_best_model_list(random_searched_best_model_list: List[RandomSearchBestModel],
                                                          base_accuracy=0.6
                                                          ) -> BestModel:
        try:
            best_model = None
            for random_searched_best_model in random_searched_best_model_list:
                if base_accuracy < random_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{random_searched_best_model}")
                    base_accuracy = random_searched_best_model.best_score

                    best_model = random_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise  e

    def get_best_model(self, X, y,base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            #get the list of self initalized model
            initialized_model_list = self.get_initialized_model_list()
            print("loading initialized model",initialized_model_list)
            logging.info(f"Initialized model: {initialized_model_list}")

            #get the best model list applying the random search best model list which will take model list target feature and independent feature
            random_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            #pass these params to get best model from the list of model
            return ModelFinder.get_best_model_from_random_searched_best_model_list(random_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise e
        
    


