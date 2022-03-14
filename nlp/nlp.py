import os
import json
from datetime import datetime
from threading import Thread, current_thread, get_ident
from typing import Dict, List, Union
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import glob
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from keras.layers import  Dropout, Dense
from keras.models import Sequential
import numpy as np
from sklearn import metrics

class Trainer():

    def __init__(self) -> None:
        # first we get the current location of file by (os.path.abspath(__file__))
        # then we get parent of that location (director) by (os.path.dirname)
        # then we join the path of director named 'storage' under the project
        self.__storage_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'storage')
        # we check here if this director is exists or not
        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)
        # here we create status_path to save status of model
        self.__status_path = os.path.join(
            self.__storage_path, 'model_status.json')
        # here we create model_path to save path of model as a pickle file
        self.__model_path = os.path.join(
            self.__storage_path, 'model_pickle.joblib')
        # here we need to load status of model from saved json file if exists else we set model_status as below.
        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)

        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(), "classes": [], "evaluation": {}}

        # Joblib is a set of tools to provide lightweight pipelining in Python and
        # it can be used for loading trained model
        # here we check if trained model exists or not
        if os.path.exists(self.__model_path):
            self.model = joblib.load(self.__model_path)
        else:
            self.model = None
        #  _running_threads to inform that if there are any training process or not
        self._running_threads = []
        self._pipeline = None
        
    def merge(self):
        # merging the files
        joined_files = os.path.join("data/processed", "processed*.csv")

        # A list of all joined files is returned
        joined_list = glob.glob(joined_files)
        li = []
        # Finally, the files are joined
        # df = pd.concat(map(pd.read_csv(lineterminator='\n'), joined_list), ignore_index=True)
        for filename in joined_list:
            df = pd.read_csv(filename,lineterminator='\n', index_col=None, header=0,encoding="utf-8")
            li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)
        print(df.head())
        return df


    def _train_job_machine(self, x_train: List[str], x_test: List[str], y_train: List[Union[str, int]],
                   y_test: List[Union[str, int]]):
        self._pipeline.fit(x_train, y_train)
        report = classification_report(
            y_test, self._pipeline.predict(x_test), output_dict=True)
        # _pipeline.classes_ returns numpy array so we need to cast it to list to be serializable
        classes = self._pipeline.classes_.tolist()
        self._update_status("Model Ready", classes, report)
        # here we saved pipeline in __model_path as a pickle file compressed at maximum (0-9)
        joblib.dump(self._pipeline, self.__model_path, compress=9)
        self.model = self._pipeline
        self._pipeline = None
        # here we get id of running thread
        # get_ident() is an inbuilt method of the threading module in Python.It is used to
        # return the "thread identifier" of the current thread.Thread identifiers can be recycled when
        # a thread exits and another thread is created.The value has no direct meaning.
        thread_id = get_ident()
        # here we remove this id from _running_threads
        for i, t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break
    '''It is important to note that Python won't raise a TypeError if you pass a float into index,
     the reason for this is one of the main points in Python's design philosophy: 
     "We're all consenting adults here", which means you are expected to be aware of what you can 
     pass to a function and what you can't. If you really want to write code that throws TypeErrors 
     you can use the isinstance function to check that the passed argument is of the proper type or
      a subclass of it '''
    def _train_job_deep(self, x_train: List[str], x_test: List[str], y_train: List[Union[str, int]],
                   y_test: List[Union[str, int]]):

        self._pipeline.fit(x_train, y_train,validation_data=(x_test, y_test),
                      epochs=10,
                      batch_size=128,
                      verbose=2)
        report = classification_report(
            y_test, self._pipeline.predict(x_test), output_dict=True)
        # _pipeline.classes_ returns numpy array so we need to cast it to list to be serializable
        classes = self._pipeline.classes_.tolist()
        self._update_status("Model Ready", classes, report)
        # here we saved pipeline in __model_path as a pickle file compressed at maximum (0-9)
        joblib.dump(self._pipeline, self.__model_path, compress=9)
        self.model = self._pipeline
        self._pipeline = None
        # here we get id of running thread
        # get_ident() is an inbuilt method of the threading module in Python.It is used to
        # return the "thread identifier" of the current thread.Thread identifiers can be recycled when
        # a thread exits and another thread is created.The value has no direct meaning.
        thread_id = get_ident()
        # here we remove this id from _running_threads
        for i, t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break

    # Any - when it could be anything.
    # Union - when it could be anything within a specified set of types, as opposed to Any.

    def trainMachineLearning(self, texts: List[str], labels: List[Union[str, int]]) -> None:
        # check first if there are any training process or not
        if len(self._running_threads):
            raise Exception("A training process is already running.")

        x_train, x_test, y_train, y_test = train_test_split(texts, labels,stratify=labels)
        # clf = LogisticRegression()
        # vec = TfidfVectorizer(stop_words='english',
        #                       min_df=.01, max_df=.35, ngram_range=(1, 2))
        self._pipeline = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', LinearSVC()),
                             ])
        # self._pipeline = make_pipeline(vec, clf)


        # update model status
        self.model = None
        self._update_status("Training")
        # we need thread to not stop the server.
        t = Thread(target=self._train_job_machine, args=(
            x_train, x_test, y_train, y_test))
        self._running_threads.append(t)
        t.start()

    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)

    def Build_Model_DNN_Text(shape, nClasses=18, dropout=0.5):

        model = Sequential()
        node = 512  # number of nodes
        nLayers = 4  # number of  hidden layer
        model.add(Dense(node, input_dim=shape, activation='relu'))
        model.add(Dropout(dropout))
        for i in range(0, nLayers):
            model.add(Dense(node, input_dim=node, activation='relu'))
            model.add(Dropout(dropout))
        model.add(Dense(nClasses, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        return model

    def trainDeepLearning(self, texts: List[str], labels: List[Union[str, int]]) -> None:
        # check first if there are any training process or not
        if len(self._running_threads):
            raise Exception("A training process is already running.")

        x_train, x_test, y_train, y_test = train_test_split(texts, labels,stratify=labels)
        # clf = LogisticRegression()
        # vec = TfidfVectorizer(stop_words='english',
        #                       min_df=.01, max_df=.35, ngram_range=(1, 2))
        X_train_tfidf, X_test_tfidf = self.TFIDF(x_train, x_test)
        clf = KerasClassifier(build_fn=self.Build_Model_DNN_Text, verbose=0)
        # model_DNN = self.Build_Model_DNN_Text(X_train_tfidf.shape[1])

        self._pipeline = Pipeline([('clf',clf)])

        # update model status
        self.model = None
        self._update_status("Training")
        # we need thread to not stop the server.
        t = Thread(target=self._train_job, args=(
            X_train_tfidf, X_test_tfidf, y_train, y_test))
        self._running_threads.append(t)
        t.start()

    def predict(self, texts: List[str]) -> List[Dict]:
        response = []
        if self.model:
            probs = self.model.predict_proba(texts)
            for i, row in enumerate(probs):
                row_pred = {}
                row_pred['text'] = texts[i]
                row_pred['predictions'] = {class_: round(float(prob), 3) for class_, prob in zip(
                    self.model_status['classes'], row)}
                response.append(row_pred)
        else:
            raise Exception("No Trained model was found.")
        return response

    def get_status(self) -> Dict:
        return self.model_status

    def _update_status(self, status: str, classes: List[str] = [], evaluation: Dict = {}) -> None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)
