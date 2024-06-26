from mlpiper.components import ConnectableComponent
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeClassifierComp(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):

        X_train = parent_data_objs[0]
        X_test = parent_data_objs[1]
        y_train = parent_data_objs[2]
        y_test = parent_data_objs[3]

        random_state = self._params.get("random_state", 42)
        min_samples_split = self._params.get("min_samples_split", 700)

        classifier = DecisionTreeClassifier(
            random_state=random_state, min_samples_split=min_samples_split
        )
        classifier.fit(X_train, y_train)

        probas = classifier.predict_proba(X_test)[:, -1]

        logloss_score = log_loss(y_test["readmitted"], probas)
        print("Log Loss: {}".format(logloss_score))

        r_a_score = roc_auc_score(y_test["readmitted"], probas)
        print("ROC AUC Score: {}".format(r_a_score))
