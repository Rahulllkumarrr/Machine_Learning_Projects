import pandas as pd
import warnings
import timeit
from sklearn.exceptions import ConvergenceWarning
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn import preprocessing
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, \
    recall_score, classification_report,accuracy_score, \
    f1_score, roc_auc_score, mean_absolute_error,mean_squared_error

warnings.simplefilter('ignore')
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

RANDOM_STATE=4

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


class FBScore:
    DATASET = "NumberBook.csv"
    TRAIN_SIZE = 0.80
    RANDOM_STATE = 42

    def __init__(self):
        self.__data = pd.read_csv("NumberBook.csv",sep = ",", encoding = "ISO-8859-1", low_memory = False)
        self.__rows = self.__data.shape[0]
        self.__cols = self.__data.shape[1]
        self.__fited = False

        print(' * Reading the dataset from the "{}" file...'.format(FBScore.DATASET))
        self.__data = self.__data.dropna()
        print(" * {} rows and {} columns were read.".format(self.__data.shape[0], self.__data.shape[1]))
        self.__data["Grade"] = self.__data["FeedbackScore"] // 20



        self.__features = ["StepID", "TreatmentID"]
        #self.__predict = ["FeedbackScore"]
        self.__predict = ["Grade"]

    @property
    def rows(self):
        return self.__rows

    @property
    def cols(self):
        return self.__cols

    def fit(self):
        X = self.__data[self.__features]
        y = self.__data[self.__predict]

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y,
                                                                                        train_size=FBScore.TRAIN_SIZE,
                                                                                        test_size=1 - FBScore.TRAIN_SIZE,
                                                                                        random_state=FBScore.RANDOM_STATE)

        self.__model = make_pipeline(StackingEstimator(estimator=LogisticRegression(C=20.0, dual=True, penalty="l2")),
                                     LogisticRegression())
        self.__model.fit(self.__X_train, self.__y_train.values.ravel())
        self.__fited = True

    @property
    def score(self):
        if self.__fited:
            return self.__model.score(self.__X_test, self.__y_test)
        else:
            return None

    @property
    def confusion_matrix(self):
        y_train_pred = self.__model.predict(self.__X_train)
        return confusion_matrix(self.__y_train, y_train_pred)

    def predict(self, X):
        if self.__fited:
            Grade = self.__model.predict(X)
            if Grade < 20:
                return 0
            elif Grade < 40:
                return 1
            elif Grade < 60:
                return 2
            elif Grade < 80:
                return 3
            else:
                return 4
        else:
            return None

    def evaluate(self, Name):
        with open("{}Evaluate.txt".format(Name), "w") as text_file:
            print("\n  Model Accuracy:", self.__model.score(self.__X_test, self.__y_test), file=text_file)
            predicted = self.__model.predict(self.__X_test)
            print("\n Number of correct classifications:",
                  accuracy_score(self.__y_test.values.ravel(), predicted, normalize=False), file=text_file)
            print(" Mean Squared Error (MSE):", mean_squared_error(self.__y_test.values.ravel(), predicted),
                  file=text_file)
            print(" Mean Absolute Error (MAE):", mean_absolute_error(self.__y_test.values.ravel(), predicted),
                  file=text_file)
            print(" Area Under the Curve (AUC) Score:",
                  multiclass_roc_auc_score(self.__y_test.values.ravel(), predicted), file=text_file)
            print(" Root Mean Square Error (RMSE):", sqrt(mean_squared_error(self.__y_test.values.ravel(), predicted)),
                  file=text_file)
            matrix = confusion_matrix(self.__y_test.values.ravel(), predicted)
            print("\n Confusion Matrix", file=text_file)
            print(matrix, file=text_file)
            print("\n", file=text_file)
            report = classification_report(self.__y_test.values.ravel(), predicted)
            print(report, file=text_file)

    def custom_cv(self, Name):
        with open("{}CrossValidate.txt".format(Name), "w") as text_file:
            scoring = {'acc_val': make_scorer(accuracy_score), 'MSE': 'neg_mean_squared_error',
                       'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_mean_squared_log_error',
                       'AUC': make_scorer(multiclass_roc_auc_score),
                       'prec': 'precision_macro', 'rec_macro': 'recall_macro', 'f1_score': 'f1_macro',
                       'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}


            scores = cross_validate(self.__model, self.__X_train, self.__y_train.values.ravel(), scoring=scoring, cv=5,
                                    return_train_score=False)


            print(" Model Accuracy", file=text_file)
            print(scores['test_acc_val'], file=text_file)
            print(" Average Model Accuracy: %.2f%% (%.2f%%)" % (
                scores['test_acc_val'].mean() * 100, scores['test_acc_val'].std() * 100), file=text_file)
            print(" Mean Squared Error (MSE):", file=text_file)
            print(scores['test_MSE'], file=text_file)
            print(" Mean Absolute Error (MAE):", file=text_file)
            print(scores['test_MAE'], file=text_file)
            print("\n AUC:", file=text_file)
            print(scores['test_AUC'], file=text_file)
            print(" Root Mean Squared  Error (RMSE):", file=text_file)
            print(scores['test_RMSE'], file=text_file)
            print("\n Precision:", file=text_file)
            print(scores['test_prec'], file=text_file)
            print(" Recall:", file=text_file)
            print(scores['test_rec_macro'], file=text_file)
            print(" F1 Score:", file=text_file)
            print(scores['test_f1_score'], file=text_file)
            print("\n True Positives:", file=text_file)
            print(scores['test_tp'], file=text_file)
            print(" False Positives:", file=text_file)
            print(scores['test_fp'], file=text_file)
            print(" False Negatives:", file=text_file)
            print(scores['test_fn'], file=text_file)
            print(" True Negatives:", file=text_file)
            print(scores['test_tn'], file=text_file)


if __name__ == '__main__':
    # Testing
    start_time = timeit.default_timer()
    model = FBScore()
    model.fit()
    print("\n Model Accuracy:", model.score)
    print(" Execution time: {:.2f} secs.".format(timeit.default_timer() - start_time))


    # print(model.predict(X))

    model.evaluate('Logistic')
    print("\n Evaluation for 'Feedback' Model Complete")

    model.custom_cv('Logistic')
    print("\n Cross Validation for 'Feedback' Model Complete")




