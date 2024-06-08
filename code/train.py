import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics


def main():
    train_data = pd.read_csv('traindata.csv')
    train_labels = pd.read_csv('trainlabel.txt', header=None)
    test_data = pd.read_csv('testdata.csv')

    train_data['income'] = train_labels

    train_data.replace('?', np.nan, inplace=True)
    test_data.replace('?', np.nan, inplace=True)

    print(train_data.info)
    print(train_data.describe())
    numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                            'native.country']

    # Create preprocessor to fill unknown column values
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    x0 = preprocessor.fit_transform(train_data.drop('income', axis=1))
    y0 = train_data['income']

    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    print("Average Accuracy of Different Model Structures:\n")
    # Random forest classifier
    rfm = RandomForestClassifier(n_estimators=100, random_state=42)
    rfm_ac = cross_val_score(rfm, x0, y0, cv=fold).mean()
    print(f"Random Forest               {rfm_ac:.2f}")

    # Support vector machine
    svm = SVC(kernel='linear')
    svm_ac = cross_val_score(svm, x0, y0, cv=fold).mean()
    print(f"SVM                         {svm_ac:.2f}")

    # K Nearest Neighbor Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn_ac = cross_val_score(knn, x0, y0, cv=fold).mean()
    print(f"KNN                         {knn_ac:.2f}")

    # Logical Regression
    lgr = LogisticRegression()
    lgr_ac = cross_val_score(lgr, x0, y0, cv=fold).mean()
    print(f"LogisticRegression          {lgr_ac:.2f}")

    # AdaBoost with 100
    ada0 = AdaBoostClassifier(n_estimators=100, algorithm='SAMME')
    ada0_ac = cross_val_score(ada0, x0, y0, cv=fold).mean()
    print(f"AdaBoost(100 estimators)    {ada0_ac:.2f}")

    # AdaBoost with 50
    ada1 = AdaBoostClassifier(n_estimators=50, algorithm='SAMME')
    ada1_ac = cross_val_score(ada1, x0, y0, cv=fold).mean()
    print(f"AdaBoost(50 estimators)     {ada1_ac:.2f}")

    best = max({("SVM", svm, svm_ac),
                ("Random Forest", rfm, rfm_ac),
                ("KNN", knn, knn_ac),
                ("Logistic Regression", lgr, lgr_ac),
                ("ADABoost with 100 estimators", ada0, ada0_ac),
                ("AdaBoost with 50 estimators", ada1, ada1_ac)},
               key=lambda t: t[2])
    print("\nThe %s model has the largest accuracy." % best[0])

    # save the test label
    x_train, x_val, y_train, y_val = train_test_split(x0, y0, test_size=0.2, random_state=42)
    model = best[1]
    model.fit(x_train, y_train)
    x_test = preprocessor.transform(test_data)
    y_pred = model.predict(x_test)
    np.savetxt('testlabel.txt', y_pred, fmt='%d')

    y_val_pred = model.predict(x_val)
    ax = plt.subplot()
    confusion_matrix = metrics.confusion_matrix(y_val, y_val_pred)
    sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap="viridis")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix of Chosen Model')
    ax.xaxis.set_ticklabels(['class 0', 'class 1'])
    ax.yaxis.set_ticklabels(['class 0', 'class 1'])
    plt.show()


if __name__ == "__main__":
    main()
