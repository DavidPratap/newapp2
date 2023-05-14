import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('fivethirtyeight')

def main():
    st.title("The Medical Diagnoistic App")
    st.write("does the woman have diabetes ?")
    st.sidebar.title("medical App")

    @st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv('data.csv')
        # remove the redundant columns
        data.drop('Unnamed: 0', axis=1, inplace=True)
        zerofill=lambda x:x.replace(0, x.median())
        cols=data.columns[1:6]
        data[cols]=data[cols].apply(zerofill, axis=0)
        d={'Yes':1, 'No':0}
        df=data.copy()
        df['Outcome']=df['Outcome'].map(d)
        return df
    
    df=load_data()
    if st.checkbox('Show data', False):
        st.write(df)

    ### Preprocessing for Modelling
    def preprocess(data, label):
        X=data.drop(label,axis=1)
        y=df[label]
        sm=SMOTE()
        X,y=sm.fit_resample(X,y)
        x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test=preprocess(df, 'Outcome')

    classifier=st.sidebar.selectbox('Choose classifier', ('KNN', 'SVM', 'RFC'))
    plots=st.sidebar.multiselect('Choose plot', ('Confusion Matrix', 'ROC Curve'))

    def plot_metrics(clf, x_test, y_test):
        if 'Confusion Matrix' in plots:
            plot_confusion_matrix(clf, x_test, y_test, display_labels=[0,1])
            st.pyplot()

        if 'ROC Curve' in   plots:
            plot_roc_curve(clf, x_test, y_test, )
            st.pyplot()
    
    if classifier=='RFC':
        n_estimators=st.sidebar.number_input('no.of trees', 100, 500, step=50, key='trees')
        criterion=st.sidebar.radio('criterion',('gini', 'entropy'), key='est')
        max_depth=st.sidebar.number_input('maxd', 1, 20, step=1, key='maxD')
        if st.sidebar.button('Predict'):
            clf=RandomForestClassifier()
            clf.fit(x_train, y_train)
            y_pred=clf.predict(x_test)
            st.write("Accuracy =", accuracy_score(y_test, y_pred))
            st.write("Precision =", precision_score(y_test, y_pred))
            st.write("Recall =", recall_score(y_test, y_pred))
            st.write("F1 =", f1_score(y_test, y_pred))
            plot_metrics(clf, x_test, y_test)


if __name__=='__main__':
    main()
