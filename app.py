import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

st.set_page_config(page_title=" ", layout="wide")

col11, col22, col33 = st.columns((3,5,3))
with col11:
    st.write('')
with col22:
    st.image('ds.png')
with col33:
    st.write('')
#for opening the css file
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    
    vAR_selected=st.selectbox("Menu",('Select','Fraud Classification'),key="clear")
    vAR_lib=st.selectbox("",("Libraries","Streamlit","Pandas","Logistic regression"),key="clear1")

def feature_ranks(X,Rank,Support):
    feature_rank=pd.DataFrame()
    for i in range(X.shape[1]):
        new =pd.DataFrame({"Features":X.columns[i],"Rank":Rank[i],'Selected':Support[i]},index=[i])
        feature_rank=pd.concat([feature_rank,new])
    return feature_rank

st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Fraud Detection using Logistic Regression and Random Forest Classifier</p>", unsafe_allow_html=True)
st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
w1,col1,col2,w2=st.columns((0.5,1.5,2.5,0.5))
w11,col11,col22,w22=st.columns((0.5,1.5,2.5,0.5))
with col1:
    st.write("# ")
    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
with col2:
    vAR_problem = st.selectbox("",["Select",'Fraud Detection'])
    if vAR_problem == "Fraud Detection":
        with col1:
            st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
        with col2:
            vAR_Model = st.selectbox("",["Select","Logistic Regression","Random Forest Classifier"])
        if vAR_Model == "Logistic Regression":
            with col1:
                st.write("# ")
                st.write("### ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
            vAR_simple_train_file = st.file_uploader("",type="csv",key='Train')
            with col1:
                st.write("# ")
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Scaling Method</span></p>", unsafe_allow_html=True)
            with col2:
                vAR_features=st.selectbox("",['Standard Scaler','MinMaxScaler'])
            with col1:
                st.write("## ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Feature Selction</span></p>", unsafe_allow_html=True)
            with col2:
                vAR_features_sel=st.selectbox("",['SelectKBest','Recursive Feature elimination'])
            if vAR_simple_train_file is not None:
                with col2:
                    button_placeholder = st.empty()
                    button_clicked = False
                    key=0
                    while not button_clicked:
                        key=key+1
                        button_clicked = button_placeholder.button('Train',key=key)
                        break
                    if button_clicked:
                        button_placeholder.empty()
                        # Load the training dataset
                        df = pd.read_csv(vAR_simple_train_file)
                        df.fillna(method='pad',inplace=True)
                        
                        encoder = {}
                        for i in df.select_dtypes('object').columns:
                            encoder[i] = LabelEncoder()
                            df[i] = encoder[i].fit_transform(df[i])
                        

                        #Model building
                        x = df.iloc[:,:-1].values
                        y = df.iloc[:,-1].values
                        cols=[]
                        for i in df.columns[:-1]:
                            cols.append(i)
                        
                        if vAR_features=='MinMaxScaler': 
                            scaler = MinMaxScaler()
                            x=pd.DataFrame(scaler.fit_transform(x),columns=cols)
                            st.success('scaled')
                        else:
                            scaler = StandardScaler()
                            x=pd.DataFrame(scaler.fit_transform(x),columns=cols)
                            st.success('scaled  ')
                        
                        if vAR_features_sel=='SelectKBest':
                            best_fea = SelectKBest(chi2,k=8)
                            kbest = best_fea.fit_transform(x,y)
                            best = list(best_fea.get_support(indices=True))
                            x=pd.DataFrame(df,columns=(df.columns)[best])
                            
                        
                        else:
                            model = LogisticRegression(solver='liblinear')
                            rfe = RFE(model,n_features_to_select=4)
                            fit = rfe.fit(x,y)
                            feature_rank_df = feature_ranks(x,fit.ranking_,fit.support_)
                            recursive_feature_names=feature_rank_df[feature_rank_df['Selected']==True]
                            RFE_selected_features = x[recursive_feature_names['Features']]
                            RFE_selected_features = x

                        # # splitting X and y into training and testing sets
                        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
                        # Create a Decision Tree Classifier object
                        if vAR_Model=='Logistic Regression':
                            clf = LogisticRegression()  
                        else:
                            clf = RandomForestClassifier()
                        # Create and train the model
                        clf.fit(x, y)
                        y_pred=clf.predict(x_test)
                        st.success(accuracy_score(y_test, y_pred))
                        st.success("Model trained successfully")
            with col11:
                st.write("# ")
                st.write("### ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
                with col22:
                    vAR_simple_test_file = st.file_uploader("",type="csv",key='Test') 
                    if  vAR_simple_test_file is not None:
                        with col22:
                            button_placeholder = st.empty()
                            button_clicked = False
                            key=2
                            while not button_clicked:
                                key=key+2
                                button_clicked = button_placeholder.button('Test',key=key)
                                break
                            if button_clicked:
                                button_placeholder.empty()
                                # Load the testing dataset
                                df_test = pd.read_csv(vAR_simple_test_file)
                                df = pd.read_csv(vAR_simple_train_file)
                                df.fillna(method='pad',inplace=True)
                        
                                encoder = {}
                                for i in df.select_dtypes('object').columns:
                                    encoder[i] = LabelEncoder()
                                    df[i] = encoder[i].fit_transform(df[i])

                                #Model building
                                x = df.iloc[:,:-1].values
                                y = df.iloc[:,-1].values
                                cols=[]
                                for i in df.columns[:-1]:
                                    cols.append(i)
                                
                                if vAR_features=='MinMaxScaler': 
                                    scaler = MinMaxScaler()
                                    x=pd.DataFrame(scaler.fit_transform(x),columns=cols)
                                    st.success('scaled')
                                else:
                                    scaler = StandardScaler()
                                    x=pd.DataFrame(scaler.fit_transform(x),columns=cols)
                                    st.success('scaled  ')
                                
                                if vAR_features_sel=='SelectKBest':
                                    best_fea = SelectKBest(chi2,k=8)
                                    kbest = best_fea.fit_transform(x,y)
                                    best = list(best_fea.get_support(indices=True))
                                    x=pd.DataFrame(df,columns=(df.columns)[best])
                                    
                                
                                else:
                                    model = LogisticRegression(solver='liblinear')
                                    rfe = RFE(model,n_features_to_select=4)
                                    fit = rfe.fit(x,y)
                                    feature_rank_df = feature_ranks(x,fit.ranking_,fit.support_)
                                    recursive_feature_names=feature_rank_df[feature_rank_df['Selected']==True]
                                    RFE_selected_features = x[recursive_feature_names['Features']]
                                    RFE_selected_features = x
                                # # splitting X and y into training and testing sets
                                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

                                
                                
                                if vAR_Model=='Logistic Regression':
                                    clf = LogisticRegression()  
                                else:
                                    clf = RandomForestClassifier()
                                # Create and train the model
                                clf.fit(x, y)

                                
                                df_test.fillna(method='pad',inplace=True)
                                encoder = {}
                                for i in df_test.select_dtypes('object').columns:
                                    encoder[i] = LabelEncoder()
                                    df_test[i] = encoder[i].fit_transform(df_test[i])
                                
                                cols=[]
                                for i in df_test.columns:
                                    cols.append(i)
                                
                                
                                if vAR_features=='MinMaxScaler': 
                                    scaler = MinMaxScaler()
                                    x=pd.DataFrame(scaler.fit_transform(df_test),columns=cols)
                                    st.success('scaled')
                                else:
                                    scaler = StandardScaler()
                                    x=pd.DataFrame(scaler.fit_transform(df_test),columns=cols)
                                    st.success('scaled  ')
                                

                                lis=[]
                                for i in x.columns:
                                    for j in df_test.columns:
                                        if i==j:
                                            lis.append(i)

                                if vAR_features_sel=='SelectKBest':
                                    df_test=df_test[lis]
                                
                                else:
                                    df_test=df_test[lis]


                                y=[]
                                for i,j in df_test.iterrows():
                                    x=list(j.values)
                                    y.append(x)
                                resfin = []
                                for i in range(0,len(y)):
                                    result = clf.predict([y[i]])
                                    newres = result[0]
                                    resfin.append(newres)


                                df_test['res'] = resfin
                                st.table(df_test)
                                


       