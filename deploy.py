
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.cluster import KMeans
from PIL import Image
from create_ann_model import create_ann_model


## set page configuration
st.set_page_config(page_title='NBA Potentials Analysis in Early Career', layout='centered')

with open(r'd:\Users\V\Desktop\Bootcamp\NBA\cluster.pkl', 'rb') as file:
    model = pickle.load(file)
##Load two data set
df=pd.read_csv('earlyCareer.csv')


## add page title and content
st.title('NBA Players Analysis in Early Career')
col1, col2, col3=st.columns(3)
st.write("This application can allow you to input a player's weight, height, and stats in their early career and predict their tactic and commercial value in the future.")
# Open the image file
image = Image.open('NBA Logo Vector.png')

# Display the image
st.image(image, caption='NBA analysis')


##numeric_var=['child_mort', 'exports', 'health', 'imports', 'income','inflation', 'life_expec', 'total_fer', 'gdpp']
def info_callback(d1,d2,d3,d4,d5):    
    with open('dt.csv', 'w+') as f:    
        myfile=csv.writer(f)
        myfile.writerow(['player_height', 'player_weight', 'pts', 'reb', 'ast'])
        myfile.writerow([player_height, player_weight, pts, reb, ast])


with st.form(key="my_form",clear_on_submit=True):
    
    st.write("Enter the required data:")
        
    player_height=st.number_input(label="Please enter the player's height", step=1.,format="%.2f" )
    player_weight=st.number_input(label="Please enter the player's  weight", step=1.,format="%.2f" )
    pts=st.number_input(label="Please enter the player's average points per game", step=1.,format="%.2f" )
    reb=st.number_input(label="Please enter the player's average assists per game", step=1.,format="%.2f" )
    ast=st.number_input(label="Please enter the player's average rebounds per game", step=1.,format="%.2f" )
     
    if st.form_submit_button('submit'):
        info_callback(player_height, player_weight, pts, reb, ast)
  

row=pd.read_csv("dt.csv")
a=len(df)
df_cluster=df.append(row, ignore_index=True)
st.info("#### Show your player's data ")
sample_test=st.dataframe(pd.read_csv("dt.csv"))

sample=df.loc[:,['player_height', 'player_weight', 'pts', 'reb', 'ast']]

## make the prediction
if st.button("Predict the player's possition"):
    prediction = model.predict(sample)
    clusters = pd.DataFrame({'cluster': np.reshape(prediction, (prediction.shape[0],))})
    if int(clusters['cluster'].iloc[-1])==0:
        st.write('This player is suggested to be Foward in a team')
    if int(clusters['cluster'].iloc[-1])==1:
        st.write('This player is suggested to be Guard in a team')
    if int(clusters['cluster'].iloc[-1])==2:
        st.write('This player is suggested to be Centre in a team')
   





## Classification
##load ann model
with open(r'D:\Users\V\Desktop\Bootcamp\NBA\classification.pkl', 'rb') as file:
    classification_model = pickle.load(file)







# print the prediction result
if st.button("Predict the player's Potential"):
    result = classification_model.predict(sample)
    updated_res = result.flatten().astype(int)
    if (updated_res == 1).any():
        st.write("This player have the high potential to be a All-Star player")
    else:
        st.write("This player have low potential to be a All-Star player")