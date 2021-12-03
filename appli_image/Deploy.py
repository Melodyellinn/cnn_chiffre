import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import streamlit as st
import SessionState
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import matplotlib as plt
MODEL_DIR = os.path.join(os.path.dirname('C:/Users/mattb/Simplon/Rendu/Outil de visualisation pour un réseau neuronal convolutif')
                         , 'model.h5')


model = load_model('model.h5')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)
test = pd.read_csv("test.csv")
st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

st.markdown(""" <style> img {
width:250px !important; height:250px;} 
</style> """, unsafe_allow_html=True)


if canvas_result.image_data is not None:
    img = canvas_result.image_data
    
    image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    st.write('Model Input')
    print(img.shape)
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    test_x = tf.convert_to_tensor(image)
    
if st.button('Predict'):
    val = model.predict(test_x)
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])


def rand_img():
    sample = test.sample()
    row = sample.to_numpy()
    image = row.reshape([28,28])
    return image
def viz_num():
    fig = plt.pyplot.figure(figsize=(5,5))
    plt.pyplot.imshow(image, cmap=plt.pyplot.get_cmap('Blues'))
    return fig
def viz_stat():
    df_stat = pd.DataFrame(ss.list_pred, columns=['Prediction'])
    fig2 = plt.pyplot.figure(figsize=(5,5))
    plt.pyplot.hist(x="Prediction", data=df_stat)
    return fig2

#st.session_state['stat'] = []
#test1 = st.session_state.stat
#st.write(test1)
ss = SessionState.get(list_pred=[])
if st.button('Predict a random image from our dataframe'):
    image = rand_img()
    random_img = viz_num()

    st.pyplot(random_img)
    val2 = model.predict(image.reshape(1, 28, 28, 1))
    st.write(f'result: {np.argmax(val2[0])}')
    st.header('Cette prédiction est elle juste?')
    #ss = SessionState.get(list_pred=[])
if st.button('Oui'):
    ss.list_pred.append('correct')

if st.button('Non'):
    ss.list_pred.append('incorrect')

if st.button('Afficher les Statistics'):
    st.write(ss.list_pred)
    st.text(ss.list_pred)
    st.pyplot(viz_stat())


#ss = SessionState.get(list_pred=[])

#if st.button("test yes"):
    #ss.list_pred.append('yes')
#if st.button("test no"):
    #ss.list_pred.append('no')
    #st.text(ss.list_pred)