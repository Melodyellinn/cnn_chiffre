#!/usr/bin/env python

# coding: utf-8

# In[12]:
import os
import tensorflow as tf
import numpy as np
#import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'model.h5')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy train.ipynb')

model = load_model('model.h5')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

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

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    
    image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    test_x = tf.convert_to_tensor(image)
    
    
    #Version CV2 non supporté par streamlit
    #img2 = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    #img = img2
    #img = img / 255.0
    #img = img.reshape(-1,28,28,1)
    #rescaled = cv2.resize(img2, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    #st.write('Model Input')
    st.image(image)

if st.button('Predict'):
    #test_x = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x)
    #.reshape(-1, 28, 28,1))
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
    
