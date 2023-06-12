import streamlit as st

st.title("전남대 AI")

st.header("Linear Regression")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import *
import io
from contextlib import redirect_stdout

num_epochs = 100
data_np = np.loadtxt('./data.csv', dtype = float, delimiter = ',')
st.text('data_csv file -------------')
st.write(data_np)
st.write(data_np.shape)
x = data_np[: , 0:1]
y = data_np[: , 1:2]


col1, col2 = st.columns(2)

col1.header("x values")
col1.write(x)

col2.header("y values")
col2.write(y)

fig = plt.figure(figsize = (5,2))
plt.plot(x,y)
plt.scatter(plt)

fig = plt.figure(figsize = (5,2))
plt.plot(x,y)
plt.scatter(x,y)
st.pyplot(plt)

st.write(x.shape)
st.write(x.shape[1:])

x1_input = input(shape=(x.shape[1:]), name = 'x1_input')
x1_Dense_1 = layers.Dense(50, name = 'Dense_1')(x1_input)
x1_Dense_2 = layers.Dense(50, name = 'Dense_2')(x1_Dense_1)
final = layers.Dense(1, name = 'final')(x1_Dense_2)
model = model(inputs=x1_input, outputs = final)
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()


buf = io.StringIO()
with redirect_stdout(buf):
    model.summary()

st.text(buf.getvalue())
model_train = model.fit(x, y, epochs=st.sidebar.slider("epochs 횟수", min_value=0, max_value=200, value=100, step=5))

fig = plt.figure(figsize = (15,5))
plt.plot(model_train.history['loss'])
st.text("model train")
st.pyplot(plt)

prediction = model.predict([x])
fig = plt.figure(figsize = (15,5))
plt.plot(y, label='Actual')
plt.plot(prediction, label='prediction')

plt.legend()
st.text("model prediction")
st.pyplot(plt)


sidebar_date = st.sidebar.date_input("작성 날짜")
sidebar_time = st.sidebar.time_input("작성 시간")
fig.canvas.manager.full_screen_toggle()

