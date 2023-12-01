# Importar las bibliotecas necesarias
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
# Asegúrate de tener el conjunto de datos en el mismo directorio que tu script de Python o ajusta la ruta según sea necesario
dataset = pd.read_csv("file.csv", encoding='utf-8')
dataset = dataset.iloc[:, 2:]

# Dividir el conjunto de datos en entradas (X) y la etiqueta / objetivo (y)
X = dataset.iloc[:,0:23] # Asume que tienes 20 características
Y = dataset.iloc[:,23] # Asume que 'default payment next month' es la columna 21

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Definir el modelo
model = Sequential()
model.add(Dense(1, input_dim=23)) # Capa de entrada y primera capa oculta
model.add(Dense(124, activation='sigmoid')) # Segunda capa oculta
model.add(Dense(64, activation='softmax')) # Tercera capa oculta
model.add(Dense(1, activation='tanh')) # Capa de salida


from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# Definir la parada temprana
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)

# Compilar el modelo
sgd = SGD(learning_rate=0.01, momentum=0.9)
rmsprop = RMSprop(learning_rate=0.01)
adagrad = Adagrad(learning_rate=0.01)
adadelta = Adadelta(learning_rate=1.0)
adam = Adam(learning_rate=0.01)
adamax = Adamax(learning_rate=0.01)
nadam = Nadam(learning_rate=0.01)
# Ajustar el modelo

optimizers = [sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam]

scores = []

for opt in optimizers :

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)


    # Ajustar el modelo
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10, callbacks=[early_stop])

    score = model.evaluate(X_test, Y_test)
    
    scores.append(score[1]*100)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    
plt.bar(['SGD','RMSProp','Adagrad','Adadelta','Adam','Adamax','Nadam'],scores);
plt.show()