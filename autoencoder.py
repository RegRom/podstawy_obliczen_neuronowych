# %%
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
import seaborn as sns

# %%

class AutoencoderClassifier:

    def __init__(self, input_size=0):
        self.input_size = input_size
        
        if(input_size == 0):
            self.autoencoder = None
        else:
            self.autoencoder = self.build_autoencoder(input_size)

    def tsne_plot(self, x, y): 
      
        # Setting the plotting background 
        sns.set(style ="whitegrid") 
        
        tsne = TSNE(n_components = 2, random_state = 0) 
        
        # Reducing the dimensionality of the data 
        X_transformed = tsne.fit_transform(x) 
    
        plt.figure(figsize =(12, 8)) 
        
        # Building the scatter plot 
        plt.scatter(X_transformed[np.where(y == 0), 0],  
                    X_transformed[np.where(y == 0), 1], 
                    marker ='o', color ='y', linewidth ='1', 
                    alpha = 0.8, label ='Normal') 
        plt.scatter(X_transformed[np.where(y == 1), 0], 
                    X_transformed[np.where(y == 1), 1], 
                    marker ='o', color ='k', linewidth ='1', 
                    alpha = 0.8, label ='Fraud') 
    
        # Specifying the location of the legend 
        plt.legend(loc ='best') 
        
        # Plotting the reduced data 
        plt.show() 

    def build_autoencoder(self, input_size = 0):
        if(input_size == 0):
            input_size = self.input_size

        input_layer = Input(input_size)

        encoded = Dense(100, activation='tanh', 
            activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoded = Dense(50, activation='tanh',
            activity_regularizer=regularizers.l1(10e-5))(encoded)
        encoded = Dense(25, activation='tanh',
            activity_regularizer=regularizers.l1(10e-5))(encoded)
        encoded = Dense(12, activation='tanh',
            activity_regularizer=regularizers.l1(10e-5))(encoded)
        encoded = Dense(6, activation='tanh',
            activity_regularizer=regularizers.l1(10e-5))(encoded)

        decoded = Dense(12, activation='tanh')(encoded)
        decoded = Dense(25, activation='tanh')(decoded)
        decoded = Dense(50, activation='tanh')(decoded)
        decoded = Dense(100, activation='tanh')(decoded)

        output_layer = Dense(input_size, activation='relu')(decoded)

        autoencoder = Model(input_layer, output_layer, name='autoencoder')
        autoencoder.compile(optimizer='adadelta', loss='mse')

        return autoencoder

    def build_autoencoder_classifier(self, X, y):

        self.autoencoder.summary()
        # self.encoder.summary()
        X_negative = X[y == 0] 
        X_positive = X[y == 1]

        self.autoencoder.fit(X_negative, X_negative,  
                batch_size = 16, epochs = 50,  
                shuffle = True, validation_split = 0.20)


        model = Sequential()
        model.add(self.autoencoder.layers[0])
        model.add(self.autoencoder.layers[1])
        model.add(self.autoencoder.layers[2])
        model.add(self.autoencoder.layers[3])
        model.add(self.autoencoder.layers[4])

        negative_transformed = model.predict(X_negative) 
        positive_transformed = model.predict(X_positive)

        encoded_X = np.append(negative_transformed, positive_transformed, axis = 0) 
        y_negative = np.zeros(negative_transformed.shape[0]) 
        y_positive = np.ones(positive_transformed.shape[0]) 
        encoded_y = np.append(y_negative, y_positive) 
        
        # Plotting the encoded points 
        self.tsne_plot(encoded_X, encoded_y) 

        return model
        # autoencoder_mlp.summary()
                
# %%
# Wygenerowanie zbioru danych
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
from autoencoder import AutoencoderClassifier

X, y = make_classification(
    n_features=500, 
    n_redundant=100,
    n_informative=300,
    n_clusters_per_class=1,
    n_samples=3000,
    weights=[ 0.5, 0.5 ]
    )
print(X.shape)
print(y.shape)

y_T = y[:, np.newaxis]
print(y_T.shape)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_negative_scaled = X_scaled[y == 0] 
X_positive_scaled = X_scaled[y == 1]
        
# %%

# # %% 
# from tensorflow import keras

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# autoencoder = AutoencoderClassifier(X_train.shape[1])

# model = autoencoder.build_autoencoder_classifier(X_train, y_train)

# model.fit(X_train, y_train, epochs=150, batch_size=16, verbose=0)

# loss, acc = model.evaluate(X_test, y_test, verbose=0)

# %%
# print('Test Accuracy: %.3f' % acc)




