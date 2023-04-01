from keras.models import Sequential
from keras.layers import Dense

def create_ann_model(X_train_smote):
    ann_model = Sequential()
    ann_model.add(Dense(10, input_dim=X_train_smote.shape[1], activation='relu'))
    ann_model.add(Dense(1, activation='sigmoid'))
    ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return ann_model