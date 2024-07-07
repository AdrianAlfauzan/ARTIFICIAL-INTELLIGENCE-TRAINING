import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Update the paths according to your directory structure
normal = pathlib.Path("D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 8/dataset/normal")
glaucoma = pathlib.Path("D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 8/dataset/glaucoma")
retinopathy = pathlib.Path("D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 8/dataset/diabetic_retinopathy")
cataract = pathlib.Path("D:/PRAKTIKUM KECERDASAN BUATAN/MODUL 8/dataset/cataract")

images_dict = {
    "normal": list(normal.glob("*.jpg")),
    "glaucoma": list(glaucoma.glob("*.jpg")),
    "diabetic_retinopathy": list(retinopathy.glob("*.jpg")),
    "cataract": list(cataract.glob("*.jpg"))
}

labels_dict = {
    "normal": 0, "glaucoma": 1, "diabetic_retinopathy": 2, "cataract": 3
}

X, y = [], []

for label, images in images_dict.items():
    for image in images:
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (180, 180))
        if image is not None:
            X.append(image)
            y.append(labels_dict[label])

X = np.array(X)
y = np.array(y)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Check if X or y is empty
if X.shape[0] == 0 or y.shape[0] == 0:
    print("No images loaded. Check the file paths.")

else:
    X = X / 255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    import tensorflow as tf
    from tensorflow.keras import layers

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.3),
        tf.keras.layers.RandomZoom(height_factor=0.3, width_factor=0.3),
    ])

    model = tf.keras.Sequential([
        data_augmentation,
        layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, (5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(50, activation="relu"),
        layers.Dense(4, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    pred = model.predict(X_test[:10])
    predlist = []
    for i in pred:
        predlist.append(np.argmax(i))

    answers = ["normal", "glaucoma", "diabetic_retinopathy", "cataract"]
    for i in range(10):
        plt.imshow(X_test[i])
        plt.title("Predicted : " + str(answers[predlist[i]]))
        plt.xlabel("Actual : " + str(answers[y_test[i]]))
        plt.show()
