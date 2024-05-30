import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf


def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []

    for subdir, _, files in os.walk(folder):
        for filename in files:
            label = os.path.basename(subdir)
            img_path = os.path.join(subdir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
            else:
                print(f"Failed to read image: {img_path}")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return np.array(images), np.array(labels)


def convert_to_grayscale(images):
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    return np.array(gray_images)


def normalize_images(images):
    return images / 255.0


def split_data(images, labels, test_size=0.2, validation_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                      random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_mlp_model(input_shape, hidden_layers):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for neurons in hidden_layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    return history


def plot_curves(history, model_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    conf_matrix = confusion_matrix(y_test, y_pred)
    avg_f1_score = f1_score(y_test, y_pred, average='macro')

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Average F-1 Score: {avg_f1_score}")


def main():
    # Load data
    image_folder = '/kaggle/input/gender-classification-dataset/Training'
    print(f"Image folder path: {image_folder}")
    if not os.path.exists(image_folder):
        print(f"Folder does not exist: {image_folder}")
        return

    images, labels = load_images_from_folder(image_folder)
    if len(images) == 0:
        print("No images loaded. Exiting.")
        return

    # Encode labels to binary (0 or 1)
    labels = np.array([1 if label == 'female' else 0 for label in labels])

    # Convert to grayscale and normalize images
    gray_images = convert_to_grayscale(images)
    gray_images = gray_images[..., np.newaxis]  # Add channel dimension for grayscale images
    normalized_gray_images = normalize_images(gray_images)
    normalized_rgb_images = normalize_images(images)

    # Split data
    X_train_gray, X_val_gray, X_test_gray, y_train, y_val, y_test = split_data(normalized_gray_images, labels)
    X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test = split_data(normalized_rgb_images, labels)

    print(f"Shape of X_train_gray: {X_train_gray.shape}")
    print(f"Shape of X_val_gray: {X_val_gray.shape}")
    print(f"Shape of X_test_gray: {X_test_gray.shape}")
    print(f"Shape of X_train_rgb: {X_train_rgb.shape}")
    print(f"Shape of X_val_rgb: {X_val_rgb.shape}")
    print(f"Shape of X_test_rgb: {X_test_rgb.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_val: {y_val.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # Train MLP models on grayscale images
    mlp_model1 = build_mlp_model(input_shape=X_train_gray[0].shape, hidden_layers=[512, 256, 128, 64])
    mlp_history1 = train_model(mlp_model1, X_train_gray, y_train, X_val_gray, y_val)
    plot_curves(mlp_history1, "MLP Model 1")

    mlp_model2 = build_mlp_model(input_shape=X_train_gray[0].shape, hidden_layers=[1024, 512, 256, 128, 64, 32])
    mlp_history2 = train_model(mlp_model2, X_train_gray, y_train, X_val_gray, y_val)
    plot_curves(mlp_history2, "MLP Model 2")

    # Determine and save best MLP model
    best_mlp_model = mlp_model1 if max(mlp_history1.history['val_accuracy']) > max(
        mlp_history2.history['val_accuracy']) else mlp_model2
    best_mlp_model_name = "MLP Model 1" if best_mlp_model == mlp_model1 else "MLP Model 2"

    print(f"The best MLP model is: {best_mlp_model_name}")
    best_mlp_model.save('best_mlp_model.h5')

    # Reload best MLP model
    loaded_model_mlp = tf.keras.models.load_model('best_mlp_model.h5')

    # Evaluate best MLP model
    print("Best MLP Model Evaluation:")
    evaluate_model(loaded_model_mlp, X_test_gray, y_test)

    # Clear GPU memory to avoid OOM errors
    tf.keras.backend.clear_session()

    # Train CNN models
    cnn_model_gray = build_cnn_model(input_shape=X_train_gray[0].shape)
    cnn_history_gray = train_model(cnn_model_gray, X_train_gray, y_train, X_val_gray, y_val)
    plot_curves(cnn_history_gray, "Grayscale CNN")
    cnn_model_gray.save('best_model_gray.h5')

    # Clear GPU memory to avoid OOM errors
    tf.keras.backend.clear_session()

    cnn_model_rgb = build_cnn_model(input_shape=X_train_rgb[0].shape)
    cnn_history_rgb = train_model(cnn_model_rgb, X_train_rgb, y_train, X_val_rgb, y_val)
    plot_curves(cnn_history_rgb, "RGB CNN")
    cnn_model_rgb.save('best_model_rgb.h5')

    # Reload best CNN models
    loaded_model_gray = tf.keras.models.load_model('best_model_gray.h5')
    loaded_model_rgb = tf.keras.models.load_model('best_model_rgb.h5')

    # Evaluate CNN models
    print("Grayscale CNN Model Evaluation:")
    evaluate_model(loaded_model_gray, X_test_gray, y_test)

    print("RGB CNN Model Evaluation:")
    evaluate_model(loaded_model_rgb, X_test_rgb, y_test)

    # Compare results and suggest the best model
    best_cnn_val_acc = max(cnn_history_gray.history['val_accuracy'])
    best_rgb_val_acc = max(cnn_history_rgb.history['val_accuracy'])
    best_mlp_val_acc = max(mlp_history1.history['val_accuracy']) if max(mlp_history1.history['val_accuracy']) > max(
        mlp_history2.history['val_accuracy']) else max(mlp_history2.history['val_accuracy'])

    best_model_type = "Grayscale CNN" if best_cnn_val_acc > best_rgb_val_acc else "RGB CNN"
    best_model_val_acc = best_cnn_val_acc if best_cnn_val_acc > best_rgb_val_acc else best_rgb_val_acc

    best_model_type = "MLP" if best_mlp_val_acc > best_model_val_acc else best_model_type
    best_model_val_acc = best_mlp_val_acc if best_mlp_val_acc > best_model_val_acc else best_model_val_acc

    print(f"The best model is the {best_model_type} with a validation accuracy of {best_model_val_acc:.4f}")


if __name__ == "__main__":
    main()