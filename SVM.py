import os
import numpy as np
from PIL import Image
from skimage import color
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score


def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []

    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            label = os.path.basename(subdir)  # Use the directory name as the label
            img_path = os.path.join(subdir, filename)
            try:
                img = Image.open(img_path)
                img = img.resize(image_size, Image.ANTIALIAS)
                images.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Failed to read image: {img_path}. Error: {e}")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return np.array(images), np.array(labels)


def convert_to_grayscale(images):
    gray_images = []
    for img in images:
        gray_img = color.rgb2gray(img)
        gray_images.append(gray_img)
    return np.array(gray_images)


def normalize_images(images):
    return images / 255.0


def split_data(images, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_svm_model(X_train, y_train):
    # Hyperparameter tuning with GridSearchCV
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100]}
    svc = SVC(random_state=42)
    clf = GridSearchCV(svc, parameters, cv=5, scoring='f1_macro')
    clf.fit(X_train.reshape(len(X_train), -1), y_train)
    print(f"Best parameters found: {clf.best_params_}")
    return clf.best_estimator_


def test_svm_model(svm_model, X_test, y_test):
    y_pred = svm_model.predict(X_test.reshape(len(X_test), -1))
    conf_matrix = confusion_matrix(y_test, y_pred)
    avg_f1_score = f1_score(y_test, y_pred, average='macro')
    return conf_matrix, avg_f1_score


def main():
    # Load data
    image_folder = 'train'
    print(f"Image folder path: {image_folder}")
    if not os.path.exists(image_folder):
        print(f"Folder does not exist: {image_folder}")
        return

    images, labels = load_images_from_folder(image_folder)
    if len(images) == 0:
        print("No images loaded. Exiting.")
        return

    # Check for unique labels
    unique_labels = np.unique(labels)
    print(f"Unique labels found: {unique_labels}")
    if len(unique_labels) <= 1:
        print("Error: The dataset does not contain more than one class. Exiting.")
        return

    # Convert to grayscale
    print("Converting images to grayscale...")
    gray_images = convert_to_grayscale(images)
    print(f"Converted {len(gray_images)} images to grayscale.")

    # Normalize images
    print("Normalizing images...")
    normalized_images = normalize_images(gray_images)
    print(f"Normalized images. Shape: {normalized_images.shape}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(normalized_images, labels, test_size=0.2)

    # Train an SVM model
    print("Training SVM model...")
    svm_model = train_svm_model(X_train, y_train)
    print("SVM model trained successfully.")

    # Test the SVM model
    print("Testing SVM model...")
    conf_matrix, avg_f1_score = test_svm_model(svm_model, X_test, y_test)
    print("SVM model tested successfully.")

    # Print evaluation metrics
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Average F-1 Score: {avg_f1_score}")


if __name__ == "__main__":
    main()
