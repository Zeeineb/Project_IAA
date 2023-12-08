import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def build_model(input_shape=(256, 256, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(base_dir, img_size=(256, 256), validation_split=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',  
        subset='training',
        shuffle=True
    )

    test_generator = datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, test_generator

def train_model(model, train_generator, epochs=10):
    history = model.fit(train_generator, epochs=epochs)
    return model, history

def evaluate_model(model, test_generator):
    predictions = model.predict(test_generator)
    predicted_classes = predictions.argmax(axis=1)
    true_classes = test_generator.classes

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    accuracy = accuracy_score(true_classes, predicted_classes)

    print("Matrice de Confusion:\n", conf_matrix)
    print("\nPrécision Globale:", accuracy)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Étiquettes')
    plt.title('Matrice de Confusion')
    # Display overall accuracy below the confusion matrix
    plt.text(0.5, -0.1, f'Précision Globale: {accuracy:.4f}', fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.show()

    print("\nPrécision Globale:", accuracy)

    return accuracy

def save_model(model, path):
    model.save(path)

def main():
    base_dir = 'C:/Users/ASUS/Desktop/Projet IAA - Last version/Dataset'
    img_size = (256, 256)
    epochs = 10

    # Build the model
    model = build_model()

    # Load and preprocess data
    train_generator, test_generator = load_data(base_dir, img_size=img_size)

    # Train the model
    model, history = train_model(model, train_generator, epochs=epochs)

    # Evaluate and print metrics
    evaluate_model(model, test_generator)

    # Save the trained model
    save_model(model, 'C:/Users/ASUS/Desktop/Projet IAA - Last version/MyModel')

if __name__ == "__main__":
    main()