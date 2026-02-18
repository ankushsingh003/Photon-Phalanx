import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Creates data generators for training, validation, and testing.
    """
    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and testing
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    TRAIN_DIR = "data/processed/train"
    VAL_DIR = "data/processed/val"
    TEST_DIR = "data/processed/test"
    
    train_gen, val_gen, test_gen = get_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)
    print(f"Number of classes: {train_gen.num_classes}")
    print(f"Class indices: {train_gen.class_indices}")
