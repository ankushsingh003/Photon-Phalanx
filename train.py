import os
import argparse
from scripts.preprocess import get_data_generators
from models.baseline_cnn import build_baseline_cnn
from models.transfer_learning import build_transfer_learning_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model_type='mobilenet_v2', epochs=10, batch_size=32):
    # Paths
    TRAIN_DIR = "data/processed/train"
    VAL_DIR = "data/processed/val"
    TEST_DIR = "data/processed/test"
    
    # Data Generators
    train_gen, val_gen, test_gen = get_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR, batch_size=batch_size)
    num_classes = train_gen.num_classes
    
    # Model Selection
    if model_type == 'baseline':
        model = build_baseline_cnn(num_classes=num_classes)
    else:
        model = build_transfer_learning_model(model_name=model_type, num_classes=num_classes)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        f'saved_models/best_{model_type}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    os.makedirs('saved_models', exist_ok=True)
    
    # Training
    print(f"Starting training for {model_type}...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop]
    )
    
    # Evaluation on Test set
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='baseline, mobilenet_v2, efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train_model(model_type=args.model, epochs=args.epochs)
