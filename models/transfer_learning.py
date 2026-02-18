from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def build_transfer_learning_model(model_name='mobilenet_v2', input_shape=(224, 224, 3), num_classes=6):
    """
    Builds a Transfer Learning model using MobileNetV2 or EfficientNetB0.
    """
    if model_name == 'mobilenet_v2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'efficientnet_b0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name. Use 'mobilenet_v2' or 'efficientnet_b0'.")

   
    base_model.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    
    mobilenet = build_transfer_learning_model(model_name='mobilenet_v2')
    print("MobileNetV2 Model Summary:")
    mobilenet.summary()

    
    efficientnet = build_transfer_learning_model(model_name='efficientnet_b0')
    print("\nEfficientNetB0 Model Summary:")
    efficientnet.summary()
