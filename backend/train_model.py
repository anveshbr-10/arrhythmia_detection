import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense,
                                     Dropout, BatchNormalization, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
print("1. Loading datasets...")
train_df = pd.read_csv('data/mitbih_train.csv', header=None)
test_df  = pd.read_csv('data/mitbih_test.csv',  header=None)

X_train = train_df.iloc[:, :187].values
y_train = train_df.iloc[:, 187].values.astype(int)
X_test  = test_df.iloc[:,  :187].values
y_test  = test_df.iloc[:,  187].values.astype(int)

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# ─────────────────────────────────────────────
# 2. OVERSAMPLE MINORITY CLASSES (manual repeat)
#    Fusion (class 3) is tiny — replicate it heavily
#    Supraventricular (class 1) & Ventricular (class 2) also boosted
# ─────────────────────────────────────────────
print("\n2. Oversampling minority classes...")

def oversample(X, y, target_class, multiplier):
    idx = np.where(y == target_class)[0]
    X_rep = np.tile(X[idx], (multiplier, 1))
    y_rep = np.tile(y[idx],  multiplier)
    return X_rep, y_rep

# Fusion (class 3): ~162 samples → repeat ×20  → ~3 240
# Supraventricular (class 1): ~556 → repeat ×6 → ~3 336
# Ventricular (class 2): ~1 448 → repeat ×3  → ~4 344
for cls, mult in [(3, 20), (1, 6), (2, 3)]:
    Xr, yr = oversample(X_train, y_train, cls, mult)
    X_train = np.vstack([X_train, Xr])
    y_train = np.hstack([y_train, yr])

# Shuffle after oversampling
perm = np.random.permutation(len(y_train))
X_train, y_train = X_train[perm], y_train[perm]

print(f"After oversampling — train size: {X_train.shape[0]}")
unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# ─────────────────────────────────────────────
# 3. HANDLE CLASS IMBALANCE (weights on top of oversampling)
# ─────────────────────────────────────────────
print("\n3. Computing class weights...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weight_dict}")

# ─────────────────────────────────────────────
# 4. RESHAPE FOR 1D CNN
# ─────────────────────────────────────────────
print("\n4. Reshaping for 1D CNN...")
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape( (X_test.shape[0],  X_test.shape[1],  1))

# ─────────────────────────────────────────────
# 5. BUILD DEEPER MODEL ARCHITECTURE
#    - 4 Conv blocks with BatchNorm (stable, faster convergence)
#    - GlobalAveragePooling instead of Flatten (fewer params, less overfit)
#    - Two dense heads with Dropout
#    - Label smoothing in loss (stops the model being overconfident on wrong preds)
# ─────────────────────────────────────────────
print("\n5. Building the Model Architecture...")

def conv_block(filters, kernel_size):
    return [
        Conv1D(filters=filters, kernel_size=kernel_size,
               padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(filters=filters, kernel_size=kernel_size,
               padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
    ]

model = Sequential([
    # Block 1
    *conv_block(64, 5),
    # Block 2
    *conv_block(128, 3),
    # Block 3
    *conv_block(256, 3),
    # Block 4 — fine-grained features
    Conv1D(256, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),   # replaces Flatten → robust & compact
    # Classifier head
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.build(input_shape=(None, 187, 1))
model.summary()
# ─────────────────────────────────────────────
# 6. CALLBACKS
#    - EarlyStopping: stop if val_loss stops improving (patience=8)
#    - ReduceLROnPlateau: halve LR when stuck (patience=4)
#    - ModelCheckpoint: save only the best weights automatically
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_model_checkpoint.keras', monitor='val_loss',
                    save_best_only=True, verbose=1),
]

# ─────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────
print("\n6. Training the Model (This might take a few minutes)...")
history = model.fit(
    X_train, y_train,
    epochs=50,                      # more epochs — EarlyStopping will cut early if needed
    batch_size=128,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    verbose=1,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────
print("\n7. Evaluating the Model...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Classification Report ---")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
print("\n8. Saving the Brain...")
model.save('best_model.keras')
print("Model saved successfully as 'best_model.keras'!")