import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import tensorflowjs as tfjs

# Empty lists to store preprocessed data
preprocessed_sequences = []

# Read the dataset file
with open('C:/Users/Marlon/Desktop/dataset/data/aug_mpl_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        draft_sequence = eval(row['Draft'])
        preprocessed_sequences.append(draft_sequence)

# Generate input-output pairs for training
input_sequences = []
output_sequences = []
for sequence in preprocessed_sequences:
    input_sequence = [[]]
    output_sequence = sequence[0]
    for i in range(1, len(sequence)):
        input_sequence = sequence[:i]
        output_sequence = sequence[i]
        input_sequences.append(input_sequence)
        output_sequences.append(output_sequence)


# Pad the input sequences
max_sequence_length = max(map(len, input_sequences))
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

# Convert the output sequences to numpy array
target_sequences = np.array(output_sequences)

# Define the number of unique heroes
num_numbers = 121

# Define the model architecture
latent_dim = 512

inputs = Input(shape=(max_sequence_length,))
embedding = Embedding(num_numbers, latent_dim)(inputs)
lstm = LSTM(latent_dim)(embedding)
outputs = Dense(num_numbers, activation='softmax')(lstm)

# Define the model
model = Model(inputs, outputs)

# Set the learning rate for the Adam optimizer
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Print the model summary
model.summary()

# Define callbacks
callbacks = [
    ReduceLROnPlateau(),  # Reduce learning rate on plateau
    EarlyStopping(monitor='loss',patience=3),  # Early stopping if the validation loss does not improve for 5 epochs
    TensorBoard(log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'),  # TensorBoard callback for visualization
]

# Train the model
model.fit(
    padded_input_sequences,
    target_sequences,
    validation_split=0.2,
    batch_size=1000,
    epochs=50,
    callbacks=callbacks,
)

# Save the fine-tuned model as a .h5 file
saved_model_h5_path = 'C:/Users/Marlon/Desktop/dataset/models/meta/meta_ld_1024_x5h.h5'
model.save(saved_model_h5_path)

# Save the fine-tuned model as a .tf file
saved_model_tf_path = 'C:/Users/Marlon/Desktop/dataset/models/meta/meta_ld_1024_x5h.tf'
model.save(saved_model_tf_path)

# Save the fine-tuned model in tfjs format
saved_model_tfjs_path = 'C:/Users/Marlon/Desktop/dataset/models/meta/meta_ld_1024_x5h.tfjs'
tfjs.converters.save_keras_model(model, saved_model_tfjs_path)