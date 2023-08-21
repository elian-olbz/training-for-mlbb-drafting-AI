import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, concatenate
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
        side_indicator = int(row['Side'])  # Assuming 'SideIndicator' column contains 0 or 1
        preprocessed_sequences.append((draft_sequence, side_indicator))

# Generate input-output pairs for training
input_sequences = []
output_sequences = []
for sequence, side_indicator in preprocessed_sequences:
    for i in range(1, len(sequence)):
        input_sequence = sequence[:i]
        output_sequence = sequence[i]
        input_sequences.append((input_sequence, side_indicator))
        output_sequences.append(output_sequence)

# Pad the input sequences
max_sequence_length = max(map(len, [seq for seq, _ in input_sequences]))
padded_input_sequences = [seq for seq, _ in input_sequences]
padded_input_sequences = pad_sequences(padded_input_sequences, maxlen=max_sequence_length, padding='post')

# Convert the output sequences to numpy array
target_sequences = np.array(output_sequences)

# Define the number of unique heroes
num_numbers = 121

# Define the model architecture
latent_dim = 128

input_drafts = Input(shape=(max_sequence_length,))
side_indicator = Input(shape=(1,))
embedding = Embedding(num_numbers, latent_dim)(input_drafts)
lstm = Bidirectional(LSTM(latent_dim))(embedding)  # Use Bidirectional wrapper

concatenated = concatenate([lstm, side_indicator])
outputs = Dense(num_numbers, activation='softmax')(concatenated)

# Define the model
model = Model([input_drafts, side_indicator], outputs)

# Set the learning rate for the Adam optimizer
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Print the model summary
model.summary()

# Define callbacks
callbacks = [
    ReduceLROnPlateau(),  # Reduce learning rate on plateau
    EarlyStopping(patience=5, monitor='loss'),  # Early stopping if the validation loss does not improve for 5 epochs
    TensorBoard(log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'),  # TensorBoard callback for visualization
]

# Prepare the input data as a list of two arrays
input_data = [padded_input_sequences, np.array([side for _, side in input_sequences])]

# Train the model
model.fit(
    input_data,
    target_sequences,
    validation_split=0.2,
    batch_size=1000,
    epochs=30,
    callbacks=callbacks,
)

# Save the fine-tuned model as a .h5 file
saved_model_h5_path = 'C:/Users/Marlon/Desktop/dataset/models/blacklist/blck_ld_128_x1k.h5'
model.save(saved_model_h5_path)

# Save the fine-tuned model as a .tf file
saved_model_tf_path = 'C:/Users/Marlon/Desktop/dataset/models/blacklist/blck_ld_128_x1k.tf'
model.save(saved_model_tf_path)

# Save the fine-tuned model in tfjs format
saved_model_tfjs_path = 'C:/Users/Marlon/Desktop/dataset/models/blacklist/blck_ld_128_x1k.tfjs'
tfjs.converters.save_keras_model(model, saved_model_tfjs_path)
