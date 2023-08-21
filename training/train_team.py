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

# Define the indices for blue and red turns
blue_turn = [0, 2, 4, 6, 9, 10, 13, 15, 17, 18]
red_turn = [1, 3, 5, 7, 8, 11, 12, 14, 16, 19]

# Read the dataset file
with open('C:/Users/Marlon/Desktop/dataset/data/aug_blck_s11.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        draft_sequence = eval(row['Draft'])
        side_indicator = int(row['Side'])  # Assuming 'SideIndicator' column contains 0 or 1
        preprocessed_sequences.append((draft_sequence, side_indicator))

# Generate input-output pairs for training
input_sequences = []
output_sequences = []

for sequence, side_indicator in preprocessed_sequences:
    input_sequence = []
    for i in range(1, len(sequence) + 1):  # Start from 1 instead of 0
        output_sequence = sequence[i-1]  # Access sequence elements correctly
        if side_indicator == 0 and i == 1:  # Adjust the condition for blue side
            # Add an empty list as the first input for the blue side
            input_sequence.append([])
        else:
            input_sequence.append(sequence[:i])
        input_sequences.append(input_sequence)
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
latent_dim = 256

input_drafts = Input(shape=(max_sequence_length,))
side_indicator = Input(shape=(1,))
embedding = Embedding(num_numbers, latent_dim)(input_drafts)
lstm = LSTM(latent_dim)(embedding)  # Use LSTM instead of Bidirectional LSTM

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
    EarlyStopping(monitor='loss', patience=3),  # Early stopping if the validation loss does not improve for 5 epochs
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
saved_model_h5_path = 'C:/Users/Marlon/Desktop/dataset/models/blacklist/blck_ld_256_x1k.h5'
model.save(saved_model_h5_path)

# Save the fine-tuned model as a .tf file
saved_model_tf_path = 'C:/Users/Marlon/Desktop/dataset/models/blacklist/blck_ld_256_x1k.tf'
model.save(saved_model_tf_path)

# Save the fine-tuned model in tfjs format
saved_model_tfjs_path = 'C:/Users/Marlon/Desktop/dataset/models/blacklist/blck_ld_256_x1k.tfjs'
tfjs.converters.save_keras_model(model, saved_model_tfjs_path)
