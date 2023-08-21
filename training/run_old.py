
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import random

# Load the trained model
model = tf.keras.models.load_model('C:/Users/Marlon/Desktop/dataset/models/bid1_ld256_raw.tf')

# Define the indices for blue and red turns
blue_turn = [0, 2, 4, 6, 9, 10, 13, 15, 17, 18]
red_turn = [1, 3, 5, 7, 8, 11, 12, 14, 16, 19]

# Define the indices for picks and bans
pick_indices = [6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
ban_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

# Get the number of heroes
num_heroes = 121  # Adjust this based on the actual number of unique heroes in your dataset

max_sequence_length = 19

def generate_draft_sequence(padded_sequence, blue_pick_roles, red_pick_roles, is_picking):
    predictions = model.predict(padded_sequence)[0]
    valid_heroes = [hero_id for hero_id in range(num_heroes - 1) if hero_id not in draft_sequence]
    valid_predictions = [predictions[hero_id] for hero_id in valid_heroes]

    if not valid_predictions:
        # If no valid hero IDs are left, select a random hero
        next_hero_id = random.choice(valid_heroes)
        print("Random selection:", next_hero_id)
    else:
        if is_picking:
            valid_predictions_filtered = []
            # Check if the prediction of AI does not have the same role as the previous picks
            for hero_id, prediction in zip(valid_heroes, valid_predictions):
                if get_role(hero_id) not in red_pick_roles:
                    valid_predictions_filtered.append(prediction)

            if not valid_predictions_filtered:
                # If no valid hero IDs with unique roles are left, select a random hero (error handling)
                next_hero_id = random.choice(valid_heroes)
                print("Random selection:", next_hero_id)
            else:
                max_prediction_idx = np.argmax(valid_predictions_filtered)
                next_hero_id = valid_heroes[valid_predictions.index(valid_predictions_filtered[max_prediction_idx])]
                prediction_probability = valid_predictions_filtered[max_prediction_idx]
                print("Model prediction:   ", get_name(next_hero_id), "   |",  "   Probability -", prediction_probability)
        else:
            # Banning turn for AI
            valid_predictions_filtered = []
            for hero_id, prediction in zip(valid_heroes, valid_predictions):
                if len(red_pick_roles) > 2:
                    # Target banning stage (3rd and 4th ban)
                    if get_role(hero_id) not in blue_pick_roles and get_role(hero_id) in red_pick_roles:
                        valid_predictions_filtered.append(prediction)
                else:
                    # First 3 bans
                    if get_role(hero_id) not in blue_pick_roles:
                     valid_predictions_filtered.append(prediction)

            max_prediction_idx = np.argmax(valid_predictions_filtered)
            next_hero_id = valid_heroes[valid_predictions.index(valid_predictions_filtered[max_prediction_idx])]
            prediction_probability = valid_predictions_filtered[max_prediction_idx]
            print("Model prediction:   ", get_name(next_hero_id), "   |",  "   Probability -", prediction_probability)
    
    return next_hero_id


# Load the hero roles from CSV
hero_roles = {}
hero_names = []
with open('C:/Users/Marlon/Desktop/dataset/data/hero_roles.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        hero_id = int(row['HeroID'])
        roles = [role.strip() for role in row['Role'].split('/')]
        hero_roles[hero_id] = roles
        hero_name = row['Name']
        hero_names.append(hero_name)

def get_name(hero_id):
    return hero_names[hero_id - 1]

def get_role(hero_id):
    return hero_roles.get(hero_id, [])

# Initialize the draft sequence
draft_sequence = []
final_sequence = []
blue_actions = [[], []]  #blue ban is [0], blue picks is [1]
red_actions = [[], []]  #red ban is [0], red picks is [1]
blue_pick_roles = []
red_pick_roles = []

# Draft alternatingly between human and AI
for i in range(20):
    if i in blue_turn:
        if i in pick_indices:
            # Human picks for blue side
            human_pick = int(input("Blue pick: "))
            while human_pick in draft_sequence:
                print("Invalid Move!")
                human_pick = int(input("Blue pick: "))
            draft_sequence.append(human_pick)
            final_sequence.append(human_pick)
            blue_actions[1].append(human_pick)
            blue_pick_roles.append(get_role(human_pick))
        else:
            # Human bans for blue side
            human_ban = int(input("Blue ban: "))
            while human_ban in draft_sequence:
                print("Invalid Move!")
                human_ban = int(input("Blue ban: "))
            draft_sequence.append(human_ban)
            final_sequence.append(human_ban)
            blue_actions[0].append(human_ban)
    elif i in red_turn:
        if i in pick_indices:
            # AI picks for red side
            while True:
                padded_sequence = pad_sequences([draft_sequence], maxlen=max_sequence_length, padding='post')
                next_hero_id = generate_draft_sequence(padded_sequence, blue_pick_roles, red_pick_roles, True)
                if next_hero_id not in draft_sequence:
                    break

            draft_sequence.append(next_hero_id)
            final_sequence.append(next_hero_id)
            red_actions[1].append(next_hero_id)
            red_pick_roles.append(get_role(next_hero_id))
            print("Red Pick:", get_name(next_hero_id))
        else:
            # AI bans for red side
            while True:
                padded_sequence = pad_sequences([draft_sequence], maxlen=max_sequence_length, padding='post')
                next_ban_id = generate_draft_sequence(padded_sequence, blue_pick_roles, red_pick_roles, False)
                if next_ban_id not in draft_sequence:
                    break

            draft_sequence.append(next_ban_id)
            final_sequence.append(next_ban_id)
            red_actions[0].append(next_ban_id)
            print("Red Ban:", get_name(next_ban_id))

    print("=========== Draft Status ===========")
    print("Blue Bans:  ", ', '.join(get_name(hero_id) for hero_id in blue_actions[0]))
    print("Blue Picks: ", ', '.join(get_name(hero_id) for hero_id in blue_actions[1]))
    print("Blue roles: {}".format(blue_pick_roles))
    print("")
    print("Red Bans:   ", ', '.join(get_name(hero_id) for hero_id in red_actions[0]))
    print("Red Picks:  ", ', '.join(get_name(hero_id) for hero_id in red_actions[1]))
    print("Red roles: {}".format(red_pick_roles))
    print("====================================\n\n")

print("Final draft:")
print("Blue Team: ", ', '.join(get_name(hero_id) for hero_id in blue_actions[1]))
print("Red Team:  ", ', '.join(get_name(hero_id) for hero_id in red_actions[1]))