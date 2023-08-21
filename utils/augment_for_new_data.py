import random

input_file = 'C:/Users/Marlon/Desktop/dataset/data/blck_s11.csv'
output_file = 'C:/Users/Marlon/Desktop/dataset/data/aug_blck_s11.csv'

blue_turn = [0, 2, 4, 6, 9, 10, 13, 15, 17, 18]
red_turn = [1, 3, 5, 7, 8, 11, 12, 14, 16, 19]

pick_indices = [6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
ban_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

def split_and_randomize(draft_sequence):
    blue_list = [draft_sequence[i] for i in blue_turn if i < len(draft_sequence)]
    red_list = [draft_sequence[i] for i in red_turn if i < len(draft_sequence)]

    blue_sublists = [blue_list[:3], blue_list[3:6], blue_list[6:8], blue_list[8:]]
    red_sublists = [red_list[:3], red_list[3:6], red_list[6:8], red_list[8:]]

    for sublist in blue_sublists + red_sublists:
        random.shuffle(sublist)

    randomized_blue_list = sum(blue_sublists, [])
    randomized_red_list = sum(red_sublists, [])

    randomized_draft_sequence = []
    for i in range(len(draft_sequence)):
        if i in blue_turn:
            randomized_draft_sequence.append(randomized_blue_list[blue_turn.index(i)])
        else:
            randomized_draft_sequence.append(randomized_red_list[red_turn.index(i)])

    return randomized_draft_sequence

with open(input_file, 'r') as file:
    data = file.readlines()

output_data = []
for line in data:
    original_line = line.strip()
    randomized_lines = []
    lbl = []
    for _ in range(2500):
        draft_sequence, label = line.strip().rsplit(',', 1)
        draft_sequence = [int(num) for num in draft_sequence.strip('[]"').split(', ')]
        lbl.append(int(label))

        randomized_draft_sequence = split_and_randomize(draft_sequence)
        randomized_line = randomized_draft_sequence
        randomized_lines.append(randomized_line)

    formatted_lines = ['"['+ ', '.join([str(num) for num in processed_line]) + ']",' + str(lbl[0]) for processed_line in randomized_lines]
    
    output_data.append(original_line)  # Append original line
    
    output_data.extend(formatted_lines)

with open(output_file, 'w') as file:
    file.writelines('\n'.join(output_data))
