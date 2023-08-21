import random

input_file = 'C:/Users/Marlon/Desktop/dataset/data/mpl_data.csv'
output_file = 'C:/Users/Marlon/Desktop/dataset/data/aug_mpl_data.csv'

blue_turn = [0, 2, 4, 6, 9, 10, 13, 15, 17, 18]
red_turn = [1, 3, 5, 7, 8, 11, 12, 14, 16, 19]

pick_indices = [6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
ban_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

def split_and_randomize(line):
    heroes = line.strip('[]"\n').split(', ')

    blue_list = [heroes[i] for i in blue_turn if i < len(heroes)]
    red_list = [heroes[i] for i in red_turn if i < len(heroes)]

    blue_sublists = [blue_list[:3], blue_list[3:6], blue_list[6:8], blue_list[8:]]
    red_sublists = [red_list[:3], red_list[3:6], red_list[6:8], red_list[8:]]

    for sublist in blue_sublists + red_sublists:
        random.shuffle(sublist)

    randomized_blue_list = sum(blue_sublists, [])
    randomized_red_list = sum(red_sublists, [])

    randomized_line = []
    for i in range(len(heroes)):
        if i in blue_turn:
            randomized_line.append(randomized_blue_list[blue_turn.index(i)])
        else:
            randomized_line.append(randomized_red_list[red_turn.index(i)])

    return randomized_line

def process_line(line):
    randomized_lines = []
    for _ in range(500):
        randomized_line = split_and_randomize(line)
        randomized_lines.append(randomized_line)
    return randomized_lines

with open(input_file, 'r') as file:
    data = file.readlines()

output_data = []
for line in data:
    line = line.strip()
    processed_lines = process_line(line)
    formatted_lines = ['"[' + ', '.join([str(num) for num in processed_line]) + ']"' for processed_line in processed_lines]
    output_data.append(line)
    output_data.extend(formatted_lines)

with open(output_file, 'w') as file:
    file.writelines('\n'.join(output_data))

print('Augmentation Successful!')
