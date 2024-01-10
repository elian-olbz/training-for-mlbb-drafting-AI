import csv

#Used to split red and blue side moves for easy verification of bans and picks of both sides

# Read the dataset from the CSV file
filename = 'D:/python_projects/ai/ai_draft_sim/data/blck_s11.csv'

blue_turn = [0, 2, 4, 6, 9, 10, 13, 15, 17, 18]
red_turn = [1, 3, 5, 7, 8, 11, 12, 14, 16, 19]

pick_indices = [6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
ban_indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

blue_ban = []
blue_pick = []
red_ban = []
red_pick = []

with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        data = eval(row[0])  # Convert the string representation of the list into an actual list
        blue_ban.append([data[i] for i in ban_indices if i in blue_turn][:5])
        red_ban.append([data[i] for i in ban_indices if i in red_turn][:5])
        blue_pick.append([data[i] for i in pick_indices if i in blue_turn][:5])
        red_pick.append([data[i] for i in pick_indices if i in red_turn][:5])

# Save to a new CSV file
new_filename = 'D:/python_projects/ai/ai_draft_sim/exp_data/verif_blck_s11.csv'

with open(new_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Blue Bans", "Red Bans", "Blue Picks", "Red Picks"])
    for i in range(len(blue_ban)):
        row = [
            blue_ban[i],
            red_ban[i],
            blue_pick[i],
            red_pick[i]
        ]
        writer.writerow([str(lst) for lst in row])

print(f"Split dataset saved to {new_filename} successfully.")
