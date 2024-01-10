from draft_state import DraftState
from modes import run_human_vs_human, run_ai_vs_human, run_human_vs_ai, run_ai_vs_ai

# ... (previous code)
model_path1 = 'D:/python_projects/ai/ai_draft_sim/quantized_model/meta_ld_512_x5h.tflite'
model_path2 = 'D:/python_projects/ai/ai_draft_sim/quantized_model/meta_ld_256_x5h.tflite'
hero_roles = 'D:/python_projects/ai/ai_draft_sim/data/hero_roles.csv'

pool1 = [4, 9, 14, 20, 31, 34, 37, 40, 47, 48, 52, 53, 55, 62, 65, 73, 74, 76, 80, 82, 87, 89, 95, 97, 99, 100, 101, 102, 103, 104, 105, 110, 111, 114, 115, 117, 118, 119]
heroes_x = [x for x in range(1, 122)]
pool2 = heroes_x

def main():
    mode = input("Select mode (1: Human vs. Human, 2: AI vs. Human, 3: Human vs. AI, 4: AI vs. AI): ")

    draft_state = DraftState(model_path1, hero_roles)

    if mode == '1':
        run_human_vs_human(draft_state)
    elif mode == '2':
        run_human_vs_ai(draft_state, model_path1, pool1, pool2)
    elif mode == '3':
        run_ai_vs_human(draft_state, model_path1, pool2, pool1)
    elif mode == '4':
        run_ai_vs_ai(draft_state, model_path2, model_path1, pool1, pool1)
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
