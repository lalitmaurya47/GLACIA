import os
import cv2
import csv
import random
import numpy as np
from tqdm import tqdm

# ------------------------------
# Function to detect mask position
# ------------------------------
def get_mask_position(mask):
    h, w = mask.shape
    center_x, center_y = w // 2, h // 2
    mask_coords = np.argwhere(mask > 0)
    if mask_coords.size == 0:
        return "no mask", False

    min_y, min_x = mask_coords.min(axis=0)
    max_y, max_x = mask_coords.max(axis=0)

    # Check if tumor crosses the center
    crosses_center = (min_x <= center_x <= max_x) and (min_y <= center_y <= max_y)
    if crosses_center:
        return "center", True

    # Tumor centroid
    tumor_center_x = (min_x + max_x) // 2
    tumor_center_y = (min_y + max_y) // 2

    # Distance to center
    distance_to_center = np.sqrt((tumor_center_x - center_x) ** 2 + (tumor_center_y - center_y) ** 2)
    near_center = distance_to_center < 0.2 * w  # threshold = 20% of image width

    # Determine position quadrant
    if tumor_center_x < center_x and tumor_center_y < center_y:
        position = "top left"
    elif tumor_center_x >= center_x and tumor_center_y < center_y:
        position = "top right"
    elif tumor_center_x < center_x and tumor_center_y >= center_y:
        position = "bottom left"
    else:
        position = "bottom right"

    return position, near_center


# ------------------------------
# Helper functions to load Q&A templates
# ------------------------------
# def load_qa_templates(file_path):
#     templates = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line and '\t' in line:
#                 q, a = line.split('\t')
#                 templates.append((q.strip(), a.strip()))
#     return templates

def load_qa_templates(file_path):
    """
    Splits lines containing f-string question and answer pairs into (question, answer) tuples.
    Each line should look like:
    f"<question>", f"<answer>"
    """
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
         
            for line in f:
                line = line.strip()
                if line.startswith('f"') and '", f"' in line:
                    # Split by the pattern ", f"
                    parts = line.split('", f"')
                    if len(parts) == 2:
                        question = parts[0].replace('f"', '').strip()
                        answer = parts[1].rstrip('"').strip()
                        result.append((question, answer))
        
    return result

def quote_if_comma(text):
    return f'"{text}"' if ',' in text else text


# ------------------------------
# Main processing function
# ------------------------------
def process_images(base_folder, output_file, qa_file):
    splits = ["train", "test"]
    qa_templates = load_qa_templates(qa_file)

    with open(output_file, 'w') as out_file:
        #writer = csv.writer(out_file, delimiter='\t', lineterminator='\n')
        #out_file.write("image_path, image_name, question, answer1, answer2, split\n")
        out_file.write("image_path,image_name,question,answer,position,split\n")

        for split in splits:
            split_folder = os.path.join(base_folder, split, "masks")
            print(split_folder)
            if not os.path.exists(split_folder):
                print('Hello')
                continue

            print(f"Processing split: {split}")
            for image_name in tqdm(os.listdir(split_folder)):
                if not image_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                mask_path = os.path.join(split_folder, image_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                #print(mask.shape)
                if mask is None:
                    continue

                position, near_center = get_mask_position(mask)
                #print(position)
                if position == "no mask":
                    continue

                # Determine proximity phrase
                distance_phrase = "near the center" if near_center else "far from the center"
                extra_phrase = f", {distance_phrase} of the image"

                # Generate multiple Q&A entries per image
                # num_qas = random.randint(3, 5)
                # for _ in range(num_qas):
                question_template, answer_template = random.choice(qa_templates)
                question_template.strip()
                answer_template.strip()
                # Fill placeholders
                #question = question_template.replace("{position_description}", position)
    
                # Add near/far info to answer1
                answer1 = answer_template.replace("{position_description}", position)
                #answer1 = answer_template.replace("{position_description}", f"{position} area{extra_phrase}")
                if position == "center":
                    answer1 = answer1
                else:
                    answer1 = f" {answer1}, {distance_phrase} of the image"
    
    # Quote common substring
    
                quoted_question = quote_if_comma(question_template)
                quoted_answer1 = quote_if_comma(answer1)
    
    
                # Secondary answer style (summary)
                answer2 = f"The glacier lake is in position {position} and {distance_phrase}."
    
                #out_file.write(f"{mask_path}, {image_name}, {question_template}, {answer1}, {answer2}, {split}\n")
                out_file.write(f"{mask_path},{image_name},{quoted_question},{quoted_answer1},{answer2},{split}\n")

    print(f"✅ CSV file '{output_file}' created successfully.")


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    process_images(
        base_folder="/users/lalit47/GLACIA/GLNet_single",
        output_file="/users/lalit47/GLACIA/annotation/glacier_dataset.csv",
        qa_file="/users/lalit47/GLACIA/glacier_lake_qa.txt"
    )
