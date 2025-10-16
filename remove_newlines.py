import json

# Path to the JSON file
file_path = 'saudi_laws_scraped.json'
# Path to the new JSON file
new_file_path = 'saudi_laws_cleaned.json'

# Load the JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Recursively remove newlines from strings in the JSON data
def remove_newlines(data):
    if isinstance(data, str):
        return data.replace('\n', ' ')
    elif isinstance(data, list):
        return [remove_newlines(item) for item in data]
    elif isinstance(data, dict):
        return {key: remove_newlines(value) for key, value in data.items()}
    return data

# Save the modified JSON data back to the file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Load the JSON file
    json_data = load_json(file_path)

    # Remove newlines
    cleaned_data = remove_newlines(json_data)

    # Save the cleaned data to a new file
    save_json(cleaned_data, new_file_path)

    print(f"Newlines removed and cleaned data saved to {new_file_path}.")