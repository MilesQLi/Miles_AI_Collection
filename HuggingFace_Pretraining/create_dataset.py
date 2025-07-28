import argparse
import re
from pathlib import Path
import datasets
from tqdm import tqdm

def create_text_dataset(folder_path: str, output_path: str, max_length: int):
    """
    Reads all .txt files from a folder, processes them into fixed-length samples,
    and saves them as a Hugging Face Dataset.

    Args:
        folder_path (str): The path to the folder containing .txt files.
        output_path (str): The path where the dataset will be saved.
        max_length (int): The maximum number of characters for each sample.
    """
    print(f"1. Scanning for .txt files in '{folder_path}'...")
    
    # Use pathlib to robustly find all .txt files, including in subdirectories
    p = Path(folder_path)
    if not p.is_dir():
        raise ValueError(f"Error: Provided folder path '{folder_path}' does not exist or is not a directory.")
        
    text_files = list(p.rglob("*.txt"))
    if not text_files:
        raise ValueError(f"Error: No .txt files found in '{folder_path}'.")

    print(f"Found {len(text_files)} .txt files.")

    # 2. Concatenate all text content into a single string
    print("\n2. Reading and concatenating file contents...")
    full_text = ""
    for text_file in tqdm(text_files, desc="Reading files"):
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                full_text += f.read() + "\n\n" # Add paragraph break between files
        except Exception as e:
            print(f"Warning: Could not read file {text_file}: {e}")

    # 3. Split the text into logical units (sentences or paragraphs)
    # This regex splits the text by:
    # - Positive lookbehind for sentence-ending punctuation (.!?)
    # - OR two or more newline characters (paragraphs)
    # The filter removes any empty strings that result from the split.
    print("\n3. Splitting text into sentences and paragraphs...")
    units = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+|\n{2,}', full_text) if s and s.strip()
    ]
    print(f"Split text into {len(units)} logical units.")

    # 4. Group units into samples respecting max_length
    print(f"\n4. Chunking text into samples of max {max_length} characters...")
    samples = []
    current_sample = ""

    for unit in tqdm(units, desc="Chunking"):
        # Handle cases where a single sentence/paragraph is longer than max_length
        if len(unit) > max_length:
            # If there's a sample being built, save it first
            if current_sample:
                samples.append(current_sample)
                current_sample = ""
            # Add the long unit as its own sample, printing a warning
            print(f"\nWarning: A single text unit (sentence/paragraph) was longer "
                  f"than max_length ({len(unit)} > {max_length}). It will be its own sample.")
            samples.append(unit)
            continue

        # Check if adding the next unit would exceed the max length
        # We add 1 for the space character that will join the units
        if len(current_sample) + len(unit) + 1 > max_length:
            # If so, finalize the current sample and start a new one
            samples.append(current_sample)
            current_sample = unit
        else:
            # Otherwise, add the unit to the current sample
            if current_sample:
                current_sample += " " + unit
            else:
                current_sample = unit

    # Add the last remaining sample if it exists
    if current_sample:
        samples.append(current_sample)
    
    print(f"Created {len(samples)} samples.")

    # 5. Create a Hugging Face Dataset object
    print("\n5. Creating Hugging Face Dataset object...")
    
    # The dataset needs a dictionary format: {'column_name': [list_of_values]}
    data = {"text": samples}
    hf_dataset = datasets.Dataset.from_dict(data)

    # 6. Create a DatasetDict with a 'train' split
    dataset_dict = datasets.DatasetDict({"train": hf_dataset})
    print(f"\nDataset structure:\n{dataset_dict}")

    # 7. Save the dataset to disk
    print(f"\n6. Saving dataset to '{output_path}'...")
    dataset_dict.save_to_disk(output_path)

    print("\n✅ All done! Your dataset is ready.")
    print(f"You can now load it using: `from datasets import load_from_disk; ds = load_from_disk('{output_path}')`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of .txt files into a Hugging Face dataset."
    )
    parser.add_argument(
        "folder_path", 
        type=str, 
        help="The path to the folder containing .txt files."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="The path where the final dataset will be saved.",
    )
    parser.add_argument(
        "length",
        type=int,
        help="The maximum number of characters for each training sample.",
    )

    args = parser.parse_args()
    
    create_text_dataset(args.folder_path, args.output_path, args.length)