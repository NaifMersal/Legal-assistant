import json
from typing import List, Dict, Set
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> dict:
    """Load the QA dataset from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {str(e)}")
        raise

def validate_qa_references(qa: dict) -> bool:
    """
    Validate if all reference_ids are present in selected_articles.
    Returns True if valid, False if invalid.
    """
    if not isinstance(qa.get('references_ids', []), list) or not isinstance(qa.get('selected_articles', []), list):
        return False
    
    reference_set = set(qa['references_ids'])
    selected_articles_set = set(qa['selected_articles'])
    
    return reference_set.issubset(selected_articles_set)

def remove_id_references(qa: dict) -> bool:
    """
    Check if the question contains ID references or article references in Arabic.
    Returns True if the question is clean (no references), False if it contains references.
    """
    question = qa.get('question', '').lower()
    # Check for "(ID: number)" pattern
    if "(id:" in question:
        return False
    # Check for "للمادة number" pattern in Arabic
    if "للمادة" in question:
        return False
    return True

def clean_dataset(dataset: dict) -> tuple[dict, dict]:
    """
    Clean the dataset by removing all QAs from laws that have any invalid QAs.
    Returns the cleaned dataset and statistics.
    """
    # Group QAs by law
    qa_by_law = defaultdict(list)
    for qa in dataset['qa_pairs']:
        qa_by_law[qa['law_name']].append(qa)
    
    # Statistics
    stats = {
        'total_laws': len(qa_by_law),
        'total_qas_before': len(dataset['qa_pairs']),
        'laws_removed': 0,
        'qas_removed': 0,
        'qas_removed_with_refs': 0,
        'laws_with_invalid_qas': set()
    }
    
    # Find laws with invalid QAs
    invalid_laws = set()
    for law_name, qas in qa_by_law.items():
        for qa in qas:
            if not validate_qa_references(qa):
                invalid_laws.add(law_name)
                stats['laws_with_invalid_qas'].add(law_name)
                break
    
    # Create new cleaned dataset
    cleaned_qas = []
    for qa in dataset['qa_pairs']:
        if qa['law_name'] in invalid_laws:
            stats['qas_removed'] += 1
            continue
            
        # Check for ID and article references in questions
        if not remove_id_references(qa):
            stats['qas_removed_with_refs'] += 1
            continue
            
        cleaned_qas.append(qa)
    
    stats['laws_removed'] = len(invalid_laws)
    
    # Update dataset
    cleaned_dataset = dataset.copy()
    cleaned_dataset['qa_pairs'] = cleaned_qas
    cleaned_dataset['metadata']['total_qa_pairs'] = len(cleaned_qas)
    
    # Convert sets to lists for JSON serialization
    stats['laws_with_invalid_qas'] = list(stats['laws_with_invalid_qas'])
    
    return cleaned_dataset, stats

def save_dataset(dataset: dict, file_path: str):
    """Save the dataset to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving dataset to {file_path}: {str(e)}")
        raise

def main():
    input_file = "law_qa_dataset_validated.json"
    output_file = "law_qa_dataset_validated.json"
    
    logger.info(f"Loading dataset from {input_file}")
    dataset = load_dataset(input_file)
    
    logger.info("Cleaning dataset...")
    cleaned_dataset, stats = clean_dataset(dataset)
    
    # Log statistics
    logger.info("="*60)
    logger.info("Cleaning Statistics:")
    logger.info(f"Total laws processed: {stats['total_laws']}")
    logger.info(f"Total QAs before cleaning: {stats['total_qas_before']}")
    logger.info(f"Laws removed: {stats['laws_removed']}")
    logger.info(f"QAs removed due to invalid references: {stats['qas_removed']}")
    logger.info(f"QAs removed due to ID/article references in questions: {stats['qas_removed_with_refs']}")
    logger.info(f"Total QAs removed: {stats['qas_removed'] + stats['qas_removed_with_refs']}")
    logger.info(f"Total QAs after cleaning: {len(cleaned_dataset['qa_pairs'])}")
    logger.info("\nLaws with invalid QAs:")
    for law in stats['laws_with_invalid_qas']:
        logger.info(f"- {law}")
    logger.info("="*60)
    
    logger.info(f"Saving cleaned dataset to {output_file}")
    save_dataset(cleaned_dataset, output_file)
    logger.info("Done!")

if __name__ == '__main__':
    main()