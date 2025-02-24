import sys
import os
import argparse  # Import the argparse module
import pickle
import json
from itext2kg import iText2KG
from langchain_ollama import ChatOllama, OllamaEmbeddings
from itext2kg.documents_distiller import DocumentsDistiller, DiseaseArticle


IE_query = '''
# DIRECTIVES:
- As an experienced information extractor, your task is to extract biological entities from the provided bioinformatics context.
- Only extract entities that are explicitly mentioned in the context; do not generate or create any new terms.
- Extracted entities may include, but are not limited to, gene names, protein names, disease names, biological processes, pathways, molecular interactions, and other key bioinformatics terms.
- If an entity is not clearly mentioned in the context, leave it blank and do not infer or generate non-existent information.
- The output should only include the entities, excluding any non-entity content such as descriptive text or inferences.
'''


llm = ChatOllama(
    model="deepseek-r1:32b",
    temperature=0,
)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)

def add_missing_entities(abstract_distilled, pubtator_distilled):
    miss_entities = []
    for entity_type, entities in pubtator_distilled.items():
        if entity_type not in abstract_distilled:  
            abstract_distilled[entity_type] = entities
            miss_entities.append({entity_type:entities})
            print(f'INFO --- {entities} added to the abstract')
        if isinstance(entities, list):  
            existing_names = set()
            if entity_type in abstract_distilled and isinstance(abstract_distilled[entity_type], list):
                existing_names = [item[entity_type] for item in abstract_distilled[entity_type]] #extract existing names, remove last character to match.  Use get for safety
            
            for entity in entities:
                entity_name = entity[entity_type]
                if entity_name and entity_name not in existing_names: 
                    print(f'INFO --- {entity_name} added to the abstract')
                    abstract_distilled[entity_type].append(entity) 
                    miss_entities.append({entity_type:entity})
                    
    return abstract_distilled, miss_entities

def is_entity_in_context(entity, context):
    """
    Checks if an entity exists verbatim (case-insensitive) in a given context,
    returning "true" or "false" as strings.

    Args:
        entity (str): The entity to search for.
        context (str): The text context to search within.

    Returns:
        str: "true" if the entity is found in the context, "false" otherwise.
             Returns "false" if invalid input is detected.
    """

    if not isinstance(entity, str) or not isinstance(context, str):
        print(f"Error: Both entity {entity} and context must be strings.")
        return "false"  # Or raise a TypeError, depending on your needs

    entity_lower = entity.lower()
    context_lower = context.lower()

    if entity_lower in context_lower:
        return True
    else:
        return False
    


document_distiller = DocumentsDistiller(llm_model = llm)


def process_text_and_create_kg(input_txt_path, output_pkl_path):
    """
    Processes a text file, extracts information, builds a knowledge graph,
    and saves the graph to a pickle file.

    Args:
        input_txt_path (str): Path to the input text file.
        output_pkl_path (str): Path to save the output pickle file.
    """

    try:
        with open(input_txt_path, "r") as f:
            text_line = f.readlines()
            if not text_line:
                raise ValueError(f"Input file {input_txt_path} is empty.")

            PMID = text_line[0].split('|')[0]
            title = text_line[0].split('|')[-1].strip()  # Remove potential extra spaces
            abstract = text_line[1].split('|')[-1].strip() # Remove potential extra spaces
            context = f"Title: {title}\nAbstract: {abstract}"

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_txt_path}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error reading or processing the text file: {e}")
        return


    distilled_1 = document_distiller.distill(documents=[context], IE_query=IE_query, output_data_structure=DiseaseArticle)
    json_path = output_pkl_path.split('.')[0]
    with open(f'{json_path}.json', 'w') as f:
        json.dump(distilled_1, f, indent=4)
    
    distilled_1_ = {}
    
    for key, value in distilled_1.items():
        if value != None and value != []:
            if isinstance(value, list):
                distilled_1_[key] = []
                for item in value:
                    for k, v in item.items():
                        if is_entity_in_context(v, context):
                            distilled_1_[key].append({k: v})
            else:
                if is_entity_in_context(value, context):
                    distilled_1_[key] = value
    

    pubtator_info = {}
    pubtator_distilled = {
        'disease': [],
        'gene': [],
        'variant':[],
        'cell_line':[],
        'chemical':[],
    }
    species_info = []

    seen_ids = set()
    for entity_line in text_line[2:]:
        entity_line = entity_line.strip().split("\t")
        if len(entity_line) == 6:
            label = entity_line[4]
            name = entity_line[3]
            if label == 'Gene':
                unique_ID = f"Gene ID:{entity_line[5]}"
            else:
                unique_ID = entity_line[5]
            if name not in seen_ids:
                if label == 'Species':
                    species_info.append(unique_ID)
                    seen_ids.add(name)
                else:
                    pubtator_distilled[label.lower()].append({label.lower():name})
                    pubtator_info[name]={"label": label.lower(), "unique_id": unique_ID}
                    seen_ids.add(name)
                    
    abstract_distilled, _= add_missing_entities(abstract_distilled = distilled_1_, pubtator_distilled = pubtator_distilled)
    
    properties_info = {}
    for k, v in abstract_distilled.items():
        if k not in ['disease', 'pathway', 'gene', 'metabolite', 'protein','processes', 'region','regulation'] and v != "" and v != None and v != []:
            properties_info[k] = v
    properties_info['source'] = f'PMID{PMID}'
    if species_info != []:
        properties_info['species'] = ','.join(list(set(species_info)))

    semantic_blocks_1 = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_1.items() if value !=[] and value != ""  and value is not None]
    semantic_blocks_1.append(f'{context}')

    itext2kg = iText2KG(llm_model = llm, embeddings_model = embeddings)
    try:
        kg1 = itext2kg.build_graph(
        sections=[semantic_blocks_1], 
        source=properties_info,  
        entities_info=pubtator_info,
        ent_threshold=0.9, 
        rel_threshold=0.4
        )
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        return


    # Ensure the output directory exists
    output_dir = os.path.dirname(output_pkl_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return

    try:
        with open(output_pkl_path, "wb") as file:
            pickle.dump(kg1, file)
        print(f"Knowledge graph saved to {output_pkl_path}")  # Success message
    except OSError as e:
        print(f"Error writing to pickle file: {e}")
        return
    except Exception as e:
        print(f"Error pickling the knowledge graph: {e}")
        return


if __name__ == "__main__":
    # # Create the argument parser
    # parser = argparse.ArgumentParser(description="Process a text file and create a knowledge graph.")

    # # Add arguments
    # parser.add_argument("-i", "--input", dest="input_txt_path", help="Path to the input text file.")
    # parser.add_argument("-o", "--output", dest="output_pkl_path", help="Path to save the output pickle file.")

    # # Parse the arguments
    # args = parser.parse_args()

    # # Check if both arguments are provided
    # if not args.input_txt_path or not args.output_pkl_path:
    #     parser.print_help()
    #     sys.exit(1)

    # # Call the processing function with the arguments
    # process_text_and_create_kg(args.input_txt_path, args.output_pkl_path)
    AD_PATH = '/home/jovyan/my_code/itext2kg/datasets/demo_data/abstract/AD'
    AD_OUT_PATH = '/home/jovyan/my_code/itext2kg/output_kg/AD'
    DELIRIUM_PATH = '/home/jovyan/my_code/itext2kg/datasets/demo_data/abstract/Delirium'  
    DELIRIUM_OUT_PATH = '/home/jovyan/my_code/itext2kg/output_kg/Deilirium'  
    for txt_file in os.listdir(AD_PATH):
        if txt_file.endswith('.txt'):
            input_txt_path = os.path.join(AD_PATH, txt_file)
            output_pkl_path = os.path.join(AD_OUT_PATH, txt_file.replace('.txt', '.pkl'))
            process_text_and_create_kg(input_txt_path, output_pkl_path)
    
    for txt_file in os.listdir(DELIRIUM_PATH):
        if txt_file.endswith('.txt'):
            input_txt_path = os.path.join(DELIRIUM_PATH, txt_file)
            output_pkl_path = os.path.join(DELIRIUM_OUT_PATH, txt_file.replace('.txt', '.pkl'))
            process_text_and_create_kg(input_txt_path, output_pkl_path)
    print('Done!')