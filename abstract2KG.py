import pandas as pd
import logging
import os
import pickle
from langchain_ollama import ChatOllama, OllamaEmbeddings
from itext2kg.utils import PubtatorProcessor
from itext2kg import iText2KG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------LLM and Embeddings models---------
llm = ChatOllama(
    model="deepseek-r1:32b",
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)


def main(PUTATOR_PATH, OUTPUT_PATH, PMID):
    #----------------load context----------------
    pubtator_file = f"{PUTATOR_PATH}/{PMID}.txt"
    pubtator_process = PubtatorProcessor(pubtator_file, llm)
    semantic_blocks = pubtator_process.block
    properties_info = pubtator_process.properties_info
    pubtator_info = pubtator_process.pubtator_info
    
    pubtator_info['abstract'] = {'context':semantic_blocks[-1], 'source':properties_info['source']}

        
    # -----------------Build the graph-----------------
    itext2kg = iText2KG(llm_model = llm, embeddings_model = embeddings)
    kg1 = itext2kg.build_graph(
        sections=[semantic_blocks], 
        source=properties_info,  
        entities_info=pubtator_info,
        ent_threshold=0.9, 
        rel_threshold=0.4
        )

    # -----------------save the graph-----------------
    with open(f'{OUTPUT_PATH}/{PMID}.pkl', 'wb') as f:  
        pickle.dump(kg1, f)
    

if __name__ == "__main__":
    # Delirium
    DATA_PATH = "/home/mindrank/fuli/itext2kg/Data/delirium"
    OUTPUT_PATH = "/home/mindrank/fuli/itext2kg/output_kg/delirium"
    
    for file_name in os.listdir(DATA_PATH):
        PMID  = file_name.split('.')[0]
        main(DATA_PATH, OUTPUT_PATH, PMID)
        break
    print("Done")
    

    
    
    