import pandas as pd
import logging
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
    pubtator_file = f"{PUTATOR_PATH}{PMID}.txt"
    pubtator_process = PubtatorProcessor(pubtator_file, llm)
    semantic_blocks = pubtator_process.block
    properties_info = pubtator_process.properties_info
    pubtator_info = pubtator_process.pubtator_info

        
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
    # Alzheimer
    # df = pd.read_csv("/home/jovyan/my_code/RAG/AD_pmid_results.csv")
    # pmid_list = list(df['pmid'].astype(str))
    # pmid_list = pmid_list[:10]
    # #-------------PATH---------
    # PUTATOR_PATH = "/home/jovyan/my_code/RAG/Data_v2/AD_pubtabor/"
    # JSON_PATH = "/home/jovyan/my_code/itext2kg/output_kg/AD"
    # OUTPUT_PATH = JSON_PATH
    
    # for PMID in pmid_list:
    #     main(PUTATOR_PATH, JSON_PATH, OUTPUT_PATH, PMID)
    # print("Done")
    
    # Delirium
    df1 = pd.read_csv("/home/jovyan/my_code/RAG/Delirium_pmid_results.csv")
    pmid_list = list(df1['pmid'].astype(str))
    pmid_list = pmid_list[:10]  
    PUTATOR_PATH = "/home/jovyan/my_code/RAG/Data_v2/Delirium_pubtator/"
    JSON_PATH = "/home/jovyan/my_code/itext2kg/output_kg/Deilirum"
    
    OUTPUT_PATH = JSON_PATH
    for PMID in pmid_list:
        main(PUTATOR_PATH, OUTPUT_PATH, PMID)
    print("Done")
    
    
    