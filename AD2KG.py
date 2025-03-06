import pandas as pd
import logging
import pickle
import os
from langchain_community.chat_models import ChatOllama  # Use langchain_community
from langchain_ollama import OllamaEmbeddings    # Use langchain_community
from itext2kg.utils import PubtatorProcessor
from itext2kg import iText2KG
import multiprocessing
from functools import partial
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pmid(pmid, pubtator_path, output_path, llm_model_name, embeddings_model_name):
    """
    Processes a single PMID.  Now takes model names instead of instances.
    """
    if os.exists(f'{output_path}/{pmid}.pkl'):
        return
    try:
        # Initialize LLM and Embeddings *inside* the worker process
        llm = ChatOllama(model=llm_model_name, temperature=0)
        embeddings = OllamaEmbeddings(model=embeddings_model_name)

        # Load context
        pubtator_file = f"{pubtator_path}{pmid}.txt"
        pubtator_process = PubtatorProcessor(pubtator_file, llm)
        semantic_blocks = pubtator_process.block
        properties_info = pubtator_process.properties_info
        pubtator_info = pubtator_process.pubtator_info

        pubtator_info['abstract'] = {'context': semantic_blocks[-1], 'source': properties_info['source']}

        # Build the graph
        itext2kg = iText2KG(llm_model=llm, embeddings_model=embeddings)
        kg1 = itext2kg.build_graph(
            sections=[semantic_blocks],
            source=properties_info,
            entities_info=pubtator_info,
            ent_threshold=0.9,
            rel_threshold=0.4
        )

        # Save the graph
        with open(f'{output_path}/{pmid}.pkl', 'wb') as f:
            pickle.dump(kg1, f)

    except FileNotFoundError:
        logging.error(f"PubTator file not found for PMID: {pmid}")
    except Exception as e:
        logging.error(f"Error processing PMID {pmid}: {e}")



def main():
    # Define model names *here*
    llm_model_name = "deepseek-r1:32b"
    embeddings_model_name = "nomic-embed-text:latest"

    # Delirium data paths
    df1 = pd.read_csv("/home/jovyan/my_code/RAG/Delirium_pmid_results_fa.txt", sep='\t')
    pmid_list = list(df1['pmid'].astype(str))
    pmid_list = pmid_list[:10000]
    PUTATOR_PATH = "/home/jovyan/my_code/RAG/Data_v2/delirium/"
    OUTPUT_PATH = "/home/jovyan/my_code/itext2kg/output_kg/Deilirium"

    # Create a partial function, passing model *names*
    process_func = partial(process_pmid, pubtator_path=PUTATOR_PATH, output_path=OUTPUT_PATH,
                           llm_model_name=llm_model_name, embeddings_model_name=embeddings_model_name)

    # Use imap_unordered with tqdm for progress display
    with multiprocessing.Pool(5) as pool:
        for _ in tqdm(pool.imap_unordered(process_func, pmid_list), total=len(pmid_list), desc="Processing PMIDs"):
            pass  # Result is not used, only iteration for progress bar

    print("Done")

if __name__ == "__main__":
    main()