from typing import List
from .ientities_extraction import iEntitiesExtractor
from .irelations_extraction import iRelationsExtractor
from .utils import Matcher, LangchainOutputParser
from .models import KnowledgeGraph
import random


def find_longest_string(names):
    """
  Finds the longest string in a list of strings.

  Args:
    names: A list of strings.

  Returns:
    The longest string in the list.
    Returns None if the input list is empty or contains non-string elements.
  """

    if not names:
        return None  # Handle empty list case

    longest_string = ""
    for name in names:
        if not isinstance(name, str):
            print(f"Warning: Skipping non-string element: {name}")
            continue  # Skip non-string elements

        if len(name) > len(longest_string):
            longest_string = name

    return longest_string


def merge_relationships(global_relationships):
    """
    Merges relationships in a list where the start and end entities are the same,
    removing duplicates. If multiple relationships share the same start and end entities,
    a random name is chosen from the available names, and only one relationship object remains.

    Args:
        global_relationships (list): A list of relationship objects.

    Returns:
        list: A new list with merged and deduplicated relationships.
    """

    merged_relationships = []
    processed = set()  # To keep track of relationship indices already merged

    for i, ri in enumerate(global_relationships):
        if i in processed:
            continue  # Skip if already processed

        # Collect all relationships that share the same start and end entities
        duplicates = []
        duplicate_indices = []  # Store indices of duplicate relationships

        for j, rj in enumerate(global_relationships):
            if i != j and ri.startEntity == rj.startEntity and ri.endEntity == rj.endEntity:
                duplicates.append(rj)
                duplicate_indices.append(j) #Store the indexes to remove later
                processed.add(j)  # Mark relationship as processed

        # Choose a random name from all names
        all_names = [ri.name] + [r.name for r in duplicates]  # Collect all names
        ri.name = random.choice(all_names)  # Choose a random name

        merged_relationships.append(ri)  # Add the merged relationship
        processed.add(i)

    return merged_relationships

class iText2KG:
    """
    A class designed to extract knowledge from text and structure it into a knowledge graph using
    entity and relationship extraction powered by language models.
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the iText2KG with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance to be used for extracting entities and relationships from text.
        embeddings_model: The embeddings model instance to be used for creating vector representations of extracted entities.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
        self.ientities_extractor =  iEntitiesExtractor(llm_model=llm_model, 
                                                       embeddings_model=embeddings_model,
                                                       sleep_time=sleep_time) 
        
        self.irelations_extractor = iRelationsExtractor(llm_model=llm_model, 
                                                        embeddings_model=embeddings_model,
                                                        sleep_time=sleep_time)

        self.matcher = Matcher()
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)


    def build_graph(self, 
                    sections:List[str], 
                    existing_knowledge_graph:KnowledgeGraph=None, 
                    source:str=None,
                    entities_info:dict=None,
                    ent_threshold:float = 0.7, 
                    rel_threshold:float = 0.7, 
                    max_tries:int=5, 
                    max_tries_isolated_entities:int=3,
                    entity_name_weight:float=0.6,
                    entity_label_weight:float=0.4,
                    ) -> KnowledgeGraph:
        """
        Builds a knowledge graph from text by extracting entities and relationships, then integrating them into a structured graph.
        This function leverages language models to extract and merge knowledge from multiple sections of text.

        Args:
        sections (List[str]): A list of strings where each string represents a section of the document from which entities 
                              and relationships will be extracted.
        existing_knowledge_graph (KnowledgeGraph, optional): An existing knowledge graph to merge the newly extracted 
                                                             entities and relationships into. Default is None.
        ent_threshold (float, optional): The threshold for entity matching, used to merge entities from different 
                                         sections. A higher value indicates stricter matching. Default is 0.7.
        rel_threshold (float, optional): The threshold for relationship matching, used to merge relationships from 
                                         different sections. Default is 0.7.
        entity_name_weight (float): The weight of the entity name, set to 0.6, indicating its
                                     relative importance in the overall evaluation process.
        entity_label_weight (float): The weight of the entity label, set to 0.4, reflecting its
                                      secondary significance in the evaluation process.
        max_tries (int, optional): The maximum number of attempts to extract entities and relationships. Defaults to 5.
        max_tries_isolated_entities (int, optional): The maximum number of attempts to process isolated entities 
                                                     (entities without relationships). Defaults to 3.
        

        Returns:
        KnowledgeGraph: A constructed knowledge graph consisting of the merged entities and relationships extracted 
                        from the text.
        """
        
        
        print("[INFO] ------- Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0],
                                                                    entities_info=entities_info,
                                                                    entity_name_weight= entity_name_weight,
                                                                    entity_label_weight=entity_label_weight)
        print("[INFO] ------- Extracting Relations from the Document", 1)
        global_relationships = self.irelations_extractor.extract_verify_and_correct_relations(context=sections[0][-1], 
                                                                                              entities = global_entities, 
                                                                                              source=source,
                                                                                              rel_threshold=rel_threshold, 
                                                                                              max_tries=max_tries, 
                                                                                              max_tries_isolated_entities=max_tries_isolated_entities,
                                                                                              entity_name_weight= entity_name_weight,
                                                                                              entity_label_weight=entity_label_weight)
        
        for i in range(1, len(sections)):
            print("[INFO] ------- Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i],
                                                                 entities_info=entities_info,
                                                                 entity_name_weight= entity_name_weight,
                                                                 entity_label_weight=entity_label_weight)
            processed_entities, global_entities = self.matcher.process_lists(list1 = entities, list2=global_entities, threshold=ent_threshold)
            
            print("[INFO] ------- Extracting Relations from the Document", i+1)
            relationships = self.irelations_extractor.extract_verify_and_correct_relations(context= sections[i], 
                                                                                           entities=processed_entities, 
                                                                                           rel_threshold=rel_threshold,
                                                                                           max_tries=max_tries, 
                                                                                           max_tries_isolated_entities=max_tries_isolated_entities,
                                                                                           entity_name_weight= entity_name_weight,
                                                                                           entity_label_weight=entity_label_weight)
            processed_relationships, _ = self.matcher.process_lists(list1 = relationships, list2=global_relationships, threshold=rel_threshold)
            
            global_relationships.extend(processed_relationships)
            
        processed_relationships, _ = self.matcher.process_lists(list1 = global_relationships, list2=global_relationships, threshold=rel_threshold)
        global_relationships = processed_relationships
        
        if existing_knowledge_graph:
            # print(f"[INFO] ------- Matching the Document {1} Entities and Relationships with the Existing Global Entities/Relations")
            # global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(entities1=global_entities,
            #                                                      entities2=existing_knowledge_graph.entities,
            #                                                      relationships1=global_relationships,
            #                                                      relationships2=existing_knowledge_graph.relationships,
            #                                                      ent_threshold=0.95,
            #                                                      rel_threshold=1)    
            print("[INFO] ------- 合并两个 KG")
            print(f"len of global_entities {len(global_entities)}",)
            global_entities.extend(existing_knowledge_graph.entities)
            print(f"len of global_entities of merge {len(global_entities)}")
            global_entities = list(set(global_entities))
            print(f"len of global_entities of set {len(global_entities)}")
            global_relationships.extend(existing_knowledge_graph.relationships)
        
        # 检查重复的实体，关系
        duplicate_entities = {}
        for eni in global_entities:
            if len(eni.properties_info) == 1:
                # Correctly access 'unique_id' from the first element of properties_info list
                unique_ID = eni.properties_info.get('unique_id')  

                if unique_ID:  # Check if unique_ID is not None or empty
                    if unique_ID not in duplicate_entities:
                        duplicate_entities[unique_ID] = []  # Create a new list for this unique_ID
                    duplicate_entities[unique_ID].append(eni.name)  

        print("[INFO] ------- remove duplicate entities and relationships by unique_id")
        entities_output = []
        relationships_output = []
        for unique_ID, names in duplicate_entities.items():
            if len(names) > 1:
                finally_name = find_longest_string(names)
                for eni in global_entities:
                    if eni.name == finally_name:
                        unique_entity = eni
                for eni in global_entities:
                    if eni.properties_info.get('unique_id') == unique_ID:
                        eni = unique_entity
                    entities_output.append(eni)
                for r in global_relationships:
                    if r.startEntity.name in names:
                        r.startEntity = unique_entity
                    if r.endEntity.name in names:
                        r.endEntity = unique_entity
                    if r.startEntity.name == r.endEntity.name:
                        continue
                    else:
                        relationships_output.append(r)
        global_entities = entities_output
        global_relationships = merge_relationships(relationships_output)

        constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)
        constructed_kg.remove_isolated_entities()
        constructed_kg.remove_duplicates_entities()
        constructed_kg.remove_duplicates_relationships()
        return constructed_kg
    