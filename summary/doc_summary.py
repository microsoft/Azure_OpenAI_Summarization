# Document Summary Class

import json
import logging
import os
import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from langchain_core.messages import HumanMessage
from langchain.chat_models import AzureChatOpenAI

from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser,StructuredOutputParser,ResponseSchema
from pydantic import BaseModel, Field, validator
from typing import List
import datetime
from rouge import Rouge
from bert_score import score
from summary.utility import Utility

# chunk_medium = 8000
# chunk_long = 5000
# chunk_overlap = 1000
# number_clusters = 8

# chunk_medium = 1000
# chunk_long = 1500
# chunk_overlap = 500
# number_clusters = 8


chunk_medium = 8000
chunk_long = 8000
chunk_overlap = 1000
number_clusters = 8
config_file = "./config/config.json"


class DocSummary:
    def __init__(self,text):
        self.config = self.get_config()
        # self.setup_env()
        self.text = text
        self.llm,self.embeddings_model = self.get_model()
        

    def get_config(self):
        with open(config_file,"r") as f:
            config_data = json.load(f)
            return config_data

    def setup_env(self):
        vars_to_delete = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT',"OPENAI_API_BASE","OPENAI_API_KEY","OPENAI_API_TYPE","OPENAI_API_VERSION","OPENAI_ENGINE"]
        for var in vars_to_delete:
            if var in os.environ:
                del os.environ[var]
          
        
    def get_model(self):
        os.environ["AZURE_OPENAI_API_KEY"] = self.config["openai_api_key"]
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.config["openai_api_base"]  
        llm = AzureChatOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version= self.config["openai_api_version"],
            azure_deployment= self.config["openai_engine"]
        )
    
     
        embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version= self.config["openai_api_version"],
            azure_deployment= self.config["openai_embeddings_model"],

            )

        return llm,embeddings_model
        

    def summary_short(self):
        start_time = datetime.datetime.now()
        util = Utility(self.text)
        docs = util.get_chunks(chunk_size=len(self.text),chunk_overlap=0)
        prompt_file = os.path.join(".","prompts",self.config["summary_short_prompt_file"])
        prompt_template = util.get_prompt(prompt_file)
        prompt = PromptTemplate.from_template(template=prompt_template)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        # Summarize the text
        
        summary = stuff_chain.run({"input_documents":docs,"number": self.config["summary_word_count"]})
        # Return the summary
        end_time = datetime.datetime.now()
        return summary,(end_time - start_time).total_seconds()
    
    def summary_medium(self,chunks = None):
        start_time = datetime.datetime.now()
        util = Utility(self.text)
        if chunks is None:          
            docs = util.get_chunks(chunk_size=chunk_medium,chunk_overlap=chunk_overlap)
        else:
            docs = chunks
        map_prompt_file = os.path.join(".","prompts",self.config["summary_map_prompt_file"])
        reduce_prompt_file = os.path.join(".","prompts",self.config["summary_combine_prompt_file"])
        map_prompt =  util.get_prompt(map_prompt_file)

        reduce_prompt = util.get_prompt(reduce_prompt_file)

        map_prompt_template = PromptTemplate.from_template(template=map_prompt)
        reduce_prompt_template = PromptTemplate.from_template(template=reduce_prompt)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt_template)


        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="text"
        )

        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            token_max=6000)
        
        map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="text",
        # Return the results of the map steps in the output
        return_intermediate_steps=False)

        summary =  map_reduce_chain.run({"input_documents":docs,"number": self.config["summary_word_count"]})
        end_time = datetime.datetime.now()
        return summary,(end_time - start_time).total_seconds()

    def summary_long(self,clustering_type="kmeans", **kwargs):
        start_time = datetime.datetime.now()
        util = Utility(self.text)
        docs = util.get_chunks(chunk_size=chunk_medium,chunk_overlap=chunk_overlap)
        chunk_content = [x.page_content for x in docs]
        batch_size = 16
        # get embeddings for each chuck in a batch_size of 16
        vectors =[]
        for i in range(0, len(chunk_content), batch_size):
            vectors.extend(self.embeddings_model.embed_documents(chunk_content[i:i+batch_size]))

        #get cluster chunks
        allowed_types = ["kmeans", "agglomerative"]

        if clustering_type not in allowed_types:
            raise ValueError(f"Invalid clustering type. Allowed types are: {', '.join(allowed_types)}")

        allowed_kwargs = {
            "kmeans": {"n_clusters", "n_init"},
            "agglomerative": {"n_clusters"}
        }

        invalid_kwargs = set(kwargs) - allowed_kwargs.get(clustering_type, set())
        if invalid_kwargs:
            raise ValueError(f"Invalid keyword argument(s) for {clustering_type}: {', '.join(invalid_kwargs)}")
        
        selected_indices = []
        selected_chunks = []
        number_clusters = kwargs.get("n_clusters")
        if number_clusters is None:
            number_clusters = self.config["num_clusters"]
        if clustering_type == "kmeans":
            model = KMeans(n_init=10, n_clusters=number_clusters, random_state=0).fit(vectors)
            closest_indices = []
            # find the closest vector to each cluster center
            for i in range(number_clusters):
                distances = np.linalg.norm(vectors - model.cluster_centers_[i], axis=1)
                closest_index = np.argmin(distances)
                closest_indices.append(closest_index)
            selected_indices = sorted(closest_indices)

        elif clustering_type == "agglomerative":
            model = AgglomerativeClustering(n_clusters=number_clusters).fit(vectors)
            cluster_labels = model.labels_
            for cluster_id in np.unique(cluster_labels):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_embeddings = [vectors[idx] for idx in cluster_indices]
                medoid_index = pairwise_distances_argmin_min(cluster_embeddings, [np.mean(cluster_embeddings, axis=0)])[0][0]
                selected_indices.append(cluster_indices[medoid_index])
        else:
            raise ValueError("Invalid clustering type.")
        
        # selected_indices = sorted(closest_indices)
        selected_chunks = [docs[idx] for idx in selected_indices]

        #get summary
        summary,response_time = self.summary_medium(selected_chunks)
        end_time = datetime.datetime.now()
        return summary,(end_time - start_time).total_seconds()
    
class summaryMetrics:

    def __init__(self,summary_text,reference_text):
        self.rouge = Rouge()
        self.hypothesis = summary_text
        self.reference = reference_text

    def get_rouge_score(self):
        scores = self.rouge.get_scores(self.hypothesis, self.reference)
        rouge_1 = scores[0]['rouge-1']
        rouge_2 = scores[0]['rouge-2']
        rouge_l = scores[0]['rouge-l']
        rouge_1_p = rouge_1['p']
        rouge_1_r = rouge_1['r']
        rouge_1_f = rouge_1['f']
        rouge_2_p = rouge_2['p']
        rouge_2_r = rouge_2['r']
        rouge_2_f = rouge_2['f']
        rouge_l_p= rouge_l['p']
        rouge_l_r = rouge_l['r']
        rouge_l_f = rouge_l['f']
        return rouge_1_p,rouge_1_r,rouge_1_f,rouge_2_p,rouge_2_r,rouge_2_f,rouge_l_p,rouge_l_r,rouge_l_f
    
    def get_bert_score(self):
        p,r,f = score([self.hypothesis],[self.reference],lang="en")
        return p.item(),r.item(),f.item()
        

class NLGMetrics:
    '''
        Caluculate NLG metrics : Consistency, coherence, fluency & relevance based on GPT models.
    '''

    def __init__(self,document,summary):
        self.src_document = document
        self.doc_summary = summary
        self.config = self.get_config()
        os.environ["AZURE_OPENAI_API_KEY"] = self.config["openai_api_key"]
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.config["openai_api_base"]  
        self.llm = AzureChatOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version= self.config["openai_api_version"],
            azure_deployment= self.config["openai_engine"]
        )


    def get_config(self):
        with open(config_file,"r") as f:
            config_data = json.load(f)
            return config_data
        
    def metric_scores(self,metric,prompt_file):
        if metric == "fluency":
            util = Utility(self.doc_summary)
            docs = util.get_chunks(chunk_size=len(self.doc_summary),chunk_overlap=0)
            prompt_template = util.get_prompt(prompt_file)
            prompt = PromptTemplate.from_template(template=prompt_template)
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            output = stuff_chain.run({"input_documents":docs})
        else:
            util = Utility(self.src_document)
            docs = util.get_chunks(chunk_size=len(self.src_document),chunk_overlap=0)
            prompt_template = util.get_prompt(prompt_file)
            prompt = PromptTemplate.from_template(template=prompt_template)
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            output = stuff_chain.run({"input_documents":docs,"summary":self.doc_summary})
        return output
    
    def get_nlg_metrics(self):
        scores_dict = {}
        metric_prompts = {"coherence": self.config["metrics_coherence_prompt_file"],
                       "consistency": self.config["metrics_consistency_prompt_file"],
                       "fluency": self.config["metrics_fluency_prompt_file"],
                       "relevance": self.config["metrics_relevance_prompt_file"]}
     
        for metric in metric_prompts:
            prompt_file = os.path.join(".","prompts",metric_prompts[metric])
            score = self.metric_scores(metric,prompt_file)
            scores_dict[metric] = score
        return scores_dict
            
