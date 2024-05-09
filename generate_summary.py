import logging
import os
import json
import numpy as np
import streamlit as st
from summary.doc_summary import DocSummary

st.title('Summarization with LLM')



def get_summary(text):
    docSum = DocSummary(text)
    num_tokens = docSum.llm.get_num_tokens(text)
    if num_tokens < int(docSum.config["threshold_short_medium"]):
        return docSum.summary_short()
    elif num_tokens < int(docSum.config["threshold_medium_long"]):
        return docSum.summary_medium()
    elif num_tokens < int(docSum.config["threshold_length_limit"]):
        return docSum.summary_long()
    else:
        return "document too long to process",""




# results = get_summary(text)
# print(results)

with st.form('summary_form'):
    text = st.text_area('Enter text to summarize:', height=300)
    submitted = st.form_submit_button('Submit')
    if submitted:
        if text == "":
            st.warning('Please enter text to summarize!', icon='⚠')
        else:
            results = get_summary(text)
            print(results)
            if results:
                # st.write(results["summary_aoai"])
                st.divider()
                st.caption("Summary")
                st.write(results[0])  






