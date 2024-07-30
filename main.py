import numpy as np
import streamlit as st
import pandas as pd
import getpass
import getpass
import os
from langchain_community.utilities import SQLDatabase

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

import re

import custom_sql_agent

lang_smith= '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'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = lang_smith

os.environ["OPENAI_API_KEY"] = 'sk-my-test-service-key-JIfuqb1hH8XcVxBCV9rvT3BlbkFJekdveAFWIG7tGOCDSX27'

db = SQLDatabase.from_uri("postgresql://do_invoice:wohhup2021@wh-idd-test-dev.c9lyw52w9grj.ap-southeast-1.rds.amazonaws.com:5432/ai_pgvector")

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def get_example_by_table(table):
    return ""

def get_table_scheme_by_table(table):
    
    if table =="PDIP Invoice":
        return """Only use this following table:
            CREATE TABLE invoice_raw (
            "S/N" BIGINT, 
            "Project code" TEXT, 
            "BP code" TEXT, 
            "BP name" TEXT, 
            "Group code" TEXT, 
            "Status" TEXT, 
            "Cost Code" TEXT, 
            "Cost Code ref" TEXT, 
            "Product" TEXT, 
            "Invoice No" TEXT, 
            "Invoice Amt" DOUBLE PRECISION, 
            "GST" DOUBLE PRECISION, 
            "Invoice Amt with GST" DOUBLE PRECISION, 
            "Invoice Date" TIMESTAMP WITHOUT TIME ZONE, 
            "Posting Date" TIMESTAMP WITHOUT TIME ZONE, 
            "Due Date" TIMESTAMP WITHOUT TIME ZONE, 
            "SAP Creation Date" TIMESTAMP WITHOUT TIME ZONE, 
            "Date Paid" TIMESTAMP WITHOUT TIME ZONE, 
            "Paid" TEXT, 
            "WD Month" TEXT, 
            "Credit Terms days" BIGINT
        )

        /*
        3 rows from invoice_raw table:
        S/N	Project code	BP code	BP name	Group code	Status	Cost Code	Cost Code ref	Product	Invoice No	Invoice Amt	GST	Invoice Amt 
        incl GST	Invoice Date	Posting Date	Due Date	SAP Creation Date	Date Paid	Paid	WD Month	Credit Terms days
        1	FJX	RP 2001	TANGLIN CORPORATION PTE LTD	DOMESTIC SUBCONTRACTORS	PAID	01E01	01E01	Mixed Keruing Sawn Timber 2"x1"x12'	25007522	594.0	47.52	641.52	2023-04-30 00:00:00	2023-07-01 00:00:00	2023-06-29 00:00:00	2023-07-20 00:00:00	2023-07-20 00:00:00	Yes	04-2023	60
        2	FJX	RP 2001	TANGLIN CORPORATION PTE LTD	DOMESTIC SUBCONTRACTORS	PAID	01G02	01G02	High Speed Diesel ( Drum Purchase )	25007549	8739.335140000001	699.1468112000001	9438.4819512	2023-04-30 00:00:00	2023-07-01 00:00:00	2023-06-29 00:00:00	2023-07-28 00:00:00	2023-07-28 00:00:00	Yes	04-2023	60
        3	FJX	RP 2001	TANGLIN CORPORATION PTE LTD	DOMESTIC SUBCONTRACTORS	PAID	01G02	01G02	High Speed Diesel ( Drum Purchase )	25007741	21139.272	1691.1417600000004	22830.41376	2023-05-31 00:00:00	2023-07-01 00:00:00	2023-07-30 00:00:00	2023-08-03 00:00:00	2023-08-03 00:00:00	Yes	05-2023	60
        */
        """
    
    return ""

def get_agent_by_table(table):
    
    #get samples
    
    examples = [
        {
            "input": "Find all companies works in project BFC?",
            "query": """SELECT id,"Company Name" FROM procurement WHERE "Project Code" = 'BFC'""",
        }
    ]
    
    #get table scheme
    table_scheme = get_table_scheme_by_table(table)
    
    
    #get embedding instruction
    embedding_instruction = """You can use an extra extension which allows you to run semantic similarity using <-> operator on tables containing columns named "Description of Works_embeddings".
    <-> operator can ONLY be used on embeddings columns.
    The embeddings value for a given row typically represents the semantic meaning of that row.
    The vector represents an embedding representation of the question, given below. 
    Do NOT fill in the vector values directly, but rather specify a `[search_word]` placeholder, which should contain the word that would be embedded for filtering.
    For example, if the user asks for work description about 'structure work' the query could be:
    'SELECT "[whatever_table_name]"."Company Name" FROM "[whatever_table_name]" ORDER BY "Description of Works_embeddings" <-> '[structure work]' LIMIT 5'"""


    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    system_prefix = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.

    {embedding_instruction}

    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.

    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use date('now') function to get the current date, if the question involves "today".
    Only use this following table:           
    {table_scheme}
    If the question does not seem related to the database, just return "I don't know" as the answer.

    Here are some examples of user inputs and their corresponding SQL queries:
    {examples}
    """
    


    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    
    full_prompt = full_prompt.partial(table_scheme=table_scheme)
    full_prompt = full_prompt.partial(examples=examples)
    full_prompt = full_prompt.partial(embedding_instruction=embedding_instruction)
    

    agent = custom_sql_agent.create_sql_agent(
        llm=llm,
        db=db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
        top_k=100
    )
    
    return agent
    

def main():
    st.title("WHGPT-SQL")

    options = ["PDIP Invoice","Procurement", "Cost progress report"]
    selected_option = st.selectbox("Please select database", options)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        agent=get_agent_by_table(selected_option)
        
        res=agent.invoke({"input": prompt})
        print(res)
        response = f"{res.get('output')}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
