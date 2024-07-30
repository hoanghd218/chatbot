import getpass
import getpass
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
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

lang_smith= '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'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = lang_smith

os.environ["OPENAI_API_KEY"] = 'sk-my-test-service-key-JIfuqb1hH8XcVxBCV9rvT3BlbkFJekdveAFWIG7tGOCDSX27'

db = SQLDatabase.from_uri("postgresql://do_invoice:wohhup2021@wh-idd-test-dev.c9lyw52w9grj.ap-southeast-1.rds.amazonaws.com:5432/ai_pgvector")
print(db.dialect)
print(db.get_usable_table_names())
# db.run('SELECT * FROM "wa_employee" LIMIT 10;')

embeddings_model = OpenAIEmbeddings()

descriptions = db.run('SELECT "Description of Works" FROM "procurement"')
descriptions = [s[0] for s in eval(descriptions)]
descriptions = [item for item in descriptions if item is not None]


descriptions_embeddings = embeddings_model.embed_documents(descriptions)
len(descriptions_embeddings)

from tqdm import tqdm

for i in tqdm(range(len(descriptions_embeddings))):
   
    i_description= descriptions[i]
    i_description=i_description.replace("'","''")
    embedding = descriptions_embeddings[i]
    sql_command = (
        f"""UPDATE "procurement" SET "Description of Works_embeddings" = ARRAY{embedding} WHERE "Description of Works" = '{i_description}'"""
    )
    db.run(sql_command)
    
# def g(s):
#     embeded_query = embeddings_model.embed_query(s)
#     query = (
#         'SELECT description FROM test WHERE embedding IS NOT NULL ORDER BY embedding <-> '
#         + f"'{embeded_query}' LIMIT 5"
#     )

#     b=db.run(query)

#     print(s +  ":  " +b)
#     aa=1

# g("list all structure works")
# g("list all earth works")
