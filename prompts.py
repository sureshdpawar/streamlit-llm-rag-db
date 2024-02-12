import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone


FS_TEMPLATE = """ You are an expert SQL developer querying about financials statements. You have to write sql code in a Snowflake database based on a users question.
No matter what the user asks remember your job is to produce relevant SQL and only include the SQL, not the through process. So if a user asks to display something, you still should just produce SQL.
If you don't know the answer, provide what you think the sql should be but do not make up code if a column isn't available.

As an example, a user will ask "Display the last 5 years of net income for Johnson and Johnson?" The SQL to generate this would be:

select year, net_income
from financials.prod.income_statement_annual
where ticker = 'JNJ'
order by year desc
limit 5;

Questions about income statement fields should query financials.prod.income_statement_annual
Questions about balance sheet fields (assets, liabilities, etc.) should query  financials.prod.balance_sheet_annual
Questions about cash flow fields (operating cash, investing activities, etc.) should query financials.prod.cash_flow_statement_annual

The financial figure column names include underscores _, so if a user asks for free cash flow, make sure this is converted to FREE_CASH_FLOW. 
Some figures may have slightly different terminology, so find the best match to the question. For instance, if the user asks about Sales and General expenses, look for something like SELLING_AND_GENERAL_AND_ADMINISTRATIVE_EXPENSES

If the user asks about multiple figures from different financial statements, create join logic that uses the ticker and year columns. Don't use SQL terms for the table alias though. Just use a, b, c, etc.
The user may use a company name so convert that to a ticker.

Question: {question}
Context: {context}

SQL: ```sql ``` \n
 
"""
FS_PROMPT = PromptTemplate(input_variables=["question", "context"], template=FS_TEMPLATE, )

LETTER_TEMPLATE = """ You are tasked with retreiving questions regarding Warren Buffett from his shareholder letters.
Provide an answer based on this retreival, and if you can't find anything relevant, just say "I'm sorry, I couldn't find that."
{context}
Question: {question}
Anwer:
 
"""
LETTER_PROMPT = PromptTemplate(input_variables=["question", "context"], template=LETTER_TEMPLATE, )

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1000, 
    openai_api_key=st.secrets["openai_key"]
)


def get_faiss():
    " get the loaded FAISS embeddings"
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    return FAISS.load_local("faiss_index", embeddings)


def get_pinecone():
    " get the pinecone embeddings"
    pinecone.init(
        api_key=st.secrets['pinecone_key'], 
        environment=st.secrets['pinecone_env'] 
        )
    
    index_name = "buffett"
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    return Pinecone.from_existing_index(index_name,embeddings)


def fs_chain(question):
    """
    returns a question answer chain for faiss vectordb
    """

    docsearch = get_faiss()
    qa_chain = RetrievalQA.from_chain_type(llm, 
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs={"prompt": FS_PROMPT})
    return qa_chain({"query": question})


import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def query_pinecone_with_text(query_text, index):
    # Convert the query text to an embedding
    query_embedding = get_embeddings_for_text(query_text)

    # Perform the query in Pinecone
    query_results = index.query(query_embedding, top_k=5)

    # Here, you might want to print, return, or further process the query results
    return query_results

def get_embeddings_for_text(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def letter_chainV(question):
    # Initialize OpenAI with your API key
    openai.api_key = 'sk-'

    # Initialize Pinecone with your API key and environment
    pinecone.init(api_key='', environment='gcp-starter')

    # Connect to your Pinecone index
    index_name = "buffet"  # Make sure this matches the name used during your document upload process
    index = pinecone.Index(index_name)

    results = query_pinecone_with_text(question, index)
    print(results)
    return results

def letter_chain(question):
    logging.info("Starting letter_chain function")

    try:
        docsearch = get_pinecone()
        logging.info(f"docsearch initialized: {docsearch}")

        retriever = docsearch.as_retriever(
            search_kwargs={"k": 3}
        )
        logging.info(f"Retriever created: {retriever}")

        #qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="map_rerank",  return_source_documents=True)
        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

        logging.info("QA chain initialized")

        response = qa_chain({"query": question})
        logging.info(f"QA chain response: {response}")
        return response
    except Exception as e:
        pinecone.init(api_key="", environment="gcp-starter")
        index = pinecone.Index("buffet")
        index.query(question)
        logging.error(f"An error occurred: {e}")
        #raise



def letter_chainx(question):
    """returns a question answer chain for pinecone vectordb"""
    
    docsearch = get_pinecone()
    print("docsearch", docsearch)
    retreiver = docsearch.as_retriever(#
        #search_type="similarity", #"similarity", "mmr"
        search_kwargs={"k":3}
    )
    print("retreiver", retreiver)
    qa_chain = RetrievalQA.from_chain_type(llm, 
                                            retriever=retreiver,
                                           chain_type="stuff", #"stuff", "map_reduce","refine", "map_rerank"
                                           return_source_documents=True,
                                           #hain_type_kwargs={"prompt": LETTER_PROMPT}
                                          )

    qa_chain({"query": question})
    print("qa_chain", qa_chain)
    return qa_chain({"query": question})


def pinecone_search():
    # Replace 'your_api_key' with your actual Pinecone API key
    pinecone.init(api_key='', environment='gcp-starter')

    # Replace 'your_index_name' with the name of your Pinecone index
    index_name = 'buffet'

    # Check if the index exists and create a connection to the index
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"The index '{index_name}' does not exist. Please create it in the Pinecone dashboard.")

    # Connect to your Pinecone index
    index = pinecone.Index(index_name=index_name)

    return index


import pinecone


def initialize_pinecone():
    # Initialize Pinecone
    pinecone.init(api_key='', environment='gcp-starter')
    index_name = "buffet"
    # Check if the index exists, if not, create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

    # Connect to the index
    index = pinecone.Index(index_name)
    print("initialize_pinecone done")
    return index

def query_pinecone(query_text, index):
    try:
        print("1")
        # Convert the query text to an embedding using OpenAI's embedding model
        embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",  # Consider matching the embedding model with your document processing
            input=query_text
        )
        print("2")
        query_embedding = embedding_response['data'][0]['embedding']

        # Query the Pinecone index with the query embedding
        query_result = index.query(queries=[query_embedding], top_k=5)
        print("query result", query_result)
        return query_result
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def letter_qa(query):
    # Initialize Pinecone
    index = initialize_pinecone()
    query_pinecone(query, index)


def letter_qay(query):
    # Configuration parameters
    temperature = 0.7  # Adjust as needed
    model_name = 'gpt-3.5-turbo'  # Specify the model you're using
    openai_api_key = st.secrets["openai_key"]  # Ensure your Streamlit secrets are correctly set up

    # Query to execute
    try :
        query = query
        print("query**", query)
        # Create the PDF QA system
        pdf_qa = ChatVectorDBChain.from_llm(
            OpenAI(temperature=temperature, model_name=model_name, openai_api_key=openai_api_key),
            pinecone_search(),
            return_source_documents=True
        )
        print("pdf_qa**", pdf_qa)
        # Execute the query
        response = pdf_qa({"question": query, "chat_history": ""})

        # Use the response
        print(response)
        return response
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def letter_qax(query, temperature=.1,model_name="gpt-3.5-turbo"):
    """
    this method was deprecated but seems to be more efficient from a token perspective
    """
    pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=temperature, model_name=model_name, openai_api_key=st.secrets["openai_key"]),
                    pinecone_search(), return_source_documents=True)
    return pdf_qa({"question": query, "chat_history": ""})

