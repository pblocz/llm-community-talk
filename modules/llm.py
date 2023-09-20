from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain, Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import itertools as it
from langchain.callbacks import get_openai_callback


map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])


reduce_prompt = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
reduce_prompt_template = PromptTemplate(template=reduce_prompt, input_variables=["text"])


def summarization_chain(verbose=False):
    llm = OpenAI(temperature=0, max_tokens=512)

    map_chain = LLMChain(llm=llm, prompt=map_prompt_template, verbose=verbose)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt_template, verbose=verbose)

    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="text",
        verbose=verbose,
    )

    mapreduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        combine_document_chain=combine_document_chain,
        document_variable_name=combine_document_chain.document_variable_name,
        verbose=verbose
    )

    return mapreduce_chain


def split_text(text, separators=["\n\n", "\n", " "], chunk_size=3000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(separators=separators + [""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    return docs


def split_in_chuncks(docs, number_of_chunks):

    chunk_size = (len(docs) // number_of_chunks)

    groups = [[group for _, group in enumerated_group] for _, enumerated_group in it.groupby(enumerate(docs), key=lambda e: e[0] // chunk_size)]
    return groups


def summarize_text(text, number_of_chunks, include_costs=False):
    with get_openai_callback() as cb:
        chain = summarization_chain()
        docs = split_text(text)
        groups = split_in_chuncks(docs, number_of_chunks)
        summaries = [chain.run(group) for group in groups]
        joined_summary = "\n\n".join(summaries)

        if include_costs:
            return joined_summary, cb
        return joined_summary