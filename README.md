# Langchain with RAG using vector database data

This is an exploratory and demonstration project that uses Langchain and Retrieval-Augmented Generation (RAG) to leverage data in a vector database to improve Large Language Model (LLM) responses.

## Built With

* Ollama locally-hosted gpt-oss:20b LLM
* Langchain with [VectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore) and [VectorStoreRetreiver](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html#langchain_core.vectorstores.base.VectorStoreRetriever)
* NFL player data from https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2024/WR_season.json

## Overview

RAG is one of several techniques that improve LLM performance by providing additional data to the model.  This additional data may be something that the LLM was not originally trained on and is unable to consider in its response.  Similarly, the additional data may frequently change and providing it to the LLM allows the model to provide more accurate responses based on current information.  An advantage of RAG compared to other techniques is that the additional data can be provided along with the LLM prompt and can therefore change along with the prompt.  However this is also a disadvantage in that the processing and other requirements to implement and support RAG must be incurred with each prompt.

The goal of this project is to ask the LLM a question related to NFL player performance (discussed below) and see if the response improves if supporting data is provided to the LLM via RAG, as follows:

* The question is provided to the LLM with no supporting data (intended to validate that the LLM does not have the data and cannot answer the question properly)
* The question is provided to the same LLM but with RAG and a vector database that contains the data needed to answer the question

Overall the LLM without RAG data was unable to answer the questions, but was able to with RAG data.

## Discussion

### Vector Database

The source data used is available [here](https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2024/WR_season.json).  This data is for wide receivers the entire 2024 NFL season.  Each row represents a single player with statistics about their performance during the 2024 season.

To prepare the vector database, the ```vector.py``` code was used to load the source data json into a dataframe, create embeddings using the ```OllamaEmbeddings``` class and the ```mxbai-embed-large``` model, and finally create the Chroma vector database.

The only modification to the source data was the addition of Season column with the value of 2024 for all data rows, as the source data did not specify the season:

```df["Season"] = "2024"```

The relevant columns of data for the project question are as follows:

* PlayerName
* ReceivingYDS

It is important to note that the LLM needs to interpret several aspects from the question, e.g. relating a player to the PlayerName column and relating the number of receiving yards they made to the ReceivingYDS column.  There is no contextual, prefix, or other data provided to indicate to the LLM what each column means.  It is also important to note that there are several similar columns that could be misinterpreted, e.g. ReceivingRec, ReceivingYDS, TouchReceptions, and others.

### Question

The question asked for both scenarios was:

***how many receiving yards did khalil shakir have in 2024?***

Providing the question to the LLM without the RAG vector database data, the response was as follows:

```
I don't have any information on a player named Khalil Shakir from 2024, as my knowledge cutoff is December 2023. I'd be happy to try and help you find the information you're looking for, but I wouldn't be able to provide an accurate answer based on data that hasn't been released yet.

However, I can tell you that there is a wide receiver named Khalil Shakir who currently plays for the Buffalo Bills in the NFL. If you'd like, I can try and find his stats from previous seasons (2022 or 2023) instead?
```

This validates that the LLM was not trained on, and otherwise does not have access to the recent NFL player data.

Providing the same question to the same LLM with the RAG vector database data, the response was:

```
Khalil Shakir had 821 receiving yards in 2024.
```

This is the correct answer, and is verified by manual inspection of the source data.

This question is fairly straightforward for the LLM with RAG vector database data to answer, as it requires simply looking up the data and determining the max value.

## Conclusion and future work

This was an interesting project that I think clearly demonstrates the capability of RAG to add new data to an LLM.

* explore more nuanced prompts that require the LLM to interpret and analyze the data more extensively
* evaluate the performance and capability of different models and model sizes as well as larger data sets (multiple seasons, play-by-play data for individual games, etc.)
