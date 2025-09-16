from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:

    # df = pd.read_csv("realistic_restaurant_reviews.csv")
    # df = pd.read_csv("WR_season.csv")
    df = pd.read_json("WR_season.json")
    df["Season"] = "2024" # source data does not specify the season

    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            # page_content=row["Title"] + " " + row["Review"],
            # metadata={"rating": row["Rating"], "date": row["Date"]},
            page_content = " ".join(str(row[col]) for col in [
                "Season", "PlayerName","Pos","Team","PassingYDS","PassingTD","PassingInt",
                "RushingYDS","RushingTD","ReceivingRec","ReceivingYDS","ReceivingTD","RetTD",
                "FumTD","2PT","Fum","FanPtsAgainst-pts","TouchCarries","TouchReceptions",
                "Touches","TargetsReceptions","Targets","ReceptionPercentage","RzTarget",
                "RzTouch","RzG2G","TotalPoints"
            ]),
            metadata={"rank": row["Rank"], "player_id": row["PlayerId"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="WR_season_2024",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)