from fastapi import FastAPI
from fastapi import Form
from dotenv import dotenv_values
from pymongo import MongoClient
import urllib
import datetime
from routes import router as test_router

'''
    start up, shut down and API routers management
'''
config = dotenv_values(".env")

app = FastAPI()




@app.on_event("startup")
def startup_db_client():
    #info parsing
    username = urllib.parse.quote_plus(config['USER_NAME'])
    password = urllib.parse.quote_plus(config['PASSWORD'])
    client = f'mongodb+srv://{username}:{password}@cluster0.1wvuto3.mongodb.net/?retryWrites=true&w=majority'
    #connect to the cluster and database
    app.mongodb_client = MongoClient(client)
    print("Connected to the MongoDB database!")
    app.database = app.mongodb_client[config["DB_NAME"]]
    print("???", app.database["testcollection"].find(limit=100))

    

    

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()

app.include_router(test_router, prefix="/student")
