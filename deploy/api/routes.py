import os

from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import Student, QuestionAnswer

working_dir = os.environ['dirs']
database_dir = f"{working_dir}/data/database"


# router = APIRouter()
# @router.post("/", response_description="Add new student", status_code=status.HTTP_201_CREATED, response_model=Student)
# def post_item(request: Request, student: Student = Body(...)):
#     student = jsonable_encoder(student)
#     new_student = request.app.database['testcollection'].insert_one(student)
#     created_student = request.app.database["testcollection"].find_one(
#         {"_id": new_student.inserted_id}
#     )
    

#     return created_student

# @router.get("/", response_description="List all students", response_model=List[Student])
# def list_books(request: Request):
#     # books = list(request.app.database["testcollection"].find(limit=100))
#     # return books
#     return list(request.app.database["testcollection"].find({"_id": "20200083"}))
#tourxQA
qa_router = APIRouter( )



@qa_router.post("/", response_description="Get Intent and Slot", response_model=QuestionAnswer)
def get_intent(request: Request, input: QuestionAnswer = Body(...)):
    '''
    given question, run the modules and save results in log file for using
    '''
    
    question = input['question']
    #IDSF module
    #if good intent and slots -> automation
    #else -> QA module

    



    

@qa_router.get("/", response_description="Get Answer", response_model=QuestionAnswer)
def get_answer(request: Request):
    pass
