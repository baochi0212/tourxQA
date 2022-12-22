import os

from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import Student, QuestionAnswer

working_dir = os.environ['dirs']
database_dir = f"{working_dir}/data/database"
source_dir = f"{working_dir}/source"
log_dir = './log.txt'
automation_dir = f"{working_dir}/crawler/automatic_post"


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
    os.system(f'python {source_dir}/predict.py --text_question {question}')
    with open(log_dir, 'r') as f:
        output = f.readlines()[0] #result
        #if intent prob is reliable:
        intent_prob = float(output.split('->')[0].split('<')[1].split('>')[0])
        if intent_prob > 0.7:
            intent = output.split('->')[2].strip()
            slots = [line.split(':') for line in f.readlines()[1:]]
            slot_dict = dict([(key, value) for key, value in slots])
            
            #flight case:
            if intent == '<flight>':
                #parse the slots
                parse_dict = {'from_city': '', 'to_city': '', 'num_class': 0, 'pass_dict': {'adult': 0, 'child': 0, 'infant': 0 }}
                for key, value in slot_dict.items():
                    if 'fromloc.city_name' in key:
                        parse_dict['from_city'] += f' {value}'
                    elif 'toloc.city_name' in key:
                        parse_dict['to_city'] += f' {value}'
                
    #if good intent and slots -> automation
    if intent == 'flight':
        os.system(f'python {automation_dir}/web_service.py') #parse the arguments
    #else -> QA module

    



    

@qa_router.get("/", response_description="Get Answer", response_model=QuestionAnswer)
def get_answer(request: Request):
    pass
