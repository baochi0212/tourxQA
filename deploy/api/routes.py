import os

from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import Student, QuestionAnswer

working_dir = os.environ['dir']
database_dir = f"{working_dir}/data/database"
source_dir = f"{working_dir}/source"
log_dir = './log.txt'
automation_dir = f"{working_dir}/crawler/automatic_post"

def reset_dict(sys_dict):
    '''
    reset the system storage for the new request
    '''
    def reset_value(key, dict):
        if type(dict[key]) == str:
            dict[key] = ''
        elif type(dict[key]) == int:
            dict[key] = 0
    for key in sys_dict.keys():
        reset_value(key, sys_dict)
        if type(sys_dict[key]) == dict:
            for key in sys_dict[key].keys():
                reset_value(key, sys_dict[key])

    return sys_dict

def status_dict(sys_dict):
    '''
    check the status of the system dict
    '''
    lack_items = []
    for key, value in sys_dict.items():
        if type(value) == dict:
            for k, value in sys_dict.items():
                if value in ['', 0]:
                    lack_items.append(key)
                    

        elif value in ['', 0]:
            lack_items.append(key)

    return lack_items

    

    
            


router = APIRouter()
@router.post("/", response_description="Add new student", status_code=status.HTTP_201_CREATED, response_model=Student)
def post_item(request: Request, student: Student = Body(...)):
    student = jsonable_encoder(student)
    new_student = request.app.database['testcollection'].insert_one(student)
    created_student = request.app.database["testcollection"].find_one(
        {"_id": new_student.inserted_id}
    )
    

    return created_student

@router.get("/", response_description="List all students", response_model=List[Student])
def list_books(request: Request):
    # books = list(request.app.database["testcollection"].find(limit=100))
    # return books
    return list(request.app.database["testcollection"].find({"_id": "20200083"}))
#tourxQA
qa_router = APIRouter( )
flight_dict = {'from_city': '', 'to_city': '', 'num_class': '', 'pass_dict': {'adult': 0, 'child': 0, 'infant': 0 }, 'num_person': ''}


@qa_router.post("/", response_description="Get Answer", response_model=None)
def get_response(request: Request, input: QuestionAnswer = Body(...)):
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
        slots = [line.split(':') for line in f.readlines()[1:]]
        slot_dict = dict([(key, value) for key, value in slots])
        if intent_prob > 0.5:
            intent = output.split('->')[2].strip()

            
        #flight case:
        if intent == '<flight>':
            #reset the dict
            flight_dict = reset_dict(flight_dict)
            #parse the slots
            for key, value in slot_dict.items():
                if 'fromloc.city_name' in key:
                    flight_dict['from_city'] += f' {value}'
                elif 'toloc.city_name' in key:
                    flight_dict['to_city'] += f' {value}'
                elif 'class_type' in key:
                    flight_dict['num_class'] += f' {value}'
                elif 'num_person' in key:
                    flight_dict['num_person'] += f' {value}'

            #LACK items
            lack_items = status_dict(flight_dict)
            if len(lack_items) > 0:
                pass

            
        
        elif 'class_type' in list(slot_dict.keys())[0]:
            flight_dict['num_class'] += f' {value}'

        elif 'num_person' in list(slot_dict.keys())[0]:
            # passenger:
            flight_dict['num_person'] += f' {value}'
        #parse string values to numbers
        #class_type:
        if 'thương_gia' in flight_dict['num_class']:
            flight_dict['num_class'] = 2
        elif 'hạng_nhất' in flight_dict['num_class']:
            flight_dict['num_class'] = 1
        elif 'hạng_nhì' in flight_dict['num_class']:
            flight_dict['num_class'] = 0
        #num_person:

        
                    


    #checking fi                 

    #if good intent and slots -> automation
    if intent == 'flight':
        os.system(f'python {automation_dir}/web_service.py') #parse the arguments


    #else -> QA module

    

    



    

