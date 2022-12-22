import os

from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.responses import FileResponse
from typing import List
from glob import glob
from PIL import Image


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
            for k in sys_dict[key].keys():
                reset_value(k, sys_dict[key])

    return sys_dict

def status_dict(sys_dict):
    '''
    check the status of the system dict
    '''
    lack_items = []
    status = True
    for key, value in sys_dict.items():
        if type(value) == dict:
            all_blank = True
            for k, value in sys_dict[key].items():
                if value != '':
                    all_blank = False

            if all_blank:
                lack_items.extend(list(sys_dict[key].keys()))
                status = False
            else:
                for k, value in sys_dict[key].items():
                    if value == '':
                        sys_dict[key][k] = 0
                    

        elif value in ['']:
            lack_items.append(key)
            status = False

    return lack_items, sys_dict, status

    

    
            


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



@qa_router.post("/", response_description="Get Answer")
def get_response(request: Request, input: QuestionAnswer = Body(...)):
    '''
    given question, run the modules and save results in log file for using
    '''
    flight_dict = {'from_city': '', 'to_city': '', 'num_class': '', 'pass_dict': {'adult': '', 'child': '', 'infant': ''}, 'num_person': ''}
    # question = input['question']
    #IDSF module
    # os.system(f'python {source_dir}/predict.py --text_question {question}')
    with open(log_dir, 'r') as f:
        output = f.readlines() #result
        #if intent prob is reliable:
        intent_prob = float(output[0].split('->')[0].split('<')[1].split('>')[0])
        slots = [tuple(line.strip().split(':')) for line in output[1:] if len(line.strip()) > 0]
        print("SLOTS: ", slots)
        # print([line for line in output[1:]])
        slot_dict = dict([item for item in slots])
        if intent_prob > 0.5:
            intent = output[0].split('->')[2].strip()

            
        #flight case:
        if intent == '<flight>':
            #reset the dict
            flight_dict = reset_dict(flight_dict)
            #parse the slots
            for key, value in slot_dict.items():
                if 'fromloc.city_name' in key:
                    value = ' '.join([i for i in value.split('_')])
                    flight_dict['from_city'] += f' {value}'
                elif 'toloc.city_name' in key:
                    value = ' '.join([i for i in value.split('_')])
                    flight_dict['to_city'] += f' {value}'
                elif 'class_type' in key:
                    flight_dict['num_class'] += f' {value}'
                elif 'num_person' in key:
                    flight_dict['num_person'] += f' {value}'



                

            
        
        elif 'class_type' in list(slot_dict.keys()):
            flight_dict['num_class'] += f' {value}'

        elif 'num_person' in list(slot_dict.keys()):
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
        #"x nguoi lon y tre em z so sinh"
        flight_dict['pass_dict']['adult'] = int(flight_dict['num_person'].split("người_lớn")[0].strip())


        #Haven't own 'nuff information
        lack_items, flight_dict, web_service = status_dict(flight_dict)
        if len(lack_items) == 0:
            pass
        else:
            web_service = True

       

        
                    


    #checking if           

    #if good intent and slots -> automation
    if web_service == True:
        #demo the ticket query
        flight_dict.pop('num_person'    )
        queries = [f'--{key} "{value}"' for key, value in flight_dict.items() if type(value) != dict]
        queries.extend([f'--{key} {value}' for key, value in flight_dict['pass_dict'].items()])
        web_query = ' '.join(queries)
        # return web_query
        # return queries
        os.system(f'python {automation_dir}/web_service.py ' + web_query) #parse the arguments
        page_img= Image.open(f"{automation_dir}/traveloka.png")
        request_img = Image.open(f"{automation_dir}/request.png")
        result_images = []
        for file in glob(f"{automation_dir}/results/*"):
            result_images.append(Image.open(file))
        
        return page_img, request_img


    #else -> QA module

    

    



    

