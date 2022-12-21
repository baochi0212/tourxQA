from pydantic import BaseModel, Field

class Student(BaseModel):
    id: str = Field(alias="_id")
    name: str = Field(...)


class QuestionAnswer(BaseModel):
    question: str = Field(...)