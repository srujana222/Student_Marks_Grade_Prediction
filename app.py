from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


model = joblib.load("model.pkl")


app = FastAPI(
    title="Student Grade Prediction API",
    description="Regression API using FastAPI",
    version="1.0"
)
class StudentData(BaseModel):
    AGE: int
    GENDER: int
    HS_TYPE: int
    SCHOLARSHIP: int
    WORK: int
    ACTIVITY: int
    PARTNER: int
    SALARY: int
    TRANSPORT: int
    LIVING: int
    MOTHER_EDU: int
    FATHER_EDU: int
    KIDS: int
    MOTHER_JOB: int
    FATHER_JOB: int
    STUDY_HRS: int
    READ_FREQ: int
    READ_FREQ_SCI: int
    ATTEND_DEPT: int
    IMPACT: int
    ATTEND: int
    PREP_EXAM: int
    NOTES: int
    LISTENS: int
    LIKES_DISCUSS: int
    CLASSROOM: int
    CUML_GPA: int
    EXP_GPA: int
    PREP_STUDY: int
    

@app.get("/")
def home():
    return {
        "message": "FastAPI Regression Model Running"
    }


@app.post("/predict")
def predict(data: StudentData):
    try:
        input_data = pd.DataFrame([data.dict()])

        prediction = model.predict(input_data)[0]

        return {
             "predicted_grade": round(float(prediction), 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
