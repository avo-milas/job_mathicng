from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import re
import pandas as pd
import json

class LlmExtractor:

    def preprocess_text(self, text):
        tokens = self.mystem.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in self.russian_stopwords\
                and token != " " \
                and token.strip() not in punctuation]
    
        text = " ".join(tokens)
        
        return text
    
    def get_llm_extracted_data(self, vacancies):
        
        llm = open('./data/llm_vacancies.json')
        llm_vacancies = json.load(llm)
    
        for i in range(len(vacancies)):
            llm_vacancies[i]['uuid'] = vacancies[i]['vacancy']['uuid']

        return llm_vacancies
    
    def parse_work_type(self, llm_work_type):
        if llm_work_type == None:
            return None
        result = None
        if re.findall('график|офис*|удал*|дист|гибридн*|гибкий', llm_work_type, re.IGNORECASE):
            result = "office"
        if re.findall('офис*', llm_work_type, re.IGNORECASE):
            result = "office"
        if re.findall('удал*|дист*', llm_work_type, re.IGNORECASE):
            result = "distance"
        if re.findall('гибридн*|гибкий', llm_work_type, re.IGNORECASE):
            result = "flexible"

        return result
    
    def extract_grade(self, position):
        is_junior, is_middle, is_senior, is_teamlead = False, False, False, False

        if re.findall('джун*|junior', position, re.IGNORECASE):
            is_junior = True
        if re.findall('миддл|мидл|middle|midle', position, re.IGNORECASE):
            is_middle = True
        if re.findall('senior|с(и|е)нь(е|ё|о)р', position, re.IGNORECASE): 
            is_senior = True
        if re.findall('teamlead|тимлид|руководитель|lead|head|ведущий', position, re.IGNORECASE):
            is_teamlead = True
        
        return is_junior, is_middle, is_senior, is_teamlead

    def __init__(self, vacancies):

        llm_data = self.get_llm_extracted_data(vacancies)

        self.mystem = Mystem() 
        self.russian_stopwords = stopwords.words("russian")
        
        self.hard_skills = []
        self.extra_skills = []
        self.benefits = []
        self.experience = []
        self.work_type = []
        self.monthly_wage = []
        self.is_junior = []
        self.is_middle = []
        self.is_senior = []
        self.is_teamlead = []
        self.vacancy_uuid = []


        for vacancy in llm_data:
            if vacancy['hard_skills'] is None:
                self.hard_skills.append([])
            else:
                self.hard_skills.append([self.preprocess_text(hard_skill) for hard_skill in vacancy['hard_skills'].split(', ')])

            if vacancy['soft_skills'] is None:
                self.extra_skills.append([])
            else:
                self.extra_skills.append([self.preprocess_text(soft_skill) for soft_skill in vacancy['soft_skills'].split(', ')])

            self.benefits.append(len(vacancy['benefits'].split(', ')) if vacancy['benefits'] is not None else 0)
            self.experience.append(vacancy['experience'])
            self.work_type.append(self.parse_work_type(vacancy['type']))
            self.monthly_wage.append(vacancy['monthly_wage'])

            is_junior, is_middle, is_senior, is_teamlead = self.extract_grade('' if vacancy['level'] is None else vacancy['level'])

            self.is_junior.append(is_junior)
            self.is_middle.append(is_middle)
            self.is_senior.append(is_senior)
            self.is_teamlead.append(is_teamlead)
            self.vacancy_uuid.append(vacancy['uuid'])


    def form_dataset(self):
        return pd.DataFrame({
            'vacancy_uuid': self.vacancy_uuid,
            'hard_skills': self.hard_skills,
            'extra_skills': self.extra_skills,
            'benefits': self.benefits,
            'experience': self.experience,
            'work_type': self.work_type,
            'monthly_wage': self.monthly_wage,
            'is_junior': self.is_junior,
            'is_middle': self.is_middle,
            'is_senior': self.is_senior,
            'is_teamlead': self.is_teamlead
        })