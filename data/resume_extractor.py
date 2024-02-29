from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import re
import pandas as pd

class ResumeExtractor:

    def preprocess_text(self, text):
        tokens = self.mystem.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in self.russian_stopwords\
                and token != " " \
                and token.strip() not in punctuation]
    
        text = " ".join(tokens)
        
        return text

    def extract_english_words(self, text):
        english_word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        english_words = english_word_pattern.findall(text)
        
        return english_words
    
    #возвращает кем работал, key_skills, extra_skills, сколько работал (в месяцах), возраст
    def get_working_years_extra_skills_positions(self, resume):
        positions = set()
        key_skills = set()
        if 'key_skills' in resume and not(resume['key_skills'] == None):
            key_skills = set(resume['key_skills'].replace('.', '').replace(' ', '').split(','))
        extra_skills = set()
        working_months = 0
        age = None if resume['birth_date'] == None else 2024 - int(resume['birth_date'].split('-')[0])

        last_position = ''

        about = resume['about'] if 'description' in resume and  resume['about'] is not None  else ''
        extra_skills = set(self.extract_english_words(about))
    
        if 'experienceItem' in resume and resume['experienceItem'] is not None:
            for last_job in resume['experienceItem']:
                if 'position' in last_job and last_job['position'] is not None:
                    positions.add( last_job['position'])
                    last_position = last_job['position']
            
                if last_job['starts'] != None:
                    start = list(map(int, last_job['starts'].split('-')))
                    ends = list(map(int, last_job['ends'].split('-'))) if last_job['ends'] != None else [2024]
                    
                    if len(start) < 2:
                        start.append(0)
                        
                    if len(ends) < 2:
                        ends.append(0)
        
            
                    months = (ends[0] - start[0]) * 12
                    months += ends[1] - start[1]        
                    working_months += months
                
                descript = last_job['description'] if 'description' in last_job and last_job['description'] is not None else ''
            
                #print(descript)
                skills = set(self.extract_english_words(descript))
            
                extra_skills = skills | extra_skills
        extra_skills -= key_skills

        is_junior, is_middle, is_senior, is_teamlead = self.extract_grade(last_position)
    
        return positions, key_skills, extra_skills, working_months, age, is_junior, is_middle, is_senior, is_teamlead
    
    def extract_grade(self, position):
        is_junior, is_middle, is_senior, is_teamlead = False, False, False, False

        if re.findall('джун*|junior', position, re.IGNORECASE):
            is_junior = True
        elif re.findall('миддл|мидл|middle|midle', position, re.IGNORECASE):
            is_middle = True
        elif re.findall('senior|с(и|е)нь(е|ё|о)р', position, re.IGNORECASE): 
            is_senior = True
        elif re.findall('teamlead|тимлид|руководитель|lead|head|ведущий', position, re.IGNORECASE):
            is_teamlead = True
        else:
            is_middle = True
        
        return is_junior, is_middle, is_senior, is_teamlead

    def __init__(self, vacancies):

        self.mystem = Mystem() 
        self.russian_stopwords = stopwords.words("russian")
        
        self.relevance = []
        self.positions = []
        self.key_skills = []
        self.extra_skills = []
        self.working_months = []
        self.age = []
        self.uuid = []
        self.is_junior = []
        self.is_middle = []
        self.is_senior = []
        self.is_teamlead = []
        self.vacancy_uuid = []

        for vacancy_index in range(len(vacancies)):
            for confirmation_type in ['confirmed_resumes', 'failed_resumes']:
                for resume in vacancies[vacancy_index][confirmation_type]:
                    positions, key_skills, extra_skills, working_months, age, is_junior, is_middle, is_senior, is_teamlead = self.get_working_years_extra_skills_positions(resume)

                    key_skills = list(key_skills)
                    extra_skills = list(extra_skills)

                    for index in range(len(key_skills)):
                        key_skills[index] = self.preprocess_text(key_skills[index])

                    for index in range(len(extra_skills)):
                        extra_skills[index] = self.preprocess_text(extra_skills[index])

                    self.positions.append(positions)
                    self.key_skills.append(key_skills)
                    self.extra_skills.append(extra_skills)
                    self.working_months.append(working_months)
                    self.age.append(age)
                    self.uuid.append(resume['uuid'])
                    self.vacancy_uuid.append(vacancies[vacancy_index]['vacancy']['uuid'])
                    self.relevance.append(1 if confirmation_type == 'confirmed_resumes' else 0)
                    self.is_junior.append(is_junior)
                    self.is_middle.append(is_middle)
                    self.is_senior.append(is_senior)
                    self.is_teamlead.append(is_teamlead)

    def form_dataset(self):
        return pd.DataFrame({
            'vacancy_uuid': self.vacancy_uuid,
            'is_junior': self.is_junior,
            'is_middle': self.is_middle,
            'is_senior': self.is_senior,
            'is_teamlead': self.is_teamlead,
            'relevance': self.relevance,
            'positions': self.positions,
            'key_skills': self.key_skills,
            'extra_skills': self.extra_skills,
            'working_months': self.working_months,
            'uuid': self.uuid,
            'age': self.age
        })