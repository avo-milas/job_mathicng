from nltk.tokenize import sent_tokenize, word_tokenize
import re
import pandas as pd
import nltk
nltk.download("stopwords")
nltk.download('punkt')
#--------#

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

class DumbParser:

    #Preprocess function
    def preprocess_text(self, text):
        tokens = self.mystem.lemmatize(text.lower())
        tokens = [token for token in tokens if token not in self.russian_stopwords\
                and token != " " \
                and token.strip() not in punctuation]
    
        text = " ".join(tokens)
        
        return text

    def _is_intersect(self, a_l, a_r, b_l, b_r):
        return not ((a_r < b_l) or (b_r < a_l))
    
    def get_grade(self, sentences):
        is_junior = {}
        is_middle = {}
        is_senior = {}
        is_teamlead = {}
        detected_count = 0
        for curr_vacancy in range(len(sentences)):
            is_detected = False
            is_junior[curr_vacancy] = None
            is_middle[curr_vacancy] = None
            is_senior[curr_vacancy] = None
            is_teamlead[curr_vacancy] = None
            for index, description in enumerate(sentences[curr_vacancy]):
                if description.lower().find('грейд') >= 0 or description.lower().find('grade') >= 0:
                   numbers = re.findall(r'\d+', description)
                   min_grade = min([int(number) if 10 <= int(number) and int(number) <= 22 else 22 for number in numbers])
                   max_grade = max([int(number) if 10 <= int(number) and int(number) <= 22 else 10 for number in numbers])
                   is_junior[curr_vacancy] = self._is_intersect(10, 13, min_grade, max_grade)
                   is_middle[curr_vacancy] = self._is_intersect(14, 16, min_grade, max_grade)
                   is_senior[curr_vacancy] = self._is_intersect(17, 19, min_grade, max_grade)
                   is_teamlead[curr_vacancy] = self._is_intersect(20, 22, min_grade, max_grade)
                   is_detected = True
                elif re.findall('джун*|junior|миддл|мидл|middle|midle|senior|синьор|синьеры|teamlead|тимлид', description, re.IGNORECASE):
                    is_junior[curr_vacancy] = False
                    is_middle[curr_vacancy] = False
                    is_senior[curr_vacancy] = False
                    is_teamlead[curr_vacancy] = False
                    if re.findall('джун*|junior', description, re.IGNORECASE):
                        is_junior[curr_vacancy] = True
                    elif re.findall('миддл|мидл|middle|midle', description, re.IGNORECASE):
                       is_middle[curr_vacancy] = True
                    elif re.findall('senior|с(и|е)нь(е|ё|о)р', description, re.IGNORECASE): 
                        is_senior[curr_vacancy] = True
                    elif re.findall('teamlead|тимлид|руководитель|lead|head', description, re.IGNORECASE):
                        is_teamlead[curr_vacancy] = True
                    is_detected = True
                    
            detected_count += (1 if is_detected else 0)
        
        print(f"get_grade data extractor fullness: {detected_count}/{len(sentences)}")
        return is_junior, is_middle, is_senior, is_teamlead

    def get_work_type(self, sentences):
        work_type = {}
        detected_count = 0
        for curr_vacancy in range(len(sentences)):
            is_detected = False
            work_type[curr_vacancy] = None
            for index, description in enumerate(sentences[curr_vacancy]):
                if re.findall('график|офис*|удал*|дист|гибридн*|гибкий', description, re.IGNORECASE):
                    work_type[curr_vacancy] = "office"
                    if re.findall('офис*', description, re.IGNORECASE):
                        work_type[curr_vacancy] = "office"
                    if re.findall('удал*|дист*', description, re.IGNORECASE):
                        work_type[curr_vacancy] = "distance"
                    if re.findall('гибридн*|гибкий', description, re.IGNORECASE):
                        work_type[curr_vacancy] = "flexible"
                    is_detected = True
            detected_count += (1 if is_detected else 0)
        
        print(f"get_work_type data extractor fullness: {detected_count}/{len(sentences)}")
        return work_type
    
    def get_experience(self, sentences):
        experience = {}
        detected_count = 0
        for curr_vacancy in range(len(sentences)):
            is_detected = False
            experience[curr_vacancy] = None
            for index, description in enumerate(sentences[curr_vacancy]):
                p = re.compile('\d* (год*|лет)', re.IGNORECASE)
                if p.finditer(description):
                    values = [match.group().split(' ')[0] for match in p.finditer(description) if match.group().split(' ')[0] != '']
                    if len([float(experience) for experience in values if 1 <= float(experience) and float(experience) <= 5]) == 0:
                        continue
                    experience[curr_vacancy] = [float(experience) for experience in values if 1 <= float(experience) and float(experience) <= 10][0]
                    is_detected = True
            detected_count += (1 if is_detected else 0)
        print(f"get_experience data extractor fullness: {detected_count}/{len(sentences)}")
        return experience
    
    def get_key_skills(self, sentences, hh_ru_skills):
        key_skills = {}
        detected_count = 0
        for curr_vacancy in range(len(sentences)):
            is_detected = False
            key_skills[curr_vacancy] = []
            for index, description in enumerate(sentences[curr_vacancy]):
                words = [self.preprocess_text(word) for word in word_tokenize(self.preprocess_text(description))]
                words = [word for word in words if word in hh_ru_skills]
                key_skills[curr_vacancy].extend(words)

            key_skills[curr_vacancy] = list(set(key_skills[curr_vacancy]))
            detected_count += (1 if len(key_skills[curr_vacancy]) > 0 else 0)
        print(f"get_experience data extractor fullness: {detected_count}/{len(sentences)}")

        return key_skills


    def __init__(self, vacancies):

        self.mystem = Mystem() 
        self.russian_stopwords = stopwords.words("russian")

        self.sentences = {}
        self.uuid = [vacancy['vacancy']['uuid'] for vacancy in vacancies]

        for index, vacancy in enumerate(vacancies):
            text = vacancy['vacancy']['description'] 
            self.sentences[index] = sent_tokenize(text)

        self.is_junior, self.is_middle, self.is_senior, self.is_teamlead = self.get_grade(self.sentences)
        self.work_type = self.get_work_type(self.sentences)
        self.experience = self.get_experience(self.sentences)

        # preprocessing skills

        hh_vacancies = pd.read_csv('./vacancies.csv') #hh ru dataset
        skills = []

        for index, i in hh_vacancies.iterrows():
            for j in list(i['tags'][1:-1].split(',')):
                skills.append(self.preprocess_text(j))

        hh_ru_skills = set(skills)
        hh_ru_skills.remove('')

        self.key_skills = self.get_key_skills(self.sentences, hh_ru_skills)

    def form_dataset(self):
        print(len(self.is_junior))
        print(len(self.work_type))
        return pd.DataFrame({
            'uuid': self.uuid,
            'is_junior': self.is_junior.values(),
            'is_middle': self.is_middle.values(),
            'is_senior': self.is_senior.values(),
            'is_teamlead': self.is_teamlead.values(),
            'work_type': self.work_type.values(),
            'experience': self.experience.values(),
            'skills': [str(skills_list) for skills_list in self.key_skills.values()]
        })