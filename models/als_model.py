import importlib
import json
import pandas as pd
import numpy as np
from base_model import BaseSelector
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer


class ALSModel(BaseSelector):

    def skills_score(self, req_skills, candidate_skills):
        if req_skills is not None and candidate_skills is not None:
            req_skills = set(item for item in req_skills[1:-1].split(', '))
            candidate_skills = set(item for item in candidate_skills[1:-1].split(', '))
            common_skills = req_skills.intersection(candidate_skills)
            return len(common_skills) /  len(req_skills)
        return 0


    def grade_score(self, vac, res):
        return any((vac[['is_junior', 'is_middle', 'is_senior', 'is_teamlead']].values & res[['is_junior', 'is_middle', 'is_senior', 'is_teamlead']].values).flatten())


    def sum_score(self, ss, gs):
        return ss / 2 if not gs else ss


    def __init__(self, llm_dataset, resume_dataset, llm_control_dataset, resume_control_dataset):
        self.data = []
        self.llm_dataset = llm_dataset
        self.resume_dataset = resume_dataset
        self.llm_control_dataset = llm_control_dataset
        self.resume_control_dataset = resume_control_dataset
        self.user_vectors = []
        self.item_vectors = []
        self.user_id_mapping = []
        self.item_id_mapping = []
        

    def fit(self, data):
        self.data = data

        user_ids = set()
        item_ids = set()

        for entry in data:
            item_ids.add(entry["vacancy"]["uuid"])
            for resume in entry["failed_resumes"] + entry["confirmed_resumes"]:
                user_ids.add(resume["uuid"])

        for res in self.resume_control_dataset.itertuples():
            user_ids.add(res.uuid)

        for vac in self.llm_control_dataset.itertuples():
            item_ids.add(vac.vacancy_uuid)

        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_id_mapping = {item_id: idx for idx, item_id in enumerate(item_ids)}

        rows = []
        cols = []
        values = []

        for entry in data:
            for resume in entry["failed_resumes"] + entry["confirmed_resumes"]:
                user_idx = self.user_id_mapping[resume["uuid"]]
                item_idx = self.item_id_mapping[entry["vacancy"]["uuid"]]
                rows.append(user_idx)
                cols.append(item_idx)
                values.append(1 if resume in entry["confirmed_resumes"] else 0)


        for i, resume in self.resume_control_dataset.iterrows():

            key_skills = resume.key_skills
            extra_skills = resume.extra_skills
            sum_skills = key_skills + extra_skills
            user_idx = self.user_id_mapping[resume.uuid]

            for vac_i in range(len(data)):

                item_idx = vac_i

                hard_skills_score1 = self.skills_score(self.llm_dataset.iloc[vac_i].hard_skills, sum_skills)
                grade_score1 =  self.grade_score(self.llm_dataset.iloc[vac_i], resume)

                sum_score1 = self.sum_score(hard_skills_score1, grade_score1)

                rows.append(user_idx)
                cols.append(item_idx)
                values.append(sum_score1)

        sparse_data = csr_matrix((values, (rows, cols)))

        model = AlternatingLeastSquares(factors=70, regularization=1.5, iterations=100)
        model.fit(sparse_data)

        # self.user_vectors = model.user_factors
        # self.item_vectors = model.item_factors

        user_vectors = model.user_factors
        item_vectors = model.item_factors

        user_scaler = StandardScaler()
        user_vectors_normalized = user_scaler.fit_transform(user_vectors)
        item_scaler = StandardScaler()
        item_vectors_normalized = item_scaler.fit_transform(item_vectors)

        self.user_vectors = user_vectors_normalized
        self.item_vectors = item_vectors_normalized
        

    def predict(self, k: int):
        vacancy_uuid = self.llm_control_dataset.vacancy_uuid.values[0]
        print(vacancy_uuid)
        new_item_vector = self.item_vectors[self.item_id_mapping[vacancy_uuid]]
        result = []

        for res in self.resume_control_dataset.itertuples():
            new_user_vector = self.user_vectors[self.user_id_mapping[res.uuid]]
            prediction = new_user_vector.dot(new_item_vector)
            result.append((prediction, res.uuid))

        result.sort(reverse=True)
        result = list(map(lambda x: x[1], result))[:k]
        return result

