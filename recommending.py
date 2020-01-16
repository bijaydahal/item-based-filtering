# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:53:21 2020

@author: Welcome
"""

import pandas as pd
from CFModel import CFModel


RATINGS_CSV_FILE = 'ml1m_ratings.csv'
USERS_CSV_FILE = 'ml1m_users.csv'
MOVIES_CSV_FILE = 'ml1m_movies.csv'
MODEL_WEIGHTS_FILE = 'ml1m_weights1.h5'
K_FACTORS = 120
TEST_USER = 3000

ratings = pd.read_csv(RATINGS_CSV_FILE, sep='\t', encoding='latin-1', usecols=['userid', 'movieid', 'rating'])
max_userid = ratings['userid'].drop_duplicates().max()
max_movieid = ratings['movieid'].drop_duplicates().max()
print (len(ratings), 'ratings loaded.')

users = pd.read_csv(USERS_CSV_FILE, sep='\t', encoding='latin-1', usecols=['userid', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
print (len(users), 'descriptions of', max_userid, 'users loaded.')


movies = pd.read_csv(MOVIES_CSV_FILE, sep='\t', encoding='latin-1', usecols=['movieid', 'title', 'genre'])
print (len(movies), 'descriptions of', max_movieid, 'movies loaded.')

trained_model = CFModel(max_userid, max_movieid, K_FACTORS)

trained_model.load_weights(MODEL_WEIGHTS_FILE)


users[users['userid'] == TEST_USER]


def predict_rating(userid, movieid):
    return trained_model.rate(userid - 1, movieid - 1)


def predict_rating_for_user(test_user):
    
    TEST_USER = test_user
    
    user_ratings = ratings[ratings['userid'] == TEST_USER][['userid', 'movieid', 'rating']]
    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['movieid']), axis=1)
    user_ratings.sort_values(by='rating', 
                             ascending=False).merge(movies, 
                                                    on='movieid', 
                                                    how='inner', 
                                                    suffixes=['_u', '_m'])
    recommendations = ratings[ratings['movieid'].isin(user_ratings['movieid']) == False][['movieid']].drop_duplicates()
    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(TEST_USER, x['movieid']), axis=1)
    return recommendations.sort_values(by='prediction',
                              ascending=False).merge(movies,
                                                     on='movieid',
                                                     how='inner',
                                                     suffixes=['_u', '_m']).head(10)
                                                     
     
                                                     
predict_rating_for_user(TEST_USER)