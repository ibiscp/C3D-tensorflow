import json
import os
import activities

# Collects all the json files from the folder and assign a
# unique ID for each one
def generate_activity_list(json_dir='json/'):
    files = os.listdir(json_dir)

    activities_list = []

    for f in files:
        with open(json_dir + f) as file:
            Json_dict = json.load(file)

            for video in list(Json_dict.keys()):
                for activity in list(Json_dict[video]):
                    if (activity['label'] not in activities_list):
                        activities_list.append(activity['label'])

    activities_ids = dict(map(reversed, enumerate(activities_list)))

    # print('Activity list size: ', len(activities_list))
    # print('Activities IDs: ', activities_ids)

    return activities_ids

# Show the frequency of each activity
def show_activities_frequency(json_dir='json/'):
    files = os.listdir(json_dir)

    activities_dic = {}

    activities_list = activities.activities_tfrecords

    for f in files:
        with open(json_dir + f) as file:
            Json_dict = json.load(file)

            for video in list(Json_dict.keys()):
                for activity in list(Json_dict[video]):
                    if activity['label'] in activities_list:
                        if (activity['label'] not in activities_dic):
                            activities_dic[activity['label']] = 0
                        else:
                            activities_dic[activity['label']] += 1

    return activities_dic