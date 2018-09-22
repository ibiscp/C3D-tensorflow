import json
import os

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
