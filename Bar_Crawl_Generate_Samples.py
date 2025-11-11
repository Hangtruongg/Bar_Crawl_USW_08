import pandas
from os import listdir
import pickle

path = 'data/clean_tac/' # directory where clean TAC files are stored
df = pandas.read_csv('data/all_accelerometer_data_pids_13.csv')
files = listdir(path) # list of TAC files for each participant

# loop through each TAC files
for file in files:

    tacs = pandas.read_csv(path + file) # load TAC readings tacs for one participant
    pid = file.split('_')[0] #extract pid(participant ID) from filename (before _)
    previous_time = 0 # use to select accelerometer readings between consecutive TAC timestamps

    # for each TAC measurement:
    for i in range(len(tacs)):
        item = dict() # create item dictionary
        item['tac'] = tacs.TAC_Reading[i] # store TAC value
        item['time'] = tacs.timestamp[i]*1000 # convert TAC timestamp to milliseconds
        readings = df[ (df['pid'] == pid) & (df['time'] <= item['time']) & (df['time'] > previous_time)] # filters accelerometer data (df) matching the participant and select rows between previouse_time and current TAC timestamp
        previous_time = item['time'] # update previous_time for the next TAC reading
        if len(readings)>0: # if there are any accelerometer readings for this TAC interval
            item['readings'] = readings # add readings to item
            with open('data/samples/' + pid + '_' + str(item['time']) + '.pkl', 'wb') as f: # save item as pickle file in data/samples/ and file format will be participantID_timestamp.pkl
                pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)