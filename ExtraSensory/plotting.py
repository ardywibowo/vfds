import os

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_actual_date_labels(tick_seconds):
    import datetime
    import pytz
    
    time_zone = pytz.timezone('US/Pacific') # Assuming the data comes from PST time zone
    weekday_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    datetime_labels = []
    for timestamp in tick_seconds:
        tick_datetime = datetime.datetime.fromtimestamp(timestamp,tz=time_zone)
        weekday_str = weekday_names[tick_datetime.weekday()]
        time_of_day = tick_datetime.strftime('%I:%M%p')
        datetime_labels.append('%s\n%s' % (weekday_str,time_of_day))
        pass
    
    return datetime_labels
    
def figure__context_over_participation_time(timestamps,Y,label_names,labels_to_display,label_colors):

    fig = plt.figure(figsize=(10,7),facecolor='white')
    ax = plt.subplot(1,1,1)
    
    seconds_in_day = (60*60*24)

    ylabels = []
    ax.plot(timestamps,len(ylabels)*np.ones(len(timestamps)),'|',color='0.5',label='(Collected data)')
    ylabels.append('(Collected data)')

    for (li,label) in enumerate(labels_to_display):
        lind = label_names.index(label)
        is_label_on = Y[:,lind]
        label_times = timestamps[is_label_on]

        label_str = get_label_pretty_name(label)
        ax.plot(label_times,len(ylabels)*np.ones(len(label_times)),'|',color=label_colors[li],label=label_str)
        ylabels.append(label_str)

    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day)
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)
    plt.xlabel('days of participation',fontsize=14)
    
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels,fontsize=14)

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels,fontsize=14)

    ax.set_ylim([-1,len(ylabels)])
    ax.set_xlim([min(timestamps),max(timestamps)])
    
    return


def plot_single_trajectory(predictions, labels, sel_weights, save_dir):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    sel_weights = sel_weights.cpu().numpy()

    predictions = predictions[1:]
    labels = labels[1:]
    sel_weights = sel_weights[1:]
    time = range(len(sel_weights))

    nonzero_features = np.any(sel_weights, axis=0)
    nonzero_index = np.nonzero(nonzero_features)
    nonzero_index = nonzero_index[0]

    sel_weights = sel_weights[:, nonzero_features]

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(time, labels, label="True")
    plt.plot(time, predictions, label="Prediction")
    plt.xlim([0, len(time)])
    plt.yticks([0, 1, 2, 3, 4, 5], 
        ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying'], rotation=40)
    plt.legend(loc=4)
    plt.xlabel('Time (50 Hz)')

    plt.subplot(212)
    plt.imshow(sel_weights.transpose(), aspect='auto')
    plt.yticks(range(sel_weights.shape[1]), nonzero_index, rotation=20)
    plt.ylabel('Feature Number')
    plt.xlabel('Time (50 Hz)')

    save_dir = os.path.join(save_dir)
    plt.savefig(save_dir)
    plt.close()