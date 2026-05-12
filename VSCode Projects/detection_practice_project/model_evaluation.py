import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('C:/Users/ajibo/github_repos/gesture_flow_app_project/data/practice_project_data/raw_v_smooth.csv')

def get_switch_rate():
    raw_switches = 0
    for i in range(1, len(df)):
        if df.loc[i, 'raw_pred'] != df.loc[i-1, 'raw_pred']:
            raw_switches += 1

    smooth_switches = 0

    for i in range(1, len(df)):
        if df.loc[i, 'smoothed_pred'] != df.loc[i-1, 'smoothed_pred']:
            smooth_switches += 1

    raw_switch_rate = raw_switches/len(df)
    smooth_switch_rate = smooth_switches/len(df)
    return raw_switch_rate, smooth_switch_rate


def get_correctness():
    segments = [
                (0, 150, 'Fist'),
                (160, 310, 'Open'),
                (320, 470, 'Peace')
                ]
    for start, end, true_label in segments:
        segment = df.iloc[start:end+1]
        raw_correct = (segment["raw_pred"] == true_label).sum()
        smooth_correct = (segment["smoothed_pred"] == true_label).sum()
        total = len(segment)
        raw_consistency = raw_correct/total
        smooth_consistency = smooth_correct/total
        print(f"RAW: {true_label}: {raw_consistency:.2f}")
        print(f"SMOOTH: {true_label}: {smooth_consistency:.2f}")
    return raw_consistency, smooth_consistency
    

def get_confidence_stability():
    raw_correct = 0
    for i in range(1, len(df)):
        if df.loc[i, 'raw_pred'] == df.loc[i-1, 'raw_pred']:
            if df.loc[i, 'raw_conf'] > 0.70:
                raw_correct += 1
    
    smooth_correct = 0
    for i in range(1, len(df)):
        if df.loc[i, 'smoothed_pred'] == df.loc[i-1, 'smoothed_pred']:
            if df.loc[i, 'smoothed_conf'] > 0.70:
                smooth_correct += 1

    smooth_pct = smooth_correct/len(df)
    raw_pct = raw_correct/len(df)
    return raw_pct, smooth_pct



r_switch, s_switch = get_switch_rate()
r_correct, s_correct = get_correctness()
r_confidence, s_confidence = get_confidence_stability()
labels = ['Raw Switches', 'Smoothed Switches',
          'Raw Correctness', 'Smooth Correctness',
          'Raw Confidence', 'Smooth Confidence']

values = [r_switch, s_switch, r_correct, s_correct, r_confidence, s_confidence]
plt.bar(labels, values)
plt.ylabel("Switch Rate")
plt.title("Prediction Stability Comparison")
plt.show()