import pandas as pd

data = {
    'Name': ['Kana', 'Aki', 'Ritu', 'Dev', 'Meera', 'Yuki', 'Ishaan', 'Tara', 'Rey', 'Neha'],
    'Wakeup': ['Early', 'Mid', 'Late', 'Early', 'Late', 'Mid', 'Early', 'Late', 'Mid', 'Early'],
    'Sleep': ['Early', 'Late', 'Late', 'Mid', 'Early', 'Mid', 'Late', 'Late', 'Early', 'Mid'],
    'Cleanliness': [5, 2, 3, 4, 5, 1, 3, 2, 4, 5],
    'IntroExtro': [0.2, 0.7, 0.3, 0.5, 0.1, 0.9, 0.6, 0.4, 0.8, 0.3],
    'StudyTime': ['Morning', 'Night', 'Night', 'Morning', 'Morning', 'Night', 'Night', 'Morning', 'Morning', 'Night'],
    'NoiseTolerance': [2, 5, 3, 4, 1, 5, 3, 2, 4, 2],
    'PetFriendly': [True, False, True, False, True, False, True, True, False, True],
    'IdealRoommate': [
        'Quiet and wakes up early',
        'Talkative and fun',
        'Clean and chill',
        'Loves morning workouts',
        'Very organized and lowkey',
        'Fun and loves music',
        'Flexible and calm',
        'Lazy but sweet',
        'Chill and peaceful',
        'Early sleeper, early riser'
    ]
}

df = pd.DataFrame(data)
df.to_csv("roommates_mock_data.csv", index=False)
