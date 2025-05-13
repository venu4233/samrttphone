import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
import ipywidgets as widgets

# Model training
data = {
    'screen_time': [2.5, 6.1, 4.0, 7.5, 3.3, 5.8, 6.0, 1.9],
    'unlock_count': [40, 100, 60, 130, 45, 90, 110, 30],
    'response_time': [25, 5, 20, 3, 30, 10, 7, 40],
    'label': [0, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df[['screen_time', 'unlock_count', 'response_time']]
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

# Widgets for inputs
screen_time = widgets.FloatSlider(min=0, max=12, step=0.1, description='Screen Time (hrs)')
unlock_count = widgets.IntSlider(min=0, max=200, step=1, description='Unlock Count')
response_time = widgets.IntSlider(min=0, max=60, step=1, description='Response Time (sec)')
button = widgets.Button(description="Predict")

output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        user_data = [[screen_time.value, unlock_count.value, response_time.value]]
        prediction = model.predict(user_data)[0]
        print("Prediction:", "Addicted" if prediction == 1 else "Not Addicted")
        
        print("\nRecommendations:")
        if screen_time.value > 6:
            print("- Reduce screen time below 5 hours.")
        if unlock_count.value > 100:
            print("- Decrease phone unlocks.")
        if response_time.value < 10:
            print("- Slow down notification responses.")
        if prediction == 0:
            print("- Great job managing your phone use!")

button.on_click(on_button_click)

display(screen_time, unlock_count, response_time, button, output)