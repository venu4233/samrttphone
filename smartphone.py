import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(
    page_title="Phone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

# Title and description
st.title("📱 Phone Addiction Predictor")
st.markdown(
    """
    This tool uses your daily phone usage patterns to predict whether you're at risk of phone addiction.
    Just move the sliders in the sidebar and click **Predict**!
    """
)

# --- Train the model ---
data = {
    'screen_time': [2.5, 6.1, 4.0, 7.5, 3.3, 5.8, 6.0, 1.9],
    'unlock_count': [40, 100, 60, 130, 45, 90, 110, 30],
    'response_time': [25, 5, 20, 3, 30, 10, 7, 40],
    'label': [0, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df[['screen_time', 'unlock_count', 'response_time']]
y = df['label']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- Sidebar Inputs ---
st.sidebar.header("📊 Your Usage Data")

screen_time = st.sidebar.slider("Screen Time (hours)", 0.0, 12.0, 3.0, 0.1)
unlock_count = st.sidebar.slider("Unlocks per Day", 0, 200, 60, 1)
response_time = st.sidebar.slider("Notification Response Time (sec)", 0, 60, 15, 1)

if st.sidebar.button("🔍 Predict"):
    # Make prediction
    user_data = [[screen_time, unlock_count, response_time]]
    prediction = model.predict(user_data)[0]
    prediction_proba = model.predict_proba(user_data)[0][1]  # probability of being addicted

    # Show prediction result
    st.markdown("---")
    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error("⚠️ You might be *Addicted* to your phone.")
    else:
        st.success("✅ You're *Not Addicted*. Keep it up!")

    # Visual risk indicator
    st.markdown("#### 📊 Risk Level")
    st.progress(int(prediction_proba * 100))

    # Recommendations
    st.markdown("### 💡 Personalized Recommendations")

    if screen_time > 6:
        st.write("- ⏳ Try to reduce screen time to under 5 hours daily.")
    if unlock_count > 100:
        st.write("- 🔓 Limit the number of times you unlock your phone.")
    if response_time < 10:
        st.write("- 🧘 Practice delaying responses to notifications.")
    if prediction == 0:
        st.write("- 🎉 Great job! Keep maintaining your healthy phone habits.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")
