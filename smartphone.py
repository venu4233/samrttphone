import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title of the app
st.set_page_config(page_title="Phone Addiction Predictor", page_icon="📱")
st.title("📱 Phone Addiction Predictor")
st.markdown("### Predict if you're at risk of phone addiction based on your daily usage.")

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

model = RandomForestClassifier()
model.fit(X, y)

# --- User Inputs ---
st.sidebar.header("📊 Input your phone usage data")

screen_time = st.sidebar.slider("Screen Time (hours)", 0.0, 12.0, 3.0, step=0.1)
unlock_count = st.sidebar.slider("Unlock Count per Day", 0, 200, 60, step=1)
response_time = st.sidebar.slider("Response Time to Notifications (sec)", 0, 60, 15, step=1)

if st.sidebar.button("Predict Addiction Risk"):
    user_data = [[screen_time, unlock_count, response_time]]
    prediction = model.predict(user_data)[0]

    # --- Results Display ---
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error("⚠️ You might be *Addicted* to your phone.")
    else:
        st.success("✅ You are *Not Addicted*.")

    st.markdown("---")
    st.subheader("💡 Recommendations")

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
