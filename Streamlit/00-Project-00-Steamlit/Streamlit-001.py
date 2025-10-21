import streamlit as st

st.title("Ask Me Anything 📚")
question = st.text_input("Type your question here:")

if question:
    st.write("You asked:", question)
    st.write("Answer: (Imagine I searched your PDF here)")

# 2025-10-21 17:25:50.940
#   Warning: to view this Streamlit app on a browser, run it with the following
#   command:
#
#     streamlit run E:\Programming\projects\python\useful-scripts\Streamlit\Streamlit-001.py [ARGUMENTS]