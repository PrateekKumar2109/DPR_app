import streamlit as st
st.set_page_config(
    page_title="Hello",
    page_icon="👋"
)
#st.title(" NLP app for OIl & Gas Drilling Data")

#st.header(" Welcome to O & G Assistants! 👋")
st.header(" Welcome  👋")
st.sidebar.success("Select a demo above.")
st.markdown(
    """
    We are here to make Oil and Gas Engineer life easy   
""")
activities=['Login','About']
choice=st.sidebar.selectbox("Select Activity", activities)
if choice=='Login':
    st.subheader('Login')
    username=st.text_input("Enter Username")
    password=st.text_input("Enter Password", type='password')
    if st.button("Submit"):
        if password=='12345':
            st.balloons()
            st.write("Hello {}".format(username))
        else:
            st.warning('Wrong Password')
    
elif choice=='About':   
    st.write(
        """LLMs(Large Language Models) are transforming the NLP(Natural Language Processing) downstreak tasks.LLMs are available as an api for GPT models 
        and cohere models .We are Energy professionals &  our aim is to reduce the complexitiy of O & G Industy. 
             """)
    #st.write(
    #    """With the Release Streamlit Version 1.10.0 it is now possible to make a Multi-Page application 
    # eliminating need of third party plugins. In this Web application we are working on Pressure & Temperature Survey Data
    # which is used to make decision in day to day life of a well. We 
    #         are Energy professionals &  our aim is to reduce the complexitiy of O & G Industy. 
    #         """)
expander = st.expander("Domain Knowledge of Oil & Gas ")
#expander.write("""
#     Pressure & Temperature hole Surveys are carried out in the well frequently. Shut-in & flowing 
#     pressure & temp. surveys are recorded to analyze the pressures, well flowing behaviour.
# """)
expander.write("""
     Drilling activities in the oil and gas industry are being reported on a daily basis, yet the analysis of this unstructured text data is very challenging
These Daily Drilling Progress Reports (DPR) are being created by Rig crews and the main challenges in the data are:
 high frequency of technical symbols, mistyping/abbreviation of technical terms, high inconsistency in vocabulary, structure, and spelling
.
     
 """)
