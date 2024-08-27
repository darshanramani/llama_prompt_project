import streamlit as st 
from langchain.prompts import PromptTemplate 
# from langchain.embeddings import CTransformers
# from ctransformers import CTransformers
# from langchain_community.llms import CTransformers
from langchain_community.llms import CTransformers


# from langchain.llms import CTransformers
# from langchain_community.llms import CTransformers
# from langchain_community.llms import CTransformers

def getLLamaresponse(food_type,top_n,city_name):
    ### LLama model calling
    llm= CTransformers(model='model\\llama-2-7b-chat.ggmlv3.q8_0.bin',
                       model_type='llama',
                       config={'max_new_tokens':256,
                               'temperature':0.01})
    
    ## Prompt template

    template= """
        List out {city_name}'s {top_n} {food_type} famous food places, please list out properly line by line
        """

    prompt= PromptTemplate(input_variables=['city_name','food_type','top_n'],
                           template=template)


    ##generate the response from the LLama 2 model

    response = llm.invoke(prompt.format(city_name=city_name,
                      food_type = food_type,
                      top_n= top_n))

    print(response)
    return response

st.set_page_config(
    page_title='Generate Food Places',
    page_icon='😊',
    layout= 'centered',
    initial_sidebar_state='collapsed'
)

st.header('Generate Food Places 😋')

food_type= st.text_input('Enter the city name')

## creating two more columns

col1,col2=st.columns([5,5])
with col1:
    top_n=st.selectbox('Top N',('Top 3','Top 5','Top 10'), index=0)
with col2:
    city_name= st.selectbox('Food type',('All kinds of','South indian','North indian','City local'),index=0)



submit = st.button('Generate')

## Final response

if submit:
    st.write(getLLamaresponse(food_type,top_n,city_name))