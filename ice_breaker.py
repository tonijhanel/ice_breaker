from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from third_parties.linkedin import  scrape_linkedin_profile
import os

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello Langchain")
    # print(os.environ['OPENAI_API_KEY'])

    summary_template = """
        given the Linkedin {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. Return in an ordered list
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #llm = ChatOllama(model="llama3.2")
    #llm = ChatOllama(model="mistral")
    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/tonijwilliams/", mock=True
    )


    #print(information)
    res = chain.invoke(input={"information": linkedin_data})
    print("****RESPONSE****")
    print(res)
