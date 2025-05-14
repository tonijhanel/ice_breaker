from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import os

information = """
Maya Angelou born Marguerite Annie Johnson; April 4, 1928 – May 28, 2014) was an American memoirist, poet, and civil rights activist. She published seven autobiographies, three books of essays, several books of poetry, and is credited with a list of plays, movies, and television shows spanning over 50 years. She received dozens of awards and more than 50 honorary degrees.[3] Angelou's series of seven autobiographies focus on her childhood and early adult experiences. The first, I Know Why the Caged Bird Sings (1969), tells of her life up to the age of 17 and brought her international recognition and acclaim.

She became a poet and writer after a string of odd jobs during her young adulthood. These included fry cook, sex worker, nightclub performer, Porgy and Bess cast member, Southern Christian Leadership Conference coordinator, and correspondent in Egypt and Ghana during the decolonization of Africa. Angelou was also an actress, writer, director, and producer of plays, movies, and public television programs. In 1982, she was named the first Reynolds Professor of American Studies at Wake Forest University in Winston-Salem, North Carolina. Angelou was active in the Civil Rights Movement and worked with Martin Luther King Jr. and Malcolm X. Beginning in the 1990s, she made approximately 80 appearances a year on the lecture circuit, something she continued into her eighties. In 1993, Angelou recited her poem "On the Pulse of Morning" (1993) at the first inauguration of Bill Clinton, making her the first poet to make an inaugural recitation since Robert Frost at the inauguration of John F. Kennedy in 1961.

With the publication of I Know Why the Caged Bird Sings, Angelou publicly discussed aspects of her personal life. She was respected as a spokesperson for Black people and women, and her works have been considered a defense of Black culture. Her works are widely used in schools and universities worldwide, although attempts have been made to ban her books from some U.S. libraries. Angelou's most celebrated works have been labeled as autobiographical fiction, but many critics consider them to be autobiographies. She made a deliberate attempt to challenge the common structure of the autobiography by critiquing, changing, and expanding the genre. Her books center on themes that include racism, identity, family, and travel.
"""

if __name__ == "__main__":
    load_dotenv()
    print("Hello Langchaing")
    # print(os.environ['OPENAI_API_KEY'])

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. Return in an ordered list
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #llm = ChatOllama(model="llama3.2")
    llm = ChatOllama(model="mistral")
    chain = summary_prompt_template | llm | StrOutputParser()

    print(information)
    res = chain.invoke(input={"information": information})
    print("****RESPONSE****")
    print(res)
