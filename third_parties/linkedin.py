import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock:bool = False):
    """"
    scrape information from linkedinprofiles
    """

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/tonijhanel/3a807e461302b876b941c3fa471dc493/raw/455754117042fdc501785e2121141bebd3ea809b/linkedindata-toni.json"
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ['SCRAPIN_API_KEY'],
            "linkedInUrl": linkedin_profile_url,
        }
        response = requests.get(
            api_endpoint,
            params=params,
            timeout=10,
        )
    data = response.json().get("person")
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["certifications"]
    }

    return data

if __name__=="__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/tonijwilliams/", mock=True
        )
    )