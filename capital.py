import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

credentials = Credentials(
    url=os.getenv("WATSONX_URL"),  # Fixed: Add 'api.' subdomain for Sydney CP4D
    api_key="wUk0dKlVyQzJMofK048tJgPyIZgw7s23wzOfiiqV8i-x",  # Regenerate if expired (see below)
    username="IBMid-697000DW58",  # Your IBMid
    #instance_id="openshift",  # CP4D platform
    version='5.3'  # Your CP4D version
)

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

model = ModelInference(
    model_id='mistralai/mixtral-8x7b-instruct-v01',
    params=params,
    credentials=credentials,
    project_id='daedbad3-4c0e-4754-8185-f33d1e5ab330'
)

text = """
Only reply with the answer. What is the capital of Canada?
"""
print(model.generate(text)['results'][0]['generated_text'])