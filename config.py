import os
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# Model parameters
PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}
# watsonx credentials
CREDENTIALS = {
    "url": os.getenv("WATSONX_URL"),
    "project_id": os.getenv("WATSONX_PROJECT_ID"),
    "apikey": os.getenv("WATSONX_APIKEY")
}
# Model IDs
LLAMA3_MODEL_ID = "meta-llama/llama-3-2-90b-vision-instruct"
GRANITE_MODEL_ID = "ibm/granite-3-2b-instruct"
#MIXTRAL_MODEL_ID = "mistralai/mistral-large"