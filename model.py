from langchain_ibm import WatsonxLLM
from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config import PARAMETERS, CREDENTIALS, LLAMA3_MODEL_ID, GRANITE_MODEL_ID #, MIXTRAL_MODEL_ID 

'''
# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")
'''
# Refine JSON output structure
class AIResponse(BaseModel):
    response: str = Field(description="Suggested response to the user")

json_parser = JsonOutputParser(pydantic_object=AIResponse)

# Function to initialize a model
def initialize_model(model_id):
    return ChatWatsonx(
        model_id=model_id,
        url="https://au-syd.ml.cloud.ibm.com",
        apikey=CREDENTIALS["apikey"],  # Pass as 'apikey' (required by ChatWatsonx)
        project_id=CREDENTIALS["project_id"],  # Fixed typo (removed extra 'k')
        params=PARAMETERS
    )

# Initialize models
llama3_llm = initialize_model(LLAMA3_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
#mixtral_llm = initialize_model(MIXTRAL_MODEL_ID)

# Get format instructions for JSON
format_instructions = json_parser.get_format_instructions()

# Prompt templates with JSON format instructions
llama3_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}

{format_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

granite_template = PromptTemplate(
    template="<|system|>{system_prompt}\n\n{format_instructions}\n<|user|>{user_prompt}\n<|assistant|>",
    input_variables=["system_prompt", "user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    result = chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    })
    return result

# Model-specific response functions
def llama3_response(system_prompt, user_prompt):
    return get_ai_response(llama3_llm, llama3_template, system_prompt, user_prompt)
def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_llm, granite_template, system_prompt, user_prompt)
#def mixtral_response(system_prompt, user_prompt):
    #return get_ai_response(mixtral_llm, mixtral_template, system_prompt, user_prompt)