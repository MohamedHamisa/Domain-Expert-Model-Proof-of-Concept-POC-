#EVALUATION
#SETUP
!pip install ipywidgets==7.0.0 --quiet
!pip install --upgrade sagemaker datasets --quiet

#setup the AWS services
import sagemaker, boto3, json
from sagemaker.session import Session

sagemaker_session = Session()  
aws_Role = sagemaker_session.get_caller_identity_arn()
aws_Region = boto3.Session().region_name
Sess = sagemaker.Session()
print(aws_Role)
print(aws_Region)
print(Sess)

(model_id, model_version,) = ("meta-textgeneration-llama-2-7b","2.*",) #VERSION

from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id, model_version=model_version, instance_type="ml.g5.2xlarge") #VIEW QUOTES , CREATE AN INSTANCE
Predictor = model.deploy()

def print_response(payLoad, Response):
    print(payLoad["inputs"])
    print(f"> {Response[0]['generation']}")
    print("\n==========================\n")
    
    

payLoad = {
    "inputs": "A second important aspect of ubiquitous computing environments is",
    "parameters": {
        "max_new_tokens": 64,
        "top_p": 0.8,
        "temperature": 0.7,
        "return_full_text": False,
    },
}
try:
    Response = predictor.predict(payload, custom_attributes="accept_eula=true")
    print_response(payLoad, Response)
except Exception as e:
    print(e)
    


# Code
# Delete SageMaker endpoints and the attached resources
Predictor.delete_model()
Predictor.delete_endpoint()