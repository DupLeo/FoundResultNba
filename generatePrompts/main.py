import DataNba
from DASPipelineQwen import DASPipelineQwen

if __name__ == "__main__":

    API_KEY = ""
    STUDENT_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    pipeline = DASPipelineQwen(openai_api_key=API_KEY,
                               student_model_id=STUDENT_ID)

    dataNba = DataNba.DataNba()
    prompts = dataNba.getPromptList()

    dataset = pipeline.run(prompts)
    print(f"Nombre d'exemples générés : {len(dataset)}")