import json
import os
import logging
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import functools
import re

class UserInputAnalysis(BaseModel):
    identified_tasks: List[str] = Field(description="List of identified tasks based on user input")
    extracted_info: Dict[str, Any] = Field(description="Dictionary of extracted parameters or details")
    additional_info_needed: List[str] = Field(description="List of additional information needed from the user")
    action_plan: List[Dict[str, str]] = Field(description="List of actions to be taken, each with a task name and prompt")
    user_intent: str = Field(description="Brief description of what the user is trying to accomplish")
    confidence: float = Field(description="Confidence score of the analysis, between 0 and 1")

class VisualConcept(BaseModel):
    composition: str = Field(description="Composition of the image: general appearence, foreground, background")
    key_elements: str = Field(description="Description of the key elements that should be included in the image")
    colors: str = Field(description="Color palette chosen")
    mood: str = Field(description="Overall mood and style")
    prompt: str = Field(description="A summary of composition, key elements, colors and mood. To be used as a prompt for image generation")

class Text2ImageInput(BaseModel):
    prompt: str = Field(description="The text prompt for image generation")

class Image2TextInput(BaseModel):
    image_path: str = Field(description="The path to the input image file")

class Image2ImageInput(BaseModel):
    image_path: str = Field(description="The path to the input image file")
    prompt: str = Field(description="The text prompt for image transformation")


def log_and_handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            return f"An error occurred: {str(e)}"
    return wrapper

class ArtMuse:
    def __init__(self, memory_manager, image_processor):
        self.logger = self.setup_logger()
        self.memory_manager = memory_manager
        self.image_processor = image_processor

        self.chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            #model="Mixtral-8x7b-32768",
            api_key=os.environ.get("GROQ_API_KEY", "")
        )
        self.parser = PydanticOutputParser(pydantic_object=UserInputAnalysis)
        self.parser_visual_concept = PydanticOutputParser(pydantic_object=VisualConcept)
        self.parser_text2image = PydanticOutputParser(pydantic_object=Text2ImageInput)
        self.parser_image2text = PydanticOutputParser(pydantic_object=Image2TextInput)
        self.parser_image2image = PydanticOutputParser(pydantic_object=Image2ImageInput)

        self.task_handlers = {
            "generate visual concept": self.visual_concept_generator,
            "extract style and information from image": self.extract_info_from_image,
            "transform image": self.transform_image,
            "generate image from text": self.generate_image_from_text,
            "general query": self.general_query,
        }

    def setup_logger(self):
        logger = logging.getLogger('ArtMuse')
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('artmuse.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @log_and_handle_exceptions
    def query_llm(self, prompt: str) -> str:
        response = self.chat.invoke(prompt)
        return response.content

    @log_and_handle_exceptions
    def analyze_user_input(self, user_input: str, relevant_memory: str) -> UserInputAnalysis:
        prompt_template = PromptTemplate(
            input_variables=["user_input", "task_descriptions", "relevant_memory"],
            template="""Analyze the following user input for an AI art assistant:

    User Input: "{user_input}"

    Available Tasks (choose one or more that best fit the user's request):
    {task_descriptions}

    Relevant Conversation History:
    {relevant_memory}

    Identify tasks from the list above, extract information, determine additional info needed, and suggest an action plan.
    Make sure to only choose tasks from the provided list.

    {format_instructions}
    """
        )

        prompt = prompt_template.format(
            user_input=user_input,
            task_descriptions="\n".join([f"- {task}" for task in self.task_handlers.keys()]),
            relevant_memory=relevant_memory,
            format_instructions=self.parser.get_format_instructions()
        )

        response = self.query_llm(prompt)
        analysis = self.parser.parse(response)
        self.logger.debug(f"User input analysis: {analysis}")
        return analysis

    @log_and_handle_exceptions
    def process_user_input(self, user_input: str) -> str:
        relevant_memory = self.memory_manager.get_relevant_memory(user_input)
        analysis_result = self.analyze_user_input(user_input, relevant_memory)
        
        if isinstance(analysis_result, str):
            return f"Error in analysis: {analysis_result}"
        
        # Clarification loop
        max_clarifications = 3
        clarifications = 0
        while analysis_result.additional_info_needed and clarifications < max_clarifications:
            clarification_response = "I need some additional information:\n"
            clarification_response += "\n".join(analysis_result.additional_info_needed)
            print(clarification_response)
            
            clarification = input("Please provide more information (or type 'done' to proceed): ")
            if clarification.lower().strip() == 'done':
                break
            
            user_input += f"\nAdditional info: {clarification}"
            relevant_memory = self.memory_manager.get_relevant_memory(user_input)
            analysis_result = self.analyze_user_input(user_input, relevant_memory)
            clarifications += 1
        
        # Present the list of tasks
        task_list = "\n".join([f"- {task['task']}: {task['prompt']}" for task in analysis_result.action_plan])
        print(f"Based on our conversation, here are the tasks I'm planning to perform:\n{task_list}")
        
        # Ask for permission once
        permission = input("Do you give permission to proceed with these tasks? (yes/no): ")
        if permission.lower().strip() != 'yes':
            return "Tasks cancelled due to lack of user permission."
        
        response = ""
        previous_task_output = None
        for action in analysis_result.action_plan:
            task_name = action['task']
            if task_name not in self.task_handlers:
                self.logger.error(f"Unknown task: {task_name}")
                response += f"Error: Unknown task '{task_name}'\n"
                continue
            
            # Use the previous task's output as input if available
            if previous_task_output:
                action['prompt'] += f"\n{previous_task_output}"
            
            task_result = self.execute_task(task_name, action['prompt'])
            response += f"{task_name} result: {task_result}\n\n"
            
            # Store the current task's output for the next iteration
            previous_task_output = task_result
        
        return response

    def ask_permission(self, task_description: str) -> bool:
        print(f"The assistant would like to perform the following task: {task_description}")
        user_input = input("Do you give permission to proceed? (yes/no): ")
        return user_input.lower().strip() == 'yes'

    @log_and_handle_exceptions
    def execute_task(self, task_name: str, prompt: str) -> str:
        if task_name not in self.task_handlers:
            self.logger.error(f"Unknown task: {task_name}")
            raise ValueError(f"Unknown task: {task_name}")
        
        self.logger.debug(f"Executing task: {task_name} with prompt: {prompt}")
        
        result = self.task_handlers[task_name](prompt)
        self.logger.debug(f"Task result: {result}")
        return result

    @log_and_handle_exceptions
    def generate_image_from_text(self, description: str) -> str:
        prompt = f"""Based on the following description, generate a text prompt for image generation:
        Description: '{description}'

        Ensure your response is a single, valid JSON object.
        
        {self.parser_text2image.get_format_instructions()}"""
        
        response = self.query_llm(prompt)
        input_data = self.parser_text2image.parse(response)
        return self.image_processor.text2image(input_data.prompt)

    @log_and_handle_exceptions
    def extract_info_from_image(self, description: str) -> str:
        prompt = f"""Based on the following description, provide the path to the image file:
        Description: '{description}'

        Ensure your response is a single, valid JSON object.
        
        {self.parser_image2text.get_format_instructions()}"""
        
        response = self.query_llm(prompt)

        print("DESCRIPTION", description)
        print("RESPONSE", response)
        input_data = self.parser_image2text.parse(response)

        image_path = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', input_data.image_path)

        return self.image_processor.image2text(image_path)
    
    @log_and_handle_exceptions
    def transform_image(self, description: str) -> str:
        prompt = f"""Based on the following description, provide the path to the input image file and the transformation prompt:
        Description: '{description}'

        Ensure your response is a single, valid JSON object.
        
        {self.parser_image2image.get_format_instructions()}"""
        
        response = self.query_llm(prompt)
        input_data = self.parser_image2image.parse(response)

        image_path = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', input_data.image_path)

        return self.image_processor.image2image(image_path, input_data.prompt)

    @log_and_handle_exceptions
    def visual_concept_generator(self, description: str) -> str:
        prompt = """Based on the following description: '{description}', 
        generate a detailed visual concept. Describe the composition, key elements, colors, mood and prompt.
        
        Ensure your response is a single, valid JSON object.
        
        {format_instructions}"""

        prompt = prompt.format(
            description=description,
            format_instructions=self.parser_visual_concept.get_format_instructions()
        )
        
        concept = self.query_llm(prompt)
        imgen_prompt = self.parser_visual_concept.parse(concept).prompt

        return imgen_prompt

    def general_query(self, prompt: str) -> str:
        return self.query_llm(prompt)

class ConversationManager:
    def __init__(self, art_muse: ArtMuse):
        self.art_muse = art_muse

    def manage_conversation(self, user_input: str) -> str:
        response = self.art_muse.process_user_input(user_input)
        self.art_muse.memory_manager.add_to_memory("user", user_input)
        self.art_muse.memory_manager.add_to_memory("assistant", response)
        
        while "Additional information needed:" in response:
            print(response)
            clarification = input("Please provide more information: ")
            response = self.art_muse.process_user_input(clarification)
            self.art_muse.memory_manager.add_to_memory("user", clarification)
            self.art_muse.memory_manager.add_to_memory("assistant", response)
        
        return response