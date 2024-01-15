import os
from typing import List, Union
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

search = DuckDuckGoSearchRun()

def duck_wrapper(input_text):
    # sites = ["site:medicinenet.com", "site:webmd.com", "site:mayoclinic.org"]
    # site_query = " OR ".join(sites)
    # search_results = search.run(f"{site_query} {input_text}")
    search_results = search.run(f"site:medicinenet.com {input_text}")
    return search_results

tools = [
    Tool (
        name="Search medicinenet",
        func=duck_wrapper,
        description="useful for when you need to answer medical related questions"
    )
]

# tools = [
#     Tool(
#         name="Search",
#         func=search.run,
#         description="useful for when you need answers for current events"
#     )
# ]

# Set up the base template
template = """Answer the following questions as best you can, but speaking as compasionate medical professional. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a compasionate medical professional when giving your final answer. If the condition is serious advise they speak to a doctor.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

#set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

memory=ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)


agent_executor.invoke("how can I treat cut wound?")