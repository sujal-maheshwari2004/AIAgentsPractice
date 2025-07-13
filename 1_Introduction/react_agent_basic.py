from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from datetime import datetime

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_tokens=1000,
)

search_tool = TavilySearchResults(
    search_Depth="basic",
)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format."""
    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools=[search_tool, get_system_time]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True,
)

agent.invoke("What is the date today? and will it rain in Paris tomorrow?")

#print(llm.invoke("What is the capital of France?").content)