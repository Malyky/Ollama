import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models.ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("TAVILY_API_KEY")

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import AgentExecutor, create_openai_tools_agent

import os

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERPER_API_KEY")

llm = OpenAI(openai_api_key=OPEN_AI_API_KEY)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
from langchain import hub
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages
tools = [Tool(name="Wikipedia", func=wikipedia, description="useful for when you need to search Wikipedia")]
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "Wer ist Bundestrainer?"})
print(result.get("output"))

from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
ddgo = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
matchChain = LLMMathChain.from_llm(llm)
mathTool = Tool.from_function(name="Calculator", func=matchChain, description="useful for when you need to do math")
ddgTool = Tool.from_function(name="DuckDuckGoSearch", func=ddgo, description="useful for when you need to search the web")
tools = [ddgTool, mathTool]
agent2 = create_openai_tools_agent(llm,
                                   tools,
                                   prompt
                                   )
agent_executor2 = AgentExecutor(agent=agent2, tools=tools, verbose=True)
result2 = agent_executor2.invoke({"input": "Wie alt ist der Bundestrainer? Gib mir die quadratwurzel davon"})
print(result2.get("output"))




# from langchain.chains.llm_math.base import LLMMathChain
# wikiTool = Tool.from_function(name="Wikipedia", func=wikipedia, description="useful for when you need to search Wikipedia")
# matchChain = LLMMathChain.from_llm(llm)
# mathTool = Tool.from_function(name="Calculator", func=matchChain, description="useful for when you need to do math")
# tools = [wikiTool, mathTool]
# agent2 = create_openai_tools_agent(llm,
#                                    tools,
#                                    prompt
#                                    )
# agent_executor2 = AgentExecutor(agent=agent2, tools=tools, verbose=True)
# result2 = agent_executor2.invoke({"input": "Wie alt ist der Bundestrainer? Gib mir die quadratwurzel davon"})
# print(result2.get("output"))


