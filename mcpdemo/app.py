#https://github.com/mcptutorial/mcp-use
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat():
    """Run a chat with MCP agent's built-in conversationmemory using the Groq API."""
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary
    config_file = "browser_mcp.json"
    print("Initializing chat.....")

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(config_file)

    # Create LLM
    llm = ChatGroq(model="qwen-qwq-32b")
    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=15,memory_enabled=True)
    print("Agent initialized")
    print("Type 'exit' or 'quit' to end the conversation")
    print("type 'clear' to clear the conversation history")
    print("================================================")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            elif user_input.lower() == "clear":
                agent.memory.clear()
                print("Conversation history cleared")
                continue
            print("\n Assistant:", end="",flush=True)    

            try:
                response = await agent.run(user_input)
                print(response, end="",flush=True)
            except Exception as e:
                print(f"Error: {e}")
                continue
    finally:
        if client and client.sessions:
            await client.close_all_sessions()             
        print("\n\nExiting...")         

if __name__ == "__main__":
    asyncio.run(run_memory_chat())