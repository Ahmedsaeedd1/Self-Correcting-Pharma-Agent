import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from core.config import verify_environment

def main():
    parser = argparse.ArgumentParser(description="Clinical Trial Navigator Agent")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="The query you want to run through the agent system."
    )
    args = parser.parse_args()

    # Load Env Vars
    load_dotenv()
    
    try:
        verify_environment()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # Standardizing prompt input
    if args.query:
        user_query = args.query
    else:
        print("Please enter a query for the Clinical Trial Navigator.")
        print("Example: 'Who is the current FDA Commissioner in 2026?'")
        print("Type 'exit' to quit.\n")
        user_query = input("Query: ").strip()

    if not user_query or user_query.lower() == 'exit':
        return

    # Import the compiled graph AFTER env vars are loaded to ensure tools initialize well
    from agent.graph import agent_app

    print("\n" + "=" * 50)
    print("Executing Request...")
    print("=" * 50)

    # Set up initial state with the single user message
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }

    try:
        # Stream the nodes to show progress tracking in CLI naturally
        for output in agent_app.stream(initial_state):
            for key, value in output.items():
                print(f"\n[LOG] Node '{key}' completed execution.")
                # Print the AI's final response dynamically if available
                if "messages" in value and isinstance(value["messages"][-1], type(HumanMessage(content=""))).__name__ != 'HumanMessage':
                     # Just checking if the last message generated inside a node is an AIMessage to print
                     # Using simplistic check to avoid circular imports of AIMessage
                     print(f"[RESULT]: {value['messages'][-1].content}")

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

    print("\n" + "=" * 50)
    print("Complete.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
