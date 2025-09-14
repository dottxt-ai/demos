import time
import itertools
import threading
import sys
import argparse
import asyncio
from smol_mind import SmolMind
from constants import MODEL_NAME

# Thanks to @torymur for the bunny ascii art!
bunny_ascii = r"""
(\(\ 
 ( -.-)
 o_(")(")
"""

def spinner(stop_event):
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    while not stop_event.is_set():
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        sys.stdout.write('\b')
        time.sleep(0.1)

async def main():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="SmolMind MCP Client")
    parser.add_argument('server_path', help='Path to the MCP server script (.py or .js)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print("Loading SmolMind MCP client...")
    sm = SmolMind(args.server_path, model_name=MODEL_NAME, debug=args.debug)
    
    try:
        # Connect to the server
        tools = await sm.connect_to_server()
        if args.debug:
            print("Using model:", sm.model_name)
            print("Debug mode:", "Enabled" if args.debug else "Disabled")
            print(f"Available tools: {[tool.name for tool in tools]}")
        
        print(bunny_ascii)
        print("Welcome to the Bunny B1 MCP Client! What do you need?")
        
        while True:
            user_input = input("> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Create a shared event to stop the spinner
            stop_event = threading.Event()
            
            # Start the spinner in a separate thread
            spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
            spinner_thread.daemon = True
            spinner_thread.start()

            try:
                response = await sm.process_query(user_input)
            finally:
                # Stop the spinner
                stop_event.set()
                spinner_thread.join()
                sys.stdout.write(' \b')  # Erase the spinner
            
            print(response)
    finally:
        # Ensure we close the connection
        await sm.close()

if __name__ == "__main__":
    asyncio.run(main())