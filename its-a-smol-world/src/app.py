import time
import itertools
import threading
import sys
import argparse
from smol_mind import SmolMind, load_functions
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

def main():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="SmolMind CLI")
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-i', '--instruct', action='store_true', help='Enable instruct mode (disables continue mode)')
    args = parser.parse_args()

    print("loading SmolMind...")
    functions = load_functions("./src/functions.json")
    sm = SmolMind(functions, model_name=MODEL_NAME, debug=args.debug, instruct=args.instruct)
    if args.debug:
        print("Using model:", sm.model_name)
        print("Debug mode:", "Enabled" if args.debug else "Disabled")
        print("Instruct mode:", "Enabled" if args.instruct else "Disabled")
    print(bunny_ascii)
    print("Welcome to the Bunny B1! What do you need?")
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

        response = sm.get_function_call(user_input)

        # Stop the spinner
        stop_event.set()
        spinner_thread.join()
        sys.stdout.write(' \b')  # Erase the spinner
        
        print(response)

if __name__ == "__main__":
    main()
