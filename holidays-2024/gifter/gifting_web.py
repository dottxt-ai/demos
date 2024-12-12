from flask import Flask, render_template, request
from pydantic import BaseModel, Field
from gifting import Gift, generate_gift_ideas, setup_model, MODEL_STRING
import logging
from logging.handlers import RotatingFileHandler
import time

app = Flask(__name__)
app.static_folder = 'static'

# Set up logging
handler = RotatingFileHandler('flask_app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = time.time()
        user_input = request.form['person_description']
        enable_search = request.form.get('enable_search') == 'on'
        exa_api_key = request.form.get('exa_api_key')
        max_ideas = int(request.form.get('max_ideas', 5))
        min_ideas = int(request.form.get('min_ideas', 0))
        client_ip = request.remote_addr

        # If max_ideas is smaller than min_ideas, throw an error
        if max_ideas < min_ideas:
            return render_template(
                'error.html', 
                error="max_ideas must be greater than min_ideas"
            )
        
        app.logger.info(f"Received request from {client_ip} with input length: {len(user_input)}")
        app.logger.debug(f"Search enabled: {enable_search}")
        
        try:
            # Log before gift generation
            app.logger.info("Starting gift idea generation...")
            ideas = generate_gift_ideas(
                user_input, 
                max_ideas=max_ideas,
                min_ideas=min_ideas
            )
            app.logger.info(ideas)
            
            # app.logger.info(f"Gift ideas generated. Number of ideas: {len(ideas.gift_ideas)}")
            
            # Only perform individual gift searches if enabled
            search_results = {}
            if enable_search:
                app.logger.info("Starting individual gift searches...")
                for gift in ideas.gift_ideas:
                    app.logger.debug(f"Searching for gift: {gift.name}")
                    try:
                        results = gift.search(api_key=exa_api_key).results

                        # Sometime there are blank titles, so we need to add them
                        for result in results:
                            if not result.title:
                                result.title = result.highlights[0]

                        search_results[gift.name] = results
                        app.logger.debug(f"Found {len(results)} results for {gift.name}")
                    except Exception as e:
                        app.logger.error(f"Error searching for gift {gift.name}: {str(e)}")
                        search_results[gift.name] = []
            
            processing_time = time.time() - start_time
            app.logger.info(f"Request processed successfully in {processing_time:.2f} seconds")
            
            # Log template rendering
            app.logger.info("Rendering template with results...")
            try:
                rendered = render_template('results.html', 
                                           person_description=user_input,
                                           ideas=ideas, 
                                           search_results=search_results,
                                           model_string=MODEL_STRING)
                app.logger.info("Template rendered successfully")
                return rendered
            except Exception as e:
                app.logger.error(f"Error rendering template: {str(e)}", exc_info=True)
                raise
            
        except Exception as e:
            app.logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return render_template('error.html', error=str(e))
    
    return render_template('index.html', model_string=MODEL_STRING)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=6969,
        debug=True
    ) 