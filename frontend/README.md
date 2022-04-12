# Frontend

This directory contains our frontend script.

### How to run

Create a conda environment with the packages listed in requirements and simply run it using python frontend.py.

This will then create an output like this: 
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app '2_front_end' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

Simply copy and past the initial url into your browser. Notice that this creates only a locally running webage and is not deployed. 

### Requirements

Below are the packages that you should install with the corresponding versions:

dash                      2.0.0              
dash-bootstrap-components 1.0.0              
jupyter-dash              0.3.0               
plotly                    5.3.1              


with Python 3.9.7