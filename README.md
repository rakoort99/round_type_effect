### Evaluating the Effect of Round Type on College Policy Debate Outcomes

This repository contains the code from my master's thesis, "Evaluating the Effect of Round Type on College Policy Debate Outcomes." Details are found in `paper.pdf`. For a brief overview of the contents of this repository, see below.

The process is divided across seven scripts. With few exceptions, intermediate outputs that might be of interest for other tasks are saved as Excel sheets for ease of use.

#### Dependencies
All code run on Python 3.11 with packages dependencies:
- pandas
- numpy
- pyro
- seaborn
- scipy
- trueskillthroughtime
- tqdm
- bs4
- openpyxl

#### Workflow
The 'data' folder contains four scripts.
1. `tourn_list_gen.py` takes a range of years as inputs. It is used to create an excel sheet with links to each college policy debate tournament in the given range.
2. `tabroom_scraper.py` takes that excel sheet of tournaments as an input and outputs two excel sheets. The first is a record of all debate rounds at those tournaments, and the other records information regarding each judge active at those events.
3. `round_records_fixer.py` reformats the excel sheet of round records into a cleaner, more readable form and removes entries irrelevant to my particular task. 
4. `gen_model_inputs.py` takes the human-readable excel sheet of round records and the excel sheet of judge information as inputs and transforms them into a form more amenable to the machine learning model I implement.

The base folder contains three scripts:
1. `prefs_model.py` takes the model inputs born from the previous script and outputs a file containing estimates of the ideological position of each team and judge observed. This requires CUDA- hardware to run with any reasonable efficiency. Editing the "main" function is also advised.
2. `process_model.py` takes the raw model outputs and converts them into a more human readable form. It outputs two excel files: one with learned ideological information for teams and another for judges.
3. `skill_eval.py` takes the readable files derived from the model and outputs two excel sheets. The first contains each team and a summary of their skill. The second contains round records annotated with aff, neg, and judge ideologies and estimates of aff and neg skill at the time of each round.
