Trial project on Winning Arguments (Tan et al 2016) that replicates the pair prediction task using the GPT-4o-mini API.

Original paper and data:
@inproceedings{tan+etal:16a, 
    author = {Chenhao Tan and Vlad Niculae and Cristian Danescu-Niculescu-Mizil and Lillian Lee}, 
    title = {Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions}, 
    year = {2016}, 
    booktitle = {Proceedings of WWW} 
}

## Files and Directories

### Experiment write-up
- `Winning_Arguments.pdf`: Detailed write-up of the experiment.

### Scripts
- `scripts/GPTAPI.py`: Runs the "basic" and "explain-then-predict" setups (see PDF for details).
- `scripts/GPTAPI2.py`: Runs the "explain-then-predict-2" setup.

### Model Responses
- `output_data/explain_then_predict_together.jsonl`: Responses from 500 runs of the "explain-then-predict" setup.
- `output_data/explain_then_predict_separate.jsonl`: Responses from 500 runs of the "explain-then-predict-2" setup.