import os.path
import wandb
import pandas as pd
from tqdm import tqdm

api = wandb.Api(timeout=30)
runs = api.runs(path="danieljunior/Legal DeepAL")

filename = 'data/all_runs.csv'
run_df = None

for run_ in tqdm(runs):
    history = run_.scan_history()
    rows = [row for row in history]
    run_history = pd.DataFrame(rows)
    run_history['name'] = run_.name
    run_history['tag'] = run_.tags[0]
    run_config = pd.DataFrame([run_.config])
    run_config = pd.concat([run_config.drop(['train_config'], axis=1),
                            run_config['train_config'].apply(pd.Series)], axis=1)
    tmp = pd.concat([run_config.reindex(run_history.index, method='ffill'), run_history], axis=1)
    if run_df is None:
        run_df = tmp
    else:
        run_df = pd.concat([run_df, tmp], axis=0, ignore_index=True)

run_df.to_csv(filename, sep=";", index=False)
print("Finish!")
