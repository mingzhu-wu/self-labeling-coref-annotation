# Improving the Performance of Coreference Resolvers Across Datasets by Leveraging Target Entities

This repository contains code for self-labeling method which uses Named Entities from target data to automatically add coreference annotations and improve coreference resolver performance.

## Install requirements

**Setup a Python virtual environment**

```
virtualenv venv-my --python=python3.6 or python3 -m venv venv-my 
source venv-my/bin/activate
```

**Install the requirements:**
```
pip install -r requirements.txt
```

## Running the experiments
**1. Start the Stanford CoreNLP server**
```
nohup java -mx16g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 30000 &
```

**2. Run self-labeling process, the parameter should be a json file consistent with the dataset above:**

  ```
  python3 annotate.py manual.json
  ```

**3. Split the annoated data into training subset**

   ```python3 split_data.py path/to/annotated/data```
 
Now the data is ready for training coreference resolution models.

**4. Train e2e-coref model on the split data**

Please refer to [e2e-coref](https://github.com/kentonl/e2e-coref) for detailed instructions about training and evaluating the model.

**5. Pronoun evaluation**
```
python3 scorer.py key_file prediction_file pro 
```
Please refer to [CoVal](https://github.com/ns-moosavi/coval) for more details about pronoun evaluation.


