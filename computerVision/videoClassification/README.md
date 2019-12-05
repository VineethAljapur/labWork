Program for classifying actions of fish.


### Installing requirements:

```bash
	pip install -r requirements.txt
```

### Running the script:


```bash
	python3 model.py
```

This will train the model with default metrics (refer the model.py) and generate results in directory containing saved model(s) and confusion matrix. To save the log information redirect the output to logFile.txt

By default all GPUs is used for the training.
If you want a part of GPUs, set devices using ```CUDA_VISIBLE_DEVICES=...```.

Please go through the list of arguments which parser uses and set the arguments accordingly.