# LOGIC
LOGIC: Unifying Complete and Incomplete Multi-View Clustering through An Information-Theoretic Generative Model


## Requirements

- Python 3.9.20
- numpy 1.25.2
- torch 1.11.0+cu113
- torchvision 0.12.0+cu113
- torchaudio 0.11.0
- scikit-learn 1.2.2
- munkres 1.1.4
- matplotlib
## Tests
To test the model on provided datasets, e.g., Yale, run:

```
python test_Yale.py
```

## Visualization of Data Recovery. 
After running the test program, the recovered facial data, e.g., Yale, will be saved to the `recovery_pic` folder.