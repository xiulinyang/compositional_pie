# Compositional_PIE
In this project, I propose to use the surprisal differential between the initial token and the last token of the V-N Potential Idiom Expressions to measure their compositionality.

# Instructions
- Create a virtual environment (python=3.8) and install all dependencies.
```
conda create -n pie python=3.8
pip install -r requirements.txt
```
- To replicate my experiments with the bigram model, please contact me and I will send you the KenML bigram model file. You can run ```experiment.py``` for the overall statistical test results and ```experiment_fine_tune.py``` for more fine-grained results.
- To replicate the experiments with GPT2, just run ``gpt2.py``.

--------------------------------------------
If you have questions, please contact me by email (xiulin.yang.compling@gmail.com)
