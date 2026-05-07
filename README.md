# ChemFlow

#### A Hierarchical Neural Network for Multiscale Representation Learning in Chemical Mixtures



![ChemFlow Architecture](figures/ChemFlow.png)


## Code Running Conditions



ChemFlow is implemented using **PyTorch** and runs on **Ubuntu** with **NVIDIA GeForce RTX 4090** GPUs.  

The framework also relies on **PyTorch Geometric**.



### Required Python Libraries

Please ensure the following libraries are installed:



\- numpy  2.1.2

\- pandas  2.3.3

\- rdkit  2025.9.1

\- scikit-learn  1.7.2

\- ncps  1.0.1

\- torch  2.9.0+cu128

\- torch\_geometric  2.7.0



---



## Code Structure and Data Generation



In the **Concentration dependent** directory, we provide:



\- Detailed implementation of each module  

\- Data generation scripts  

\- Example usage based on the activity coefficient dataset

In the **Non-Concentration dependent** directory, we provide a Example of CombiSolv. 



These examples help illustrate data preprocessing procedures and the training workflow.



---



## Hyperparameters and Model Prediction



During training, we **did not** perform hyperparameter tuning on the validation set for each individual property before making predictions on the test set.



**Note:**  

In practical applications, tuning hyperparameters for each property may further improve predictive performance. However, we did not perform extensive tuning due to the long training time, and because ChemFlow already outperforms models reported in existing literature.



---


## Continuously Updated



We will continue to update datasets and models, and regularly check the correctness and completeness of the code.  

If you encounter any errors while running the code or have any questions, please contact:



📧 **fanjinming@zju.edu.cn**



