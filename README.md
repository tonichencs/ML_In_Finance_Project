# ML_In_Finance_Project

To run our code, you need to navigate through multiple notebooks, prepare the raw data and allow access to google drive due to intense requirement of memory.

## Compustat Preprocessing
You can run handle_compustat.py locally or compustat_data.ipynb for forward filling, or run compustat_data_updated.ipynb for interpolation filling.

## Earning Calls
Navigate to the text_embedding folder, run the preprocessing_earning_calls.py along with data_utilities.py. Then upload the resulted collection of parquet file to the Google Drive. Run the Textual_Embedding_generation.ipynb to regenerate textual embeddings we used for our experiment.

## Target Data
Run the handle_target.py locally or handle_target.ipynb notebook.

## Merging datasets
Run the Final_Data.ipynb, this will merge the earning calls, Compustat and Target Dataset according to datetime in the way that avoids lookahead bias.

## Line to all the processed and merged dataset
https://drive.google.com/drive/folders/1Q6B3UmaYNPtOmFOPgpPUoJfheKvJZmmj?usp=sharing

## Run experiment
Finally, run the Experiment.ipynb for experiments.

## Extension
The Folder Extension contains all the code needed for the extension part. First run the factor_process.ipynb and combine_Dataset.ipynb to get the return factors and financial factors, then run final_data.ipynb to get the final dataset using for training, and finally run DNN.ipynb to train the original model. Run gplearn_ml_factor.ipynb to generate genetic algorithm and machine learning factors, then run extra_DNN.ipynb to train the model with new features. CNN_LSTM.ipynb contains  dimension increasing process and model training.
