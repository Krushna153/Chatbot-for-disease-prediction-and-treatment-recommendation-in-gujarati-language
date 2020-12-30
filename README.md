# Chatbot-for-disease-prediction-and-treatment-recommendation-in-gujarati-language


A Chatbot predicts the probability of having a disease from the given symptoms in Gujarati Language and also provides an additional information to the user like, it's causes, treatments, Home Remedies and Description of Disease.


This chatbot in health care has the potential to provide patients with access to immediate medical information and recommend diagnoses at the first sign of illness.
In rural areas, people are not that much literate and can't afford doctor's consultation. Also in Gujarat, many people are more familiar with Gujarati language rather than English or any other language. Therefore, a vernacular chatbot is created which is beneficial for them to acquire knowledge in very Gujarati language.\
The real benefit of this chatbot is to provide **સલાહ** (advice) and **માહતી** (information) for a healthy life.


### Flow of the project

**Step 1** : Take input from the user. (Either Symptoms to know disease or enter disease name to know details of it).\
**Step 2** : Tokenize the sentence into words.\
**Step 3** : If first word is ‘રોગ’, then for the other words find the disease name from the disease list and print its details like, reasons, treatments , home remedies etc.\
**Step 4** : If first word is not ‘રોગ’, then apply stemming on each word of the sentence.\
**Step 5** : Load POS dataset and apply stemming on each word of POS tag\
**Step 6** : Make dictionary of stem word as a key and its tag as value\
**Step 7** : Get the POS tag of each word from that dictionary.\
**Step 8** : Collect all nouns from that sentence and put them in one list called symptoms list.(Because we have considered all symptoms as nouns).\
**Step 9** : Fetch symptoms from symptoms list and get their maximum matching symptom name from the dataset.\
**Step 10** : Train the RandomForest model on a training data set.\
**Step 11** : Define an empty array S of length of no. of symptoms and initialize it with zero.\
**Step 12** : Put one in an array where the symptom matches with the input symptom.\
**Step 13** : Predict disease based on this array.\
**Step 14** : If symptoms are related to more than one disease, then the bot will ask other related symptoms.\
**Step 15** : Take input from the user yes/no/stop. (yes - if the symptom is related to his illness, no - if the symptom is not related to his illness and stop - if he doesn't want to do more clarification).\
**Step 16** : Based on these inputs update array S.\
**Step 17** : Predict the final disease.\
**Step 18** : Take input from the user yes/no - whether he/she wants to get details of the
disease or not.\
**Step 19** : If input is yes : Display all details of the disease. Else : Display "Thank You".\



The user can enter statement like: **મને માથુ દુખે છે** or **માથુ દુખવુ** or the user can also enter two or more symptoms at a time like **મને માથુ દુખે છે, પેટ માં
દુખે છે.**
Once the user enters the symptoms, then our bot will display all the diseases related to those symptoms.
So our bot asks the user about the other symptoms based on the given symptoms and based on the users’ answers, it will predict a disease along with the probability
of having a disease.
