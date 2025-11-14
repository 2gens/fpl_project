Predicting the best FPL players each week using Data Science 

Categorie : Data Analysis and Visualization -- Sports Analytics Tool 


Fantasy Premier League (FPL) is an online football game where participants build a team 
of real Premier League players. Each player earns points each week based on their real
life performance, for example by scoring goals, making assists, or keeping a clean sheet. 
The goal of my project is to use data science to identify which players are likely to score 
the most points in the next gameweek, while also considering their price. The idea is to 
find the best “value” players, those who bring many points for a reasonable cost. 

The dataset will come from public FPL and football data sources, such as the official FPL 
API and Kaggle. It includes player statistics (minutes played, goals, assists, form, price), 
team performance, and opponent difficulty. Using this data, I plan to predict which players 
will perform best in the following week. 

The project will be implemented in Python, mainly using pandas and NumPy for data 
preparation and cleaning, and matplotlib and seaborn for data visualization. I will begin 
with an Exploratory Data Analysis (EDA) to understand the relationships between 
variables, such as how player form or opponent difficulty affects performance. 
Then, I will use machine learning models such as Linear Regression and Random Forests 
to predict future player points. The models will be evaluated using metrics like Mean 
Absolute Error (MAE) and R² score. Cross-validation will also be used to check how well 
the model generalizes to new data.

The main challenge will be that football outcomes are uncertain. For example, a player 
might get injured or rotated. However, with enough historical data and careful model 
tuning, I expect to make useful and realistic predictions that can help FPL managers make 
smarter weekly decisions. 

This project combines sports and data science in a practical and engaging way, while 
applying the key concepts learned in the course.