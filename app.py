# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import torch.nn as nn
import torch
from kmodes.kprototypes import KPrototypes



app = Flask(__name__)

import torch
import torch.nn as nn
class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(45, 20),
      nn.ReLU(),
      nn.Linear(20, 15),
      nn.ReLU(),
      nn.Linear(15, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user

            formDict = request.form.to_dict()
            print(formDict)
            X = pd.DataFrame(formDict, index=range(0, 1))
            print(X.head())
            dic_to_replace = {"Age": {"0-17": 1, "18-25": 2, "26-35": 3, "36-45": 4, "46-50": 5, "51-55": 6, "55+": 7},
                              "Stay_In_Current_City_Years": {"4+": 5}}
            X.replace(dic_to_replace, inplace=True)
            X['Stay_In_Current_City_Years'] = X['Stay_In_Current_City_Years'].astype('int64')
            # only run below for clustering
            X['Marital_Status'] = X['Marital_Status'].astype('object')
            X['Occupation'] = X['Occupation'].astype('object')
            X['Product_Category_1'] = X['Product_Category_1'].astype('object')

            # Load clustering Model
            loaded_model = pickle.load(open('clusteringModel.sav', 'rb'))
            cluster_label = loaded_model.predict(X, categorical=[0, 2, 3, 5, 6])
            cluster_label = cluster_label[0]
            print("clusters_label", cluster_label)
            # Handling dummy variables
           # traindf = pd.read_csv("test.csv")
            columns = ['Age', 'Stay_In_Current_City_Years', 'Gender_M', 'Marital_Status_1',
                       'City_Category_B', 'City_Category_C', 'Occupation_1', 'Occupation_2',
                       'Occupation_3', 'Occupation_4', 'Occupation_5', 'Occupation_6',
                       'Occupation_7', 'Occupation_8', 'Occupation_9', 'Occupation_10',
                       'Occupation_11', 'Occupation_12', 'Occupation_13', 'Occupation_14',
                       'Occupation_15', 'Occupation_16', 'Occupation_17', 'Occupation_18',
                       'Occupation_19', 'Occupation_20', 'Product_Category_1_2',
                       'Product_Category_1_3', 'Product_Category_1_4', 'Product_Category_1_5',
                       'Product_Category_1_6', 'Product_Category_1_7', 'Product_Category_1_8',
                       'Product_Category_1_9', 'Product_Category_1_10',
                       'Product_Category_1_11', 'Product_Category_1_12',
                       'Product_Category_1_13', 'Product_Category_1_14',
                       'Product_Category_1_15', 'Product_Category_1_16',
                       'Product_Category_1_17', 'Product_Category_1_18',
                       'Product_Category_1_19', 'Product_Category_1_20']
            occupationDummies = []
            productDummies = []
            cityDummies = []
            for word in columns:
                if word.startswith("Occupation"):
                    occupationDummies.append(word)
                elif word.startswith("Product_Category_1"):
                    productDummies.append(word)
                elif word.startswith("City_Category"):
                    cityDummies.append(word)
                else:
                    pass
            # Creating empty dataframes and then assigning 1 to the respective dummies as per user Input.
            tempdf = pd.DataFrame(np.zeros([1, 45]), columns=columns)
            tempdf.head()

            tempdf['Age'] = X['Age'][0]
            print("Age is", X['Age'][0])
            tempdf['Stay_In_Current_City_Years'] = X['Stay_In_Current_City_Years'][0]

            # Handle Gender Dummy Variable
            if X['Gender'][0] == "M":
                tempdf['Gender_M'] =1

            # Handle Marital Status Dummy Variable
            if X['Marital_Status'][0] == "1":
                tempdf['Marital_Status_1'] = 1

            # Handle City Dummy Variable
            city = X['City_Category'][0]
            for i in cityDummies:
                val = i.split("City_Category_", 1)[1]
                print(val)
                if (val == city):
                    print(i)
                    tempdf[i] = 1
                    break

            # Handle Occupation Dummy variables
            occupation = city = X['Occupation'][0]
            for i in occupationDummies:
                val = i.split("Occupation_", 1)[1]
                print(val)
                if (val == occupation):
                    tempdf[i] = 1
                    break

            # Handle Product Type Dummy variables

            productType = X['Product_Category_1'][0]
            for i in productDummies:
                val = i.split("Product_Category_1_", 1)[1]
                print(val)
                if (val == productType):
                    tempdf[i] = 1
                    break

            # Convert above df to numpyArray
            X = tempdf.to_numpy()
            print("numpy x", X)

            # Load Prediction model as per the clusterNo.
            if (cluster_label == 0):
                model = torch.load('MLP_cluster0.pt')
            elif (cluster_label == 1):
                model = torch.load('MLP_cluster1.pt')
            else:
                model = torch.load('MLP_cluster2.pt')

            # Predict the Purchase value
            with torch.no_grad():
                predValue = model(torch.Tensor(X))
                predValue=predValue.numpy()[0][0]

            #predValue = model.predict(X)
            print("Predicted  purchased Value is : ", predValue)

            return render_template('results.html', prediction=round(predValue, 2))
            # showing the prediction results in a UI

        except Exception as e:
            print('The Exception message is: ', e)
            return Response('something is wrong.')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
