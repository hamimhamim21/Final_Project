from flask import Flask, render_template, redirect, jsonify, request
import scrape
from match_csvs import melb_avg, distance_crime
# from model.persist import load_model, load_model_NL, predict_value, predict_value_NL

import requests
from flask import Flask, redirect, url_for, request
import numpy as np
import pandas as pd


from joblib import dump, load
import xgboost


app = Flask(__name__)


#---------------------- LOAD MODELS   -----------------------------------
MODEL_PATH = "xgboost_best_model_2024.joblib"

def load_model():

      return load(MODEL_PATH)
      print("I loaded")


MODEL_PATH_NL = "final_model_no_landsize.joblib"

def load_model_NL():
      
      return load(MODEL_PATH_NL)

#-------------------- PREDICTION FUNCTIONS -----------------------------
def predict_value(X):

      model = load_model()
      prediction = model.predict(X)     
      print(prediction)            

      return prediction


def predict_value_NL(X):

      model = load_model_NL()
      prediction = model.predict(X)                 

      return prediction









# Home route ----------------------------------------------------------------------------------------
@app.route("/")
def home():
      #return index html from home route    
      return render_template("index_template.html", name="default")
# predict route ----------------------------------------------------------------------------------------






@app.route('/predict', methods=['POST', 'GET'])
def login():
    
      # when request POST domain listing is sent and updated
      if request.method == 'POST':
            # save value inputed in form
            domain_listing = request.form['nm']
            # run scraping function fro new domain listing url
            features = scrape.scrape_house_listing(domain_listing)
            print(features)

            # ------------------------------ PREDICTION ------------------------------------
            # check whether any features returned are from VIC and include all values
            if (features["cars"] != "Unknown" or
                  features == "error"):

                  # Check State only Victoria
                  if (features["state"] == 'VIC'):

                        # DOES NOT EQUAL CHECKS
                        if (features["landsize"] != "Unknown"):    

                              # ---------------- PROPERTY TYPE CONVERTED -----------
                              convert_type = features["ptype"][0]
                              print(f"the converted type {convert_type}")
                              # catch all types for apartment/unit/new apartment/flat
                              # and convert to numerical value
                              
                              # If House or Villa-------------------------------
                              if (convert_type == "H"):
                                    Type_h = 1
                              elif (convert_type == "V"):
                                    Type_h = 1
                              else:
                                    Type_h = 0

                              # If townhouse/unit or flat -----------------------                       
                              if (convert_type == "A"):
                                    Type_u = 1
                              elif (convert_type == "U"):
                                    Type_u = 1                  
                              elif (convert_type == "N"):
                                    Type_u = 1                     
                              elif (convert_type == "F"):
                                    Type_u = 1
                              else:
                                    Type_u = 0

                              # If Townhouse -----------------------------------
                              if (convert_type == "T"):
                                    Type_t = 1
                              else:
                                    Type_t = 0                   
                        
                              # -------------------- MELB AVG FOR PROPERTY TYPE FOR 2016-2018---------------------------
                              # read in CSV and to find the average values for property type           
                              type_lower = features["ptype"][0].lower()                                    
                              avg_rooms, avg_price, avg_bathroom, avg_car, avg_landsize = melb_avg(type_lower)
                              # append to features
                              features["melbourne_avg"].append(round(avg_rooms,2))
                              features["melbourne_avg"].append(round(avg_price,2))
                              features["melbourne_avg"].append(round(avg_bathroom,2))
                              features["melbourne_avg"].append(round(avg_car,2))
                              features["melbourne_avg"].append(round(avg_landsize,2))

                              # -------------------- CRIME AND DISTANCE FOR SUBURB AVERAGED FOR 2016-2018  ---------------------------
                              # read in csv file and find distance and crime based on suburb that is scraped                    
                                                
                              suburb_lower = features["suburb"].lower()
                              distance, crime = distance_crime(suburb_lower)
                              
                              features["suburb_distance_crime"].append(round(distance,2))
                              features["suburb_distance_crime"].append(round(crime,2)) 

                              # ------------------ PREDICTION VALUES SCALED BY MIN AND MAX VALUES --------------------------
                              # scale all values to predict
                                    # manual scaling min and max
                              Crime = (float(features["suburb_distance_crime"][1]) - 0) / (15485 - 0)                  
                              Distance = float(features["suburb_distance_crime"][0]) / (50 - 0)
                              Rooms = (float(features["bedrooms"]) - 1) / (12 - 1)
                              Bathrooms = (float(features["bathrooms"]) - 0 ) / (9 - 0)
                              Cars = (float(features["cars"]) - 0) / (18 - 0)
                              Month = ((float(features["month"]) - 1) - 0) / (11 - 0)
                              Year = (float(features["year"])-2016)/(2024 - 2016)                  
                              Landsize = (float(features["landsize"]) - 0) / (433014 - 0)
                              Type_h = float(Type_h)
                              Type_t = float(Type_t)                     
                              Type_u = float(Type_u)                

                              #================================ PREDICTION ==============================================
                              #==========================================================================================
                              #create pandas df to predict value with all features scaled
                              # #['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'Year', 'Month', 'Crime', 'Type_h', 'Type_t', 'Type_u']
                              X = pd.DataFrame([Rooms, Distance, Bathrooms, Cars, Landsize, Year, Month, Crime, Type_h, Type_t, Type_u], 
                                                      ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'Year', 'Month', 'Crime', 'Type_h', 'Type_t', 'Type_u']).T
                              print("PREDICTION VALUES ARE")
                              model = load("xgboost_best_model_2024.joblib")
                              print("Model loaded")
                              print(model)
                              print("Prediction")
                              print(model.predict(X))
                              # run predict function from persist
                              predict = round(model.predict(X))
                              print("I PREDICTED")
                              print(predict)
                              # format value predicted
                              # prediction_formated = f"{predict:,}"
                              # append values to features
                              features["prediction"].append(predict)
                              # features["predict_format"].append(prediction_formated)

                              #==========================================================================================

                              # return prediction if everything scraped and predicted correctly
                              return render_template('inner-page_prediction_template.html', features=features)                

                        elif (features["landsize"] == 'Unknown'):

                              # ---------------- PROPERTY TYPE CONVERTED -----------
                              convert_type = features["ptype"][0]
                              print(f"the converted type {convert_type}")
                              # catch all types for apartment/unit/new apartment/flat
                              # and convert to numerical value
                              
                              # If House or Villa-------------------------------
                              if (convert_type == "H"):
                                    Type_h = 1
                              elif (convert_type == "V"):
                                    Type_h = 1
                              else:
                                    Type_h = 0

                              # If townhouse/unit or flat -----------------------                       
                              if (convert_type == "A"):
                                    Type_u = 1
                              elif (convert_type == "U"):
                                    Type_u = 1                  
                              elif (convert_type == "N"):
                                    Type_u = 1                     
                              elif (convert_type == "F"):
                                    Type_u = 1
                              else:
                                    Type_u = 0

                              # If Townhouse -----------------------------------
                              if (convert_type == "T"):
                                    Type_t = 1
                              else:
                                    Type_t = 0                   
                        
                              # -------------------- MELB AVG FOR PROPERTY TYPE FOR 2016-2018---------------------------
                              # read in CSV and to find the average values for property type           
                              type_lower = 'u'                                  
                              avg_rooms, avg_price, avg_bathroom, avg_car, avg_landsize = melb_avg(type_lower)
                              # append to features
                              features["melbourne_avg"].append(round(avg_rooms,2))
                              features["melbourne_avg"].append(round(avg_price,2))
                              features["melbourne_avg"].append(round(avg_bathroom,2))
                              features["melbourne_avg"].append(round(avg_car,2))
                              features["melbourne_avg"].append(round(avg_landsize,2))

                              # -------------------- CRIME AND DISTANCE FOR SUBURB AVERAGED FOR 2016-2018  ---------------------------
                              # read in csv file and find distance and crime based on suburb that is scraped                    
                                                
                              suburb_lower = features["suburb"].lower()
                              distance, crime = distance_crime(suburb_lower)
                              
                              features["suburb_distance_crime"].append(round(distance,2))
                              features["suburb_distance_crime"].append(round(crime,2)) 

                              # ------------------ PREDICTION VALUES SCALED BY MIN AND MAX VALUES --------------------------
                              # scale all values to predict
                                    # manual scaling min and max
                              Crime = (float(features["suburb_distance_crime"][1]) - 0) / (15485 - 0)                  
                              Distance = float(features["suburb_distance_crime"][0]) / (50 - 0)
                              Rooms = (float(features["bedrooms"]) - 1) / (12 - 1)
                              Bathrooms = (float(features["bathrooms"]) - 0 ) / (9 - 0)
                              Cars = (float(features["cars"]) - 0) / (18 - 0)
                              Month = ((float(features["month"]) - 1) - 0) / (11 - 0)
                              Year = (float(features["year"])-2016)/(2024 - 2016)                
                              Type_h = float(Type_h)
                              Type_t = float(Type_t)                     
                              Type_u = float(Type_u)                

                              #================================ PREDICTION ==============================================
                              #==========================================================================================
                              #create pandas df to predict value with all features scaled
                              # #['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'Year', 'Month', 'Crime', 'Type_h', 'Type_t', 'Type_u']
                              X = pd.DataFrame([Rooms, Distance, Bathrooms, Cars, Year, Month, Crime, Type_h, Type_t, Type_u], 
                                                      ['Rooms', 'Distance', 'Bathroom', 'Car', 'Year', 'Month', 'Crime', 'Type_h', 'Type_t', 'Type_u']).T
                           
                              # run predict function from persist
                              predict = round(predict_value_NL(X)[0])
                              print(X)
                              # format value predicted
                              prediction_formated = f"{predict:,}"
                              # append values to features
                              features["prediction"].append(predict)
                              features["predict_format"].append(prediction_formated)

                              #==========================================================================================
                              return render_template('inner-page_prediction_NL_template.html', features=features)

                        else:
                              return render_template('inner-page_prediction_error_template.html', features=features)
                  else: 
                        # return error html if all features not scraped
                        return render_template('inner-page_prediction_error_template.html', features="Wrong State!")
            else:
                  # return error html if features is empty
                  return render_template('inner-page_prediction_error_template.html', features = "Invalid URL")



      else:
            # not really needed
            domain_listing = request.args.get('nm')
            return render_template('inner-page_predict_template.html', features="Enter Domain Listing URL")
 


if __name__ == "__main__":
    app.run(debug=True)
