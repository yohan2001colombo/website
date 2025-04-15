from flask import Flask,render_template,request
import pickle
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)

def prediction_urban(user_input):
     input_numind = [0, 1, 2]
     selected_num_indices = [0, 1, 3]
     filename = 'model/model_urban.pickle';
    # Load the saved objects
     with open(filename, 'rb') as f:
         loaded_data = pickle.load(f)

         urban_model = loaded_data['model']
         urban_scaler = loaded_data['scaler']

     # Extract numerical values
     numerical_input = [user_input[i] for i in input_numind]
     scaled_numerical = preprocess_user_input(numerical_input,selected_num_indices,urban_scaler).flatten()
     # If you have more numerical features
     for idx, scaled_val in zip(input_numind, scaled_numerical):
         user_input[idx] = float(scaled_val)
         
     pred_val = urban_model.predict([user_input])
     return pred_val

def prediction_nonurban(user_input):
     input_numind = [0, 1, 2]
     selected_num_indices = [0, 3, 6]
     filename = 'model/model_nonurban.pickle';
    # Load the saved objects
     with open(filename, 'rb') as f:
         loaded_data = pickle.load(f)

         nonurban_model = loaded_data['model']
         nonurban_scaler = loaded_data['scaler']

     # Extract numerical values
     numerical_input = [user_input[i] for i in input_numind]
     scaled_numerical = preprocess_user_input(numerical_input,selected_num_indices,nonurban_scaler).flatten()
     # If you have more numerical features
     for idx, scaled_val in zip(input_numind, scaled_numerical):
         user_input[idx] = float(scaled_val)
         
     pred_val = nonurban_model.predict([user_input])
     return pred_val

def preprocess_user_input(numerical_input, selected_num_indices, scaler):
   
    # Step 1: Reconstruct full feature vector (fill non-selected with 0)
    full_feature_vector = np.zeros(scaler.mean_.shape)  # Length = total original features
    full_feature_vector[selected_num_indices] = numerical_input  # Insert user values
    
    # Step 2: Apply scaling to FULL vector (to mimic training)
    scaled_full_vector = scaler.transform([full_feature_vector])[0]  # 2D -> 1D
    
    # Step 3: Extract only selected features for prediction
    scaled_selected_values = scaled_full_vector[selected_num_indices]
    
    return scaled_selected_values.reshape(1, -1)  # Reshape for model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor',methods=['POST','GET'])
def predictor():

    pred_val = "${:,.2f}".format(0)
    cuisine_profit_change = []
    cuisine = ''
    meal_profit_change = []
    meal = ''
    numeric_hints = []

    if request.method == 'POST':
         location = request.form['location']
         cuisine = request.form['cuisine']
         seats = request.form['seats']
         meal = request.form['meal_c']
         budget = request.form['market_bt']
         experience = request.form['chef_experience']
         qualityscore = request.form['service_quality_score']

         cuisine_list = ['french', 'indian','italian','japanese', 'mexican', 'american']
         meal_list = ['low','medium','high']

         feature_list = []

         print(location,cuisine,seats,meal,budget,experience,qualityscore)

         def add(input_value, input_list):
            arr = []
            for item in input_list:
                 if item == "mexican":  # Special case - don't add anything for "mexican"
                     continue
                 elif item == input_value:
                     arr.append(1)
                 else:
                     arr.append(0)
            return arr
        
         def meal_category(meal):
            if meal == "low":
                return 0
            elif meal == "medium":
                return 1
            else:
                 return 2
    
         if location == "urban":
             feature_list.append(int(seats))
             feature_list.append(int(budget))
             feature_list.append(int(experience))
             feature_list.extend(add(cuisine,cuisine_list))
             feature_list.append(int(meal_category(meal)))
             
             
             pred = prediction_urban(feature_list)
             output = float(pred[0])
             
             antilog = math.exp(output)  # Returns e^2.0 ≈ 7.389
             pred_val = "${:,.2f}".format(round(antilog,2))

             # choose the best cuisine 
             cuisine_profit_change = []
             
             for c in cuisine_list:
                 feature_list1 = []
                 if c == cuisine:
                    continue
                 else:
                    feature_list1.append(int(seats))
                    feature_list1.append(int(budget))
                    feature_list1.append(int(experience))
                    feature_list1.extend(add(c,cuisine_list))
                    feature_list1.append(int(meal_category(meal)))
             

                    test_pred = math.exp(float(prediction_urban(feature_list1)[0]))
                    percentage = (test_pred - antilog)/antilog*100
                    change = "{:,.2f}%".format(round(percentage,2))

                    cuisine_profit_change.append((c, change))
             
             # choose the meal category
             meal_profit_change = []

             for m in meal_list:
                 feature_list1 = []
                 if m == meal:
                    continue
                 else:
                    feature_list1.append(int(seats))
                    feature_list1.append(int(budget))
                    feature_list1.append(int(experience))
                    feature_list1.extend(add(cuisine,cuisine_list))
                    feature_list1.append(int(meal_category(m)))

                    test_pred = math.exp(float(prediction_urban(feature_list1)[0]))
                    percentage = (test_pred - antilog)/antilog*100
                    change = "{:,.2f}%".format(round(percentage,2))

                    meal_profit_change.append((m, change))

             numeric_hints = []
             
             if location == "rural":
                numeric_hints.append("Switching to a suburban location could potentially increase revenue.")
             if seats:
                 numeric_hints.append("Increasing seating capacity may lead to higher revenue.")
             if experience:
                numeric_hints.append("Hiring more experienced chefs could improve revenue.")
             if budget:
                 numeric_hints.append("Allocating more marketing budget might boost revenue.")
             if qualityscore:
                 numeric_hints.append("Enhancing service quality can positively impact revenue.")

 


         elif location == "suburban":
             feature_list.append(int(seats))
             feature_list.append(int(experience))
             feature_list.append(float(qualityscore))
             feature_list.extend(add(cuisine,cuisine_list))
             feature_list.append(int(meal_category(meal)))
             feature_list.append(1)


             pred = prediction_nonurban(feature_list)
             output = float(pred[0])
             
             antilog = math.exp(output)  # Returns e^2.0 ≈ 7.389
             pred_val = "${:,.2f}".format(round(antilog,2))

             # choose the best cuisine 
             cuisine_profit_change = []
             
             for c in cuisine_list:
                 feature_list1 = []
                 if c == cuisine:
                    continue
                 else:
                    feature_list1.append(int(seats))
                    feature_list1.append(int(experience))
                    feature_list1.append(float(qualityscore))
                    feature_list1.extend(add(c,cuisine_list))
                    feature_list1.append(int(meal_category(meal)))
                    feature_list1.append(1)

                    test_pred = math.exp(float(prediction_nonurban(feature_list1)[0]))
                    percentage = (test_pred - antilog)/antilog*100
                    change = "{:,.2f}%".format(round(percentage,2))

                    cuisine_profit_change.append((c, change))

            # choose the meal category
             meal_profit_change = []

             for m in meal_list:
                 feature_list1 = []
                 if m == meal:
                    continue
                 else:
                    feature_list1.append(int(seats))
                    feature_list1.append(int(experience))
                    feature_list1.append(float(qualityscore))
                    feature_list1.extend(add(cuisine,cuisine_list))
                    feature_list1.append(int(meal_category(m)))
                    feature_list1.append(1)

                    test_pred = math.exp(float(prediction_nonurban(feature_list1)[0]))
                    percentage = (test_pred - antilog)/antilog*100
                    change = "{:,.2f}%".format(round(percentage,2))

                    meal_profit_change.append((m, change))

             numeric_hints = []
             
             if location == "rural":
                numeric_hints.append("Switching to a suburban location could potentially increase revenue.")
             if seats:
                 numeric_hints.append("Increasing seating capacity may lead to higher revenue.")
             if experience:
                numeric_hints.append("Hiring more experienced chefs could improve revenue.")
             if budget:
                 numeric_hints.append("Allocating more marketing budget might boost revenue.")
             if qualityscore:
                 numeric_hints.append("Enhancing service quality can positively impact revenue.")

 
            

         else:
             feature_list.append(int(seats))
             feature_list.append(int(experience))
             feature_list.append(float(qualityscore))
             feature_list.extend(add(cuisine,cuisine_list))
             feature_list.append(int(meal_category(meal)))
             feature_list.append(0)

             pred = prediction_nonurban(feature_list)
             output = float(pred[0])
             
             antilog = math.exp(output)  # Returns e^2.0 ≈ 7.389
             pred_val = "${:,.2f}".format(round(antilog,2))

              # choose the best cuisine 
             cuisine_profit_change = []

             for c in cuisine_list:
                 feature_list1 = []
                 if c == cuisine:
                    continue
                 else:
                    feature_list1.append(int(seats))
                    feature_list1.append(int(experience))
                    feature_list1.append(float(qualityscore))
                    feature_list1.extend(add(c,cuisine_list))
                    feature_list1.append(int(meal_category(meal)))
                    feature_list1.append(0)

                    test_pred = math.exp(float(prediction_nonurban(feature_list1)[0]))
                    percentage = (test_pred - antilog)/antilog*100
                    change = "{:,.2f}%".format(round(percentage,2))

                    cuisine_profit_change.append((c, change))
             
             # choose the meal category
             meal_profit_change = []

             for m in meal_list:
                 feature_list1 = []
                 if m == meal:
                    continue
                 else:
                    feature_list1.append(int(seats))
                    feature_list1.append(int(experience))
                    feature_list1.append(float(qualityscore))
                    feature_list1.extend(add(cuisine,cuisine_list))
                    feature_list1.append(int(meal_category(m)))
                    feature_list1.append(0)

                    test_pred = math.exp(float(prediction_nonurban(feature_list1)[0]))
                    percentage = (test_pred - antilog)/antilog*100
                    change = "{:,.2f}%".format(round(percentage,2))

                    meal_profit_change.append((m, change))
             print(meal_profit_change)
    # Collect hints for numeric variables
             numeric_hints = []

             if location == "rural":
                numeric_hints.append("Switching to a suburban location could potentially increase revenue.")
             if seats:
                 numeric_hints.append("Increasing seating capacity may lead to higher revenue.")
             if experience:
                numeric_hints.append("Hiring more experienced chefs could improve revenue.")
             if budget:
                 numeric_hints.append("Allocating more marketing budget might boost revenue.")
             if qualityscore:
                 numeric_hints.append("Enhancing service quality can positively impact revenue.")



    return render_template('predictor.html', pred=pred_val,cuisine_change=cuisine_profit_change,meal_change=meal_profit_change,cuisine=cuisine,meal = meal,numeric_hints=numeric_hints)

if __name__=='__main__':
    app.run(debug=True)