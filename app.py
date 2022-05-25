# from sys import audit
import streamlit as st
import pandas as pd
import pickle
from scipy.optimize import minimize
from PIL import Image


st.set_page_config(
    page_title= "Burnout risks",
    page_icon= "💔",
    layout="wide"
    
)

st.title("BurnApp 🔥🧠")
st.markdown("""This app will help you understand what is the mental health situation of your employees, and whether you should take preventive measures to avoid  potential burnouts of your employees. 
The employee should complete the following questionary, and recommendations will be based on those answers. """)

image = Image.open('burnout.jpeg')

st.image(image, caption="Employees'feelings", use_column_width='auto')

data = pd.read_csv("test data.csv")


### Section to fill in employee information

st.header("Employee information")
st.markdown("""The following questions will assess the mental health situation of your employees. 
The answers are based on an ordinal scale, which will be specified with the questions. 
They should be as objective as possible to increase the quality of the recommendations. """)


# user input


with st.expander("Work pace and quantity"):
    st.markdown("1: Never, 2: Sometimes, 3: Often, 4: Always")
    r1c1, r1c2, r2c1 = st.columns(3)
    r1c3, r2c2, r2c3 = st.columns(3)

with st.expander("Work-home interference"):
    st.markdown("1: Always, 2: Often, 3: Sometimes, 4: Never")
    r3c1, r3c2, r4c2 = st.columns(3)
    r3c3, r4c1, r5c1 = st.columns(3)
    r5c2, r4c3, r5c3 = st.columns(3)

with st.expander("The burnout questionnaire"):
    st.markdown("1: Never, 2: Sporadic, 3: Occasionally, 4: Regularly, 5: Often, 6: Very often, 7: Always")
    r6c1, r6c2, r7c3 = st.columns(3)
    r7c1, r7c2, r6c3 = st.columns(3)
    r8c1, r8c2, r8c3 = st.columns(3)


    

#Work pace and quantity

# q1 = r1c1.slider("You have too much work to do?", int(data["q0001"].min()), int(data["q0001"]).max())
#row 1
q1 = r1c1.slider("You have too much work to do?", 1, 4, 2)
q2 = r1c2.slider("You have to work hard to reach a deadline?", 1, 4, 2)
q3 = r1c3.slider("You have to work at speed?", 1, 4, 2)

#row 2
q5 = r2c1.slider("You have problems with the work pace?", 1, 4, 2)
q6 = r2c2.slider("You have problems with the workload?", 1, 4, 2)



#Work-home interference

q31 = r3c1.slider("You are irritable at home because your work is demanding?", 1, 4, 2)

#row 3
q32 = r3c2.slider("You have difficulties fulfilling your obligations at home?", 1, 4, 2)
q37 = r3c3.slider("Your working hours make it difficult to meet your obligations at home?", 1, 4, 2)
q39 = r4c1.slider("You have so much work to do that you do not have time for your hobbies?", 1, 4, 2)

#row 4
q41 = r4c2.slider("The demands of your work make it difficult to feel relaxed at home?", 1, 4, 2)
q38 = r4c3.slider("You have no energy through your work to do nice things with your partner, family, and or friends?", 1, 4, 2)
q42 = r5c1.slider("Your work takes time you prefer spending with partner, family or friends?", 1, 4, 2)

#row 5
q43 = r5c2.slider("After a pleasant working day/week, you would like to do more activities with partner, family and or friends?", 1, 4, 2)



#The burnout questionnaire

q44 = r6c1.slider("I feel mentally exhausted by my work", 1, 7, 4)
q45 = r6c2.slider("I feel empty at the end of a working day", 1, 7, 4)

#row 6
q46 = r6c3.slider("I feel tired when I get up and there is another working day for me", 1, 7, 4)
q49 = r7c1.slider("Working with people all day is a heavy burden for me", 1, 7, 4)
q53 = r7c2.slider("I feel I became more indifferent to people since I have this job", 1, 7, 4)

#row 7
q54 = r7c3.slider("I'm concerned that work dulls me emotionally", 1, 7, 4)
q55 = r8c1.slider("I feel frustrated by my job", 1, 7, 4)
q56 = r8c2.slider("I think I am too much committed to my work", 1, 7, 4)

#row 8
q59 = r8c3.slider("I feel at the end of my Latin", 1, 7, 4)


#Demographic data
# st.markdown("1 = <30, 2 = 30-40, 3 = 41-50, 4 = 51-60, 5 = >60")
st.write("##")
st.write("How many hours do you work per week on average? ")
r9c1, r9c2, r9c3 = st.columns(3)
# q99 = r9c1.slider("How many hours do you work per week on average?", ['<30', '30-40', '41-50', '51-60', '>60'], 3)
q99 = r9c1.slider("[1 = <30, 2 = 30-40, 3 = 41-50, 4 = 51-60, 5 = >60]", 1, 5, 3)


# with st.expander("Open to see how to use the App"):

#     st.write("This is more content")


# c = st.container()
# st.write('last')
# c.write('first')
# c.write('second')


### Predictive part

# loading model and functions needed for the predictive part
loaded_model = pickle.load(open("saved model.sav", "rb"))

def new_calculations(values):
    # print(type(Q1))
    intercept = loaded_model.params['const']
    workload = (values[6] * loaded_model.params['q0032'] + values[4] * loaded_model.params['q0006'])
    work_intensity = (values[2] * loaded_model.params['q0003'] + values[10] * loaded_model.params['q0041'])
    social_distan = (values[16] * loaded_model.params['q0054'] + values[15] * loaded_model.params['q0053'])
    mental_exhaust = (values[13] * loaded_model.params['q0044'] + values[19] * loaded_model.params['q0059'] + values[17] * loaded_model.params['q0055'])
    untweakable_var = (values[9] * loaded_model.params['q0039'] + values[7] * loaded_model.params['q0037'] + values[8] * loaded_model.params['q0038'] 
                        + values[1] * loaded_model.params['q0002'] + values[14] * loaded_model.params['q0045'] + values[12] * loaded_model.params['q0043'] 
                        + values[11] * loaded_model.params['q0042']+ values[20] * loaded_model.params['q0099']+ values[0] * loaded_model.params['q0001']
                        + values[14] * loaded_model.params['q0049'] + values[5] * loaded_model.params['q0031'] + values[18] * loaded_model.params['q0056']
                        + values[13] * loaded_model.params['q0046'] + values[3] * loaded_model.params['q0005'])
    return intercept, workload, work_intensity, social_distan, mental_exhaust, untweakable_var

def quick_diagnosis(df, input="file"):

  df["burnout"] = False

  for row in range(len(df)):

    intercept, workload, work_intensity, social_distan, mental_exhaust, untweakable_var = new_calculations([* df.values[row].tolist()])

    diagnosis = intercept + workload + work_intensity + social_distan + mental_exhaust + untweakable_var #add this concept in the following equations

    if diagnosis/7 >= 0.45: #or any other value we decide
      resp = 1 
    else:
      resp = 0
    
  if input == "user":
      rate = (diagnosis/7 * 100).round(2)
      return rate, resp

  df.loc[row, "burnout"] = resp
  return df

values = [[q1, q2, q3, q5, q6, q31, q32, q37, q38, q39, q41, q42, q43, q44, q45, q46, q49, q53, q54, q55, q56, q59, q99]]
user_input_df = pd.DataFrame(values, columns= ['q0001', 'q0002', 'q0003', 'q0005', 'q0006', 'q0031', 'q0032', 'q0037', 'q0038', 'q0039', 'q0041', 'q0042', 'q0043', 'q0044', 'q0045', 'q0046', 'q0049', 'q0053', 'q0054', 'q0055', 'q0056', 'q0059', 'q0099'])

rate, resp = quick_diagnosis(user_input_df, "user")


st.header("Prediction")

if resp == 1:
    st.write("Your employee is **over** the critical threshold of burn out")
else:
    st.write("Your employee is **under** the critical threshold of burn out")

st.write(f"Burn out **rate** of **{rate} %**")




### Preventive part

# recaltulating the values using the user's inputs
intercept, workload, work_intensity, social_distan, mental_exhaust, untweakable_var = new_calculations([* user_input_df.values[0].tolist()])

# backend operations
def objective_fun(t):
    t1, t2, t3, t4 = t[0], t[1], t[2], t[3]
    return t1 * workload + t2 * work_intensity + t3 * social_distan + t4 * mental_exhaust 


def inequality_constraint(t):
    t1, t2, t3, t4 = t[0], t[1], t[2], t[3]
    return (t1 * workload + t2 * work_intensity + t3 * social_distan + t4 * mental_exhaust) - (abs(intercept + untweakable_var)+1)


def inequality_constraint3(t):
    t1, t2, t3, t4 = t[0], t[1], t[2], t[3]
    return (t1 * workload + t2 * work_intensity + t3 * social_distan + t4 * mental_exhaust) - (abs(intercept + untweakable_var)+3)


result = minimize(objective_fun, [1,1,1,1], method = 'SLSQP', bounds = [(0.1,1.9), (0.1,1.9), (0.1,1.9), (0.1,1.9)], constraints = {'type': 'ineq', 'fun': inequality_constraint})
t1, t2, t3, t4 = result.x[0], result.x[1], result.x[2], result.x[3]
optimized_diagnosis = (intercept + untweakable_var + t1 * workload + t2 * work_intensity + t3 * social_distan + t4 * mental_exhaust/7 * 100).round(3)


result3 = minimize(objective_fun, [1,1,1,1], method = 'SLSQP', bounds = [(0.1,1.9), (0.1,1.9), (0.1,1.9), (0.1,1.9)], constraints = {'type': 'ineq', 'fun': inequality_constraint3})
t31, t32, t33, t34 = result3.x[0], result3.x[1], result3.x[2], result3.x[3]
threshold_diagnosis = (intercept + untweakable_var + t31 * workload + t32 * work_intensity + t33 * social_distan + t34 * mental_exhaust/7 * 100).round(3)

# frontend recommendations
def optimal_recommendation(t_values):

  output = ''
  for value in t_values:
    if value <= 1:
      output += '**Optimally:**\n'
      break

  if t_values[0] < 1:
    output += 'you should **decrease** the *workload* by **' + str(round((abs(1-t1) * 100), 1)) + '%**\n | '
  if t_values[1] < 1:
    output += 'you should **decrease** the *work intensity* by **' + str(round((abs(1-t2) * 100), 1)) + '%**\n | '
  if t_values[2] < 1:
    output += 'you should **decrease** *social_distan* by **' + str(round((abs(1-t3) * 100), 1)) + '%**\n | '
  if t_values[3] < 1:
    output += 'you should **decrease** *mental_exhaust* by **' + str(round((abs(1-t4) * 100), 1)) + '%**\n'
  
  if len(output) == 0:
    output = "No need to optimize anything"
    return output
  return output


def threshold_recommendation(t_values):

  output = ''
  for value in t_values:
    if value <= 1:
      output += '**Critically:**\n'  
      break

    if len(output) == 0:
        output += '**Careful**:\n'

  if t_values[0] < 1:
    output += 'you should decrease the **workload** by **' + str(round((abs(1-t31) * 100), 1)) + '%**\n | '
  else:
    output += "if you increase the **workload** by **" + str(round((abs(1-t31) * 100), 1)) + '%** you risk being over the critical threshold\n | ' 

  if t_values[1] < 1:
    output += 'you should decrease the **work intensity** by **' + str(round((abs(1-t32) * 100), 1)) + '%**\n | '
  else:
    output += "if you increase the **work intensity** by **" + str(round((abs(1-t32) * 100), 1)) + '%** you risk being over the critical threshold\n | '

  if t_values[2] < 1:
    output += 'you should decrease **social_distan** by **' + str(round((abs(1-t33) * 100), 1)) + '%**\n | '
  else:
    output += "if you increase the **social state** by"** + str(round((abs(1-t33) * 100), 1)) + '%** you risk being over the critical threshold\n | '

  if t_values[3] < 1:
    output += 'you should decrease **mental_exhaust** by **' + str(round((abs(1-t34) * 100), 1)) + '%**\n'
  else:
    output += "if you increase the **mental exhaust** by **" + str(round((abs(1-t34) * 100), 1)) + '%** you risk being over the critical threshold\n'
  
  
  return output


opt_recommendation = optimal_recommendation([t1, t2, t3, t4])
thresh_recommendation = threshold_recommendation([t31, t32, t33, t34])

# st.write([t1, t2, t3, t4])
# st.write([t31, t32, t33, t34])
st.header("Prevention")
st.write(opt_recommendation)
st.write(thresh_recommendation)
st.write("##")

### Rediagnosis only if t1, t2, t3, t4 < 1

def rediagnosis(tone, ttwo, tthree, tfour):
    tone += 1
    ttwo += 1
    tthree += 1
    tfour += 1

    newd = intercept + tone * workload + ttwo * work_intensity + tthree * social_distan + tfour * mental_exhaust + untweakable_var

    if newd/7 >= 0.45: #or any other value we decide
      resp = 1 
    else:
      resp = 0

    rate = (newd/7 * 100).round(2)

    return resp, rate

if t1 < 1 or  t2 < 1 or  t3 < 1 or t4 < 1:

    red1, red2 = st.columns(2)
    red3, red4 = st.columns(2)

    tone = red1.slider("How much do you want to change the workload variable? (e.g. 0.5 = 50%)", -0.9, 0.9, 0.1)
    ttwo = red2.slider("How much do you want to change the work intensity variable? (e.g. 0.5 = 50%)", -0.9, 0.9, 0.1)
    tthree = red3.slider("How much do you want to change the social distanciation variable? (e.g. 0.5 = 50%)", -0.9, 0.9, 0.1)
    tfour = red4.slider("How much do you want to change the mental exhaustion variable? (e.g. 0.5 = 50%)", -0.9, 0.9, 0.1)

    resp, rate = rediagnosis(tone, ttwo, tthree, tfour)

    if resp == 1:
        st.write("Your employee is **over** the critical threshold of burn out")
    else:
        st.write("Your employee is **under** the critical threshold of burn out")

    st.write(f"Burn out **rate** of **{rate} %**")


