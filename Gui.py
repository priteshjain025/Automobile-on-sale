import tkinter
from PIL import ImageTk,Image

window = tkinter.Tk()
window.title("Welcome to Automobile on sale app")
#window.geometry('1500x1300')
window.geometry('1800x800')

#msg = tkinter.Message(window,text = "PLEASE ENTER YOUR INPUTS ").pack()  
#image2 =Image.open(r".img\zbEW4tX.jpg")
image2 =Image.open(r"C:\Users\Ad\Desktop\bros f proj\Automobiles_Capstone_Project-master\ss\zbEW4tX.jpg")
image1 = ImageTk.PhotoImage(image2)
w = tkinter.Label(window, image=image1)
w.pack(side='top', fill='both', expand='yes')

l1=tkinter.Label(window, text = " ENTER CAR DETAILS HERE ", fg='red',bg = "dodger blue",font=("Helvetica", 45))
l1.place(x=400, y=50)
#l1.config(width=200)


v1 = tkinter.StringVar(window)
tkinter.Label(window, text = "vehicle Type",bg="light blue").place(x=470, y=450)
e1=tkinter.OptionMenu(window, v1,'coupe','suv','kleinwagen','limousine','cabrio','bus','kombi','andere')
e1.config(width=14)
e1.place(x=550, y=450)

tkinter.Label(window, text = "year Of Registration ",bg="light blue").place(x=680, y=450)
e2=tkinter.Entry(window,bd=3)
e2.insert(0, "")
e2.place(x=800, y=450)


v2 = tkinter.StringVar(window)
tkinter.Label(window, text = "gear Box",bg="light blue").place(x=1023, y=450)
#e3=tkinter.Entry(window,bd=3)
e3 = tkinter.OptionMenu(window, v2, 'automatik','manuell')
e3.config(width=14)
e3.place(x=1080, y=450)

tkinter.Label(window, text = "powerPS",bg="light blue").place(x=490, y=500)
e4=tkinter.Entry(window,bd=3)
e4.place(x=550, y=500)

tkinter.Label(window, text = "kilometer",bg="light blue").place(x=735, y=500)
e5=tkinter.Entry(window,bd=3)
e5.place(x=800, y=500)

v3 = tkinter.StringVar(window)
tkinter.Label(window, text = "month Of Registration",bg="light blue").place(x=950, y=500)
e6= tkinter.OptionMenu(window, v3, "0", "1", "2",'3','4','5','6','7','8','9','10','11')
e6.config(width=14)
e6.place(x=1080, y=500)

v4 = tkinter.StringVar(window)
tkinter.Label(window, text = "fuel Type",bg="light blue").place(x=490, y=550)
e7=tkinter.OptionMenu(window, v4,'benzin','diesel')
e7.config(width=14)
e7.place(x=550, y=550)

v5 = tkinter.StringVar(window)
tkinter.Label(window, text = "brand",bg="light blue").place(x=755, y=550)
e8=tkinter.OptionMenu(window, v5,'volkswagen','audi','jeep','skoda','bmw','peugeot','ford','mazda','nissan',
'renault','mercedes_benz','opel','seat','citroen','honda','fiat','mini','smart','hyundai','sonstige_autos','alfa_romeo',
'subaru','volvo','mitsubishi','kia','suzuki','lancia','porsche','toyota','chevrolet','dacia','daihatsu','trabant','saab',
'chrysler','jaguar','daewoo','rover','land_rover','lada')
e8.config(width=14)
e8.place(x=800, y=550)

v6 = tkinter.StringVar(window)
tkinter.Label(window, text = "not Repaired Damage",bg="light blue").place(x=955, y=550)
e9=tkinter.OptionMenu(window, v6,'yes','no')
e9.config(width=14)
e9.place(x=1080, y=550)

tkinter.Label(window, text = "Sold in days",bg="light blue").place(x=475, y=600)
e10=tkinter.Entry(window,bd=3)
e10.place(x=550, y=600)

import pandas as pd
X = pd.read_csv(r'C:\Users\Ad\Desktop\bros f proj\Automobiles_Capstone_Project-master\X_dummy.csv')
#X = pd.read_csv(r'.data\X_dummy.csv')
import joblib
filename1 = r'C:\Users\Ad\Desktop\bros f proj\Automobiles_Capstone_Project-master\persis model\finalized_model.sav'
#filename1 = r'.model\finalized_model.sav'
rf1 = joblib.load(filename1)



import numpy
def input_to_one_hot(user_input):
    # initialize the target vector with zero values
    enc_input = numpy.zeros(64)
    # set the numerical input as they are
    enc_input[0] = user_input['yearOfRegistration']
    enc_input[1] = user_input['powerPs']
    enc_input[2] = user_input['monthOfRegistration']
    enc_input[3] = user_input['SoldInDays']

    
    ##################### Mark #########################
    
    # get the array of marks categories

    # redefine the the user inout to match the column name
    redefinded_user_input1 = 'vehicleType_'+user_input['vehicleType']
    # search for the index in columns name list 
    vehicle_column_index = X.columns.tolist().index(redefinded_user_input1)
    #print(mark_column_index)
    # fullfill the found index with 1
    enc_input[vehicle_column_index] = 1
    
    
    ##################### Fuel Type ####################
    # get the array of fuel type

    # redefine the the user inout to match the column name
    redefinded_user_input2 = 'fuelType_'+user_input['fuelType']
    # search for the index in columns name list 
    fuelType_column_index = X.columns.tolist().index(redefinded_user_input2)
    # fullfill the found index with 1
    enc_input[fuelType_column_index] = 1
    

    redefinded_user_input3 = 'gearbox_'+user_input['gearbox']
    gearbox_column_index = X.columns.tolist().index(redefinded_user_input3)
    enc_input[gearbox_column_index] = 1
    

    redefinded_user_input4 = 'brand_'+user_input['brand']
    brand_column_index = X.columns.tolist().index(redefinded_user_input4)
    enc_input[brand_column_index] = 1
    

    redefinded_user_input5 = 'notRepairedDamage_'+user_input['notRepairedDamage']
    notRepairedDamage_column_index = X.columns.tolist().index(redefinded_user_input5)
    enc_input[notRepairedDamage_column_index] = 1    
    
    return enc_input



def showop():
    user_input = {'yearOfRegistration':e2.get(),'powerPs':e4.get(),'monthOfRegistration':v3.get(),'SoldInDays':e10.get(),
    'vehicleType':v1.get(),'kilometer':e5.get(),'gearbox':v2.get(),'fuelType':v4.get(),'brand':v5.get(),'notRepairedDamage':v6.get()}  
    ab = input_to_one_hot(user_input)
    price_pred = rf1.predict([ab])
    return price_pred


def myAlert():
    response = tkinter.messagebox.askquestion("submittion", "Do you want to submit?")
    if response == "yes":
        tkinter.messagebox.showinfo("Price of the car is",str(showop()))     
    else:
        tkinter.Label(window, text = "Not submitted").pack()
                   
button_widget1 = tkinter.Button(window,text="submit",bg="red",command=myAlert,height = 1, width = 7,font=("Helvetica", 25))
button_widget1.place(x=800,y=650)
window.mainloop()






