import tkinter
from RentalModel import *

class Rental_GUI:
    def __init__(self):

        #initialising tkinter window
        self.window = tkinter.Tk()
        self.window.title("New York Room Type Predictor")

        # creating frames for each non target attribute
        self.first_frame = tkinter.Frame()
        self.second_frame = tkinter.Frame()
        self.third_frame = tkinter.Frame()
        self.fourth_frame = tkinter.Frame()
        self.fifth_frame = tkinter.Frame()
        self.sixth_frame = tkinter.Frame()
        self.seventh_frame = tkinter.Frame()
        self.eighth_frame = tkinter.Frame()
        self.ninth_frame = tkinter.Frame()
        self.button_frame = tkinter.Frame()
        self.output_frame = tkinter.Frame()


        # neighbourhood, uses option menu due to large of a variety of neighbourhoods
        self.neighbourhood_label = tkinter.Label(self.first_frame, text="Property Neighbourhood: ")
        self.click_neighbourhood_var = tkinter.StringVar()
        self.click_neighbourhood_var.set("Brooklyn")
        self.neighbourhood_input = tkinter.OptionMenu(self.first_frame, self.click_neighbourhood_var, "Brooklyn",
                                                      "Manhattan", "Queens", "Harlem", "Bedford-Stuyvesant")
        self.neighbourhood_label.pack(side="left")
        self.neighbourhood_input.pack(side="left")


        # lat
        self.latitude_label = tkinter.Label(self.second_frame, text="Enter Latitude: ")
        self.latitude_entry = tkinter.Entry(self.second_frame, bg="light blue", bd=2, width=10)
        self.latitude_label.pack(side="left")
        self.latitude_entry.pack(side="left")

        # long
        self.longitude_label = tkinter.Label(self.third_frame, text="Enter Longitude: ")
        self.longitude_entry = tkinter.Entry(self.third_frame, bg="light blue", bd=2, width=10)
        self.longitude_label.pack(side="left")
        self.longitude_entry.pack(side="left")

        # days occupied in 2019
        self.dOC_2019_label = tkinter.Label(self.fourth_frame, text="Enter Days Occupied in 2019: ")
        self.dOC_2019_entry = tkinter.Entry(self.fourth_frame, bg="light blue", bd=2, width=10)
        self.dOC_2019_label.pack(side="left")
        self.dOC_2019_entry.pack(side="left")

        # minimum nights to spend
        self.min_nights_label = tkinter.Label(self.fifth_frame, text="Minimum Nights to Stay: ")
        self.min_nights_entry = tkinter.Entry(self.fifth_frame, bg="light blue", bd=2, width=10)
        self.min_nights_label.pack(side="left")
        self.min_nights_entry.pack(side="left")

        # num of reviews
        self.num_rev_label = tkinter.Label(self.sixth_frame, text="Number of Reviews: ")
        self.num_rev_entry = tkinter.Entry(self.sixth_frame, bg="light blue", bd=2, width=10)
        self.num_rev_label.pack(side="left")
        self.num_rev_entry.pack(side="left")

        # monthly reviews
        self.monthly_rev_label = tkinter.Label(self.seventh_frame, text="number of Monthly Reviews: ")
        self.monthly_rev_entry = tkinter.Entry(self.seventh_frame, bg="light blue", bd=2, width=10)
        self.monthly_rev_label.pack(side="left")
        self.monthly_rev_entry.pack(side="left")

        # property availablity in 2020
        self.availability_label = tkinter.Label(self.eighth_frame, text="2020 Property Availability: ")
        self.availability_entry = tkinter.Entry(self.eighth_frame, bg="light blue", bd=2, width=10)
        self.availability_label.pack(side="left")
        self.availability_entry.pack(side="left")

        # price
        self.price_label = tkinter.Label(self.ninth_frame, text="Rental Price: ")
        self.price_entry = tkinter.Entry(self.ninth_frame, bg="light blue", bd=2, width=10)
        self.price_label.pack(side="left")
        self.price_entry.pack(side="left")

        # buttons
        self.predict_button = tkinter.Button(self.button_frame, text="Predict",
                                             command=self.predict_room)
        self.quit_button = tkinter.Button(self.button_frame, text="Quit",
                                          command=self.window.destroy)

        self.predict_button.pack(side="left")
        self.quit_button.pack(side="left")

        # output
        self.results = tkinter.Text(self.output_frame, bg="light blue", height=10, width=40)
        self.results.pack()

        # packing frames
        self.first_frame.pack()
        self.second_frame.pack()
        self.third_frame.pack()
        self.fourth_frame.pack()
        self.fifth_frame.pack()
        self.sixth_frame.pack()
        self.seventh_frame.pack()
        self.eighth_frame.pack()
        self.ninth_frame.pack()
        self.button_frame.pack()
        self.output_frame.pack()

        # main loop for window
        tkinter.mainloop()

    # predicting function, uses predictive alg
    def predict_room(self):
        self.results.delete(0.0, tkinter.END)

        # getting inputs
        neighbourhood = self.click_neighbourhood_var.get()
        lat = self.latitude_entry.get()
        long = self.longitude_entry.get()
        dOC_2019 = self.dOC_2019_entry.get()
        min_nights = self.min_nights_entry.get()
        num_rev = self.num_rev_entry.get()
        monthly_rev = self.monthly_rev_entry.get()
        availability = self.availability_entry.get()
        price = self.price_entry.get()

        # dictionary of limited neighbourhoods
        neighbourhoods = {
            "Brooklyn": 0,
            "Manhattan": 1,
            "Queens": 2,
            "Harlem": 3,
            "Bedford-Stuyvesant": 4
        }

        # getting value from key
        neighbourhood = neighbourhoods[neighbourhood]

        # creating list for prediction
        property_info = (neighbourhood, lat, long, dOC_2019, min_nights, num_rev, monthly_rev,
                         availability, price)
        # putting user input into model
        room_type_prediction = best_model.predict([property_info])

        # 4 types 0 = house, 1 = hotel, 2= private, 3 = shared
        # converts numerical representation to actual room type
        if room_type_prediction == 0.0:
            room_type = "Entire house/apt"
        elif room_type_prediction == 1.0:
            room_type = "Hotel room"
        elif room_type_prediction == 2.0:
            room_type = "Private room"
        else:
            room_type = "Shared room"

        # outputs result
        self.results.insert("1.0", f"Room type is predicted to be {room_type}")

my_rental_GUI = Rental_GUI()