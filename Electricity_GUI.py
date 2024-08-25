import tkinter
import pandas as pd
from ElectricityPrices_model import electricity_model_prediction

#Gui that uses the prediciton model, and makes an easy and accessible way to predict prices
class ElectricityGUI:
    def __init__(self):
        # Initialize the main window for the GUI
        self.window = tkinter.Tk()
        self.window.title("Electricity Prices")
        self.window.geometry("1200x1000")
        self.window.configure(bg="orange")
        self.setup_widgets()  # Call the function to create the GUI widgets
        tkinter.mainloop()

    def setup_widgets(self):
        self.title_label = tkinter.Label(self.window, text="Electricity Prices", font=("Roboto bold", 40), fg="white", bg="orange")
        self.title_label.grid(row=0, column=0, columnspan=3, pady=(40, 20))  # Place the header in the grid layout

        # Display model accuracy in the GUI
        # This will give users an accessible understanding of how accurate the model is
        self.accuracy_label = tkinter.Label(self.window,
                                            text=f"Model accuracy: {electricity_model_prediction.total_accuracy:.2f}%",
                                            font=("Roboto", 20), bg="orange")
        self.accuracy_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))

        self.entries = {}  # Dictionary to hold the entry widgets for user input
        input_fields = {  # Define the labels and instructions for each input field
            "Holiday?": "0 for No, 1 for Yes",
            "Day of Week": "a number between 1 and 7",
            "Week of Year": "a number between 1 and 52",
            "Day of Month": "a number between 1 and 31",
            "Month": "Enter a number between 1 and 12",
            "Year": "For example, 2024",
            "Time of Day": "a number between 0 and 47",
            "Wind Power (MW)": "in MW",
            "System Load EA": "Enter the system load EA in MW",
            "Temperature": "Enter the temperature in degrees Celsius",
            "Wind Speed": "Enter the wind speed in km/h",
            "CO2 (g/kWh)": "CO2 intensity in grams per kWh",
            "Actual Wind Power (MW)": "Enter the actual wind power in MW",
            "System Load EP2": "Enter the system load EP2 in MW",
            "Previous Price (AUD/MWh)": "Enter the price from the previous period"
        }



        # Loop through the input fields dictionary to create labels, entry fields, and instruction labels
        row_num = 2  # Start placing widgets from row 2
        for label_text, instructions in input_fields.items():
            # Create and place the label for each input field
            label = tkinter.Label(self.window, text=label_text, font=("Roboto", 12), bg="orange")
            label.grid(row=row_num, column=0, sticky="e", padx=(150, 10), pady=5)

            # placing the entry field for user input
            entry = tkinter.Entry(self.window, width=40)
            entry.grid(row=row_num, column=1, pady=5)

            # Create and place the instruction label for each input field
            instruction_label = tkinter.Label(self.window, text=instructions, font=("Roboto", 10), fg="grey", bg="orange")
            instruction_label.grid(row=row_num, column=2, sticky="w", padx=(10, 0), pady=5)

            # store the entry widget in the dictionary with its label as the key
            self.entries[label_text] = entry
            row_num += 1  # Move to the next row for the next set of widgets

        # Create and place the submit button in the GUI
        self.predict_button = tkinter.Button(self.window, text="Predict Price", padx=5, pady=5, command=self.run_prediction)
        self.predict_button.grid(row=row_num, column=0, columnspan=3, pady=20)

    def run_prediction(self):
        try:
            # Collect input data from the entry fields
            input_data = {
                "HolidayFlag": float(self.entries["Holiday?"].get().strip()),
                "DayOfWeek": float(self.entries["Day of Week"].get().strip()),
                "Day": float(self.entries["Day of Month"].get().strip()),
                "Month": float(self.entries["Month"].get().strip()),
                "PeriodOfDay": float(self.entries["Time of Day"].get().strip()),
                "ForecastWindProduction": float(self.entries["Wind Power (MW)"].get().strip()),
                "SystemLoadEA": float(self.entries["System Load EA"].get().strip()),
                "ORKTemperature": float(self.entries["Temperature"].get().strip()),
                "ORKWindspeed": float(self.entries["Wind Speed"].get().strip()),
                "CO2Intensity": float(self.entries["CO2 (g/kWh)"].get().strip()),
                "ActualWindProduction": float(self.entries["Actual Wind Power (MW)"].get().strip()),
                "SystemLoadEP2": float(self.entries["System Load EP2"].get().strip()),
                "SMPEP2": float(self.entries["Previous Price (AUD/MWh)"].get().strip())
            }


            input_df = pd.DataFrame([input_data])
            prediction = electricity_model_prediction.model.predict(input_df)[0]

            # Display the prediction result in the GUI
            result_label = tkinter.Label(self.window,text=f"Predicted Price: {prediction:.2f} AUD/MWh",font=("Roboto", 20), bg="orange")
            result_label.grid(row=self.predict_button.grid_info()['row'] + 1, column=0, columnspan=3, pady=(10, 0))


        # Handle any errors that occur during the prediction process
        # This was a useful implementation to help trouble shoot any issues that arise when inputting data
        except ValueError as e:
            error_label = tkinter.Label(self.window, text=f"Error: {e}", font=("Roboto bold", 20), fg="red", bg="orange")
            error_label.grid(row=self.predict_button.grid_info()['row'] + 1, column=0, columnspan=3, pady=(10, 0))

if __name__ == "__main__":
    ElectricityGUI()
