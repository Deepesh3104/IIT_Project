from flask import (
    Flask,
    render_template,
    request,
    redirect,
    flash,
    session,
    jsonify,
    url_for,
)
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.objectid import ObjectId
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
import bcrypt
import pymongo
import warnings
import os


app = Flask(__name__)

# Load the individual models
with open("gb_model.pkl", "rb") as file1:
    gb_model = pickle.load(file1)

with open("rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)
import secrets

# Generate a random secret key of 32 bytes (256 bits)
app.secret_key = os.urandom(24)

# Mongo DB connnection Strings
connection_string = "mongodb+srv://Deepesh2104:Deepesh2228@cluster0.7mgj9te.mongodb.net/deepesh?retryWrites=true&w=majority"
# client = MongoClient(connection_string)
# db = client.get_database('Deepesh2104')  # Database name
# users_collection = db['users']

try:
    # Connect to MongoDB Atlas
    client = pymongo.MongoClient(connection_string)

    # Access your database (replace 'your-database' with the actual name)
    db = client["deepesh"]

    # Perform your database operations here...
    # For example, you can retrieve a collection and fetch some data:
    collection = db["user"]  # Replace 'your-collection' with the actual collection name

    # Print the fetched data

except pymongo.errors.ConnectionFailure as e:
    print("Failed to connect to MongoDB Atlas. Error:", e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    marital_status = request.form.get("MaritalStatus")
    hours_week = request.form.get("Hours_Week")
    gender = request.form.get("Gender")
    occupation = request.form.get("Occupation")
    customer_id = request.form.get("CustomerID")
    age = request.form.get("Age")
    months_as_customer = request.form.get("months_as_customer")
    policy_deductable = request.form.get("policy_deductable")
    policy_annual_premium = request.form.get("policy_annual_premium")
    insured_relationship = request.form.get("insured_relationship")
    capital_gains = request.form.get("capital-gains")
    capital_loss = request.form.get("capital-loss")
    incident_type = request.form.get("incident_type")
    collision_type = request.form.get("collision_type")
    incident_severity = request.form.get("incident_severity")
    authorities_contacted = request.form.get("authorities_contacted")
    number_of_vehicles_involved = request.form.get("number_of_vehicles_involved")
    property_damage = request.form.get("property_damage")
    bodily_injuries = request.form.get("bodily_injuries")
    witnesses = request.form.get("witnesses")
    police_report_available = request.form.get("police_report_available")
    total_claim_amount = request.form.get("total_claim_amount")
    injury_claim = request.form.get("injury_claim")
    property_claim = request.form.get("property_claim")
    vehicle_claim = request.form.get("vehicle_claim")
    state = request.form.get("State")
    vehicle_age = request.form.get("vehicle_age")
    work = request.form.get("work")

    # Check for missing form fields
    if any(
        value is None
        for value in [
            marital_status,
            hours_week,
            gender,
            occupation,
            customer_id,
            age,
            months_as_customer,
            policy_deductable,
            policy_annual_premium,
            insured_relationship,
            capital_gains,
            capital_loss,
            incident_type,
            collision_type,
            incident_severity,
            authorities_contacted,
            number_of_vehicles_involved,
            property_damage,
            bodily_injuries,
            witnesses,
            police_report_available,
            total_claim_amount,
            injury_claim,
            property_claim,
            vehicle_claim,
            state,
            vehicle_age,
            work,
        ]
    ):
        return render_template("prediction.html", prediction_text="Missing form fields")

    # Convert the features to the appropriate format
    marital_status = int(marital_status)
    hours_week = float(hours_week)
    gender = int(gender)
    occupation = int(occupation)
    customer_id = str(customer_id)
    age = float(age)
    months_as_customer = float(months_as_customer)
    policy_deductable = float(policy_deductable)
    policy_annual_premium = float(policy_annual_premium)
    insured_relationship = int(insured_relationship)
    capital_gains = float(capital_gains)
    capital_loss = float(capital_loss)
    incident_type = int(incident_type)
    collision_type = int(collision_type)
    incident_severity = int(incident_severity)
    authorities_contacted = int(authorities_contacted)
    number_of_vehicles_involved = float(number_of_vehicles_involved)
    property_damage = float(property_damage)
    bodily_injuries = float(bodily_injuries)
    witnesses = float(witnesses)
    police_report_available = int(police_report_available)
    total_claim_amount = float(total_claim_amount)
    injury_claim = float(injury_claim)
    property_claim = float(property_claim)
    vehicle_claim = float(vehicle_claim)
    state = int(state)
    vehicle_age = float(vehicle_age)
    work = int(work)

    # Perform predictions using individual models
    gb_prediction = gb_model.predict(
        [
            [
                marital_status,
                hours_week,
                gender,
                occupation,
                customer_id,
                age,
                months_as_customer,
                policy_deductable,
                policy_annual_premium,
                insured_relationship,
                capital_gains,
                capital_loss,
                incident_type,
                collision_type,
                incident_severity,
                authorities_contacted,
                number_of_vehicles_involved,
                property_damage,
                bodily_injuries,
                witnesses,
                police_report_available,
                total_claim_amount,
                injury_claim,
                property_claim,
                vehicle_claim,
                state,
                vehicle_age,
                work,
            ]
        ]
    )

    rf_prediction = rf_model.predict(
        [
            [
                marital_status,
                hours_week,
                gender,
                occupation,
                customer_id,
                age,
                months_as_customer,
                policy_deductable,
                policy_annual_premium,
                insured_relationship,
                capital_gains,
                capital_loss,
                incident_type,
                collision_type,
                incident_severity,
                authorities_contacted,
                number_of_vehicles_involved,
                property_damage,
                bodily_injuries,
                witnesses,
                police_report_available,
                total_claim_amount,
                injury_claim,
                property_claim,
                vehicle_claim,
                state,
                vehicle_age,
                work,
            ]
        ]
    )

    # Perform majority voting ensemble
    ensemble_prediction = []
    for i in range(len(gb_prediction)):
        if gb_prediction[i] + rf_prediction[i] >= 1:
            ensemble_prediction.append(1)
        else:
            ensemble_prediction.append(0)

    # Calculate ensemble accuracy using voting
    ensemble_accuracy = accuracy_score(ensemble_prediction, gb_prediction)

    # Process the ensemble prediction result and return it
    result = "Fraud" if ensemble_prediction[0] == 1 else "Not fraud"
    return render_template(
        "prediction.html",
        prediction_text="Prediction: {}".format(result),
        ensemble_accuracy=ensemble_accuracy,
    )


@app.route("/import_data", methods=["GET", "POST"])
def import_data():
    # Check if a file is uploaded
    if "file" not in request.files:
        return render_template("Import.html", prediction_text="No file uploaded.")

    file = request.files["file"]

    # Check if the file is empty
    if file.filename == "":
        return render_template("Import.html", prediction_text="No file selected.")

    # Read the CSV file
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return render_template("Import.html", prediction_text="Error reading the file.")

    # Perform predictions using individual models
    gb_prediction = gb_model.predict(data)
    rf_prediction = rf_model.predict(data)

    # Perform majority voting ensemble
    ensemble_prediction = []
    for i in range(len(gb_prediction)):
        if gb_prediction[i] + rf_prediction[i] >= 1:
            ensemble_prediction.append(1)
        else:
            ensemble_prediction.append(0)

    # Calculate ensemble accuracy using voting
    ensemble_accuracy = accuracy_score(ensemble_prediction, gb_prediction)

    # Process the ensemble prediction result and return it
    result = ["Fraud" if pred == 1 else "Not fraud" for pred in ensemble_prediction]
    return jsonify({"prediction_text": result, "ensemble_accuracy": ensemble_accuracy})


@app.route("/login1", methods=["GET", "POST"])
def login1():
    # print("hello")
    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]
        # Find the user with the provided email in the database
        user = collection.find_one({"email": email})

        if user:
            # Check if the provided password matches the stored hashed password
            if bcrypt.checkpw(password.encode("utf-8"), user["password"]):
                # Store the user's email in the session to keep them logged in
                session["user_id"] = str(user["_id"])
                return redirect(
                    url_for("dashboard")
                )  
            else:
                return render_template(
                    "login1.html", error="Invalid credentials"
                )  
        else:
            return render_template(
                "login1.html", error="Invalid credentials"
            )  # Replace this with an appropriate error message

        # user = collection.find_one({'email': email, 'password': password})
        # if user:
        #     session['user_id'] = str(user['_id'])
        #     return redirect(url_for('dashboard'))
        # else:
        #     return render_template('login1.html', error='Invalid credentials')

    return render_template("login1.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        firstname = request.form["Firstname"]
        lastname = request.form["Lastname"]
        email = request.form["Email"]
        password = request.form["Password"]
        # Hash the password before storing it

        existing_user = collection.find_one({"email": email})
        if existing_user:
            return render_template("login1.html", error="Email already exists")
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        user_data = {
            "firstname": firstname,
            "lastname": lastname,
            "email": email,
            "password": hashed_password,
        }
        collection.insert_one(user_data)

        return redirect(url_for("login1"))

    return render_template("login1.html")


@app.route("/dashboard")
def dashboard():
    if "user_id" in session:
        user_id = session["user_id"]
        user = collection.find_one({"_id": ObjectId(user_id)})
        return render_template("dashboard.html")
    else:
        return redirect(url_for("login1.html"))


@app.route("/services", methods=["GET", "POST"])
def services():
    return render_template("services.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("contactus.html")


@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")


@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/logout')
def logout():
    # Clear the user session data to log out.
    session.clear()
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)
