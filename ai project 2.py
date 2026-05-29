import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load and Train the Model
df = pd.read_csv('kc_house_data.csv')
features = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built']
X = df[features]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print("--- AI Model  ---")

# 2. Get User Input
def get_user_input():
    print("\n enter the details of the house:")
    try:
        bedrooms = float(input("Number of Bedrooms: "))
        bathrooms = float(input("Number of Bathrooms: "))
        sqft = float(input("Living Area (sqft): "))
        grade = float(input("Grade (1-13): "))
        year = float(input("Year Built: "))
        
        # Predict
        input_data = pd.DataFrame([[bedrooms, bathrooms, sqft, grade, year]], columns=features)
        price = model.predict(input_data)[0]
        
        #prevent negative price
        price= max(0,price)
        print(f"\n✨ The estimated price for this house is: ${price:,.2f} ✨")
    except ValueError:
        print("Oops! Please enter valid numbers.")

# Run the input loop
while True:
    get_user_input()
    if input("\nPredict another? (y/n): ").lower() != 'y':
        break
