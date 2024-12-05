import pandas as pd
file_path=r"updated_dataset.csv"
# file_path = r"D:\archive\HandInfo.csv" 
df = pd.read_csv(file_path)

# Add the 'jewelry preferences' column based on the 'skinColor' column
# def determine_jewelry_preference(skin_color):
#     if skin_color == "fair":
#         return "diamond"
#     elif skin_color == "medium":
#         return "platinum"
#     elif skin_color == "dark":
#         return "gold"
#     return None  # Default if skinColor doesn't match any condition

# df['jewelry_preferences'] = df['skinColor'].apply(determine_jewelry_preference)
# columns_to_remove = ['id', 'accessories', 'nailPolish', 'irregularities']
# df = df.drop(columns=columns_to_remove, errors='ignore')
# Save the updated DataFrame back to a CSV file
# output_file_path = 'updated_dataset.csv'
# df.to_csv(output_file_path, index=False)


# print("Updated CSV file has been saved as", output_file_path)
print(df['aspectOfHand'].value_counts().sum())
print(df.shape[0])
# hamre aspect of hand aur no.of rows baraber hain tou ab hum further ml training  karskte hain 