import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from training import predict_image,orientation_model,skin_tone_model
# Function to browse and predict the image
def browse_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Display the selected image
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Predict using the models
    prediction = predict_image(orientation_model, skin_tone_model, file_path)
    if "Error" in prediction:
        messagebox.showerror("Error", prediction["Error"])
    else:
        orientation_var.set(prediction["Orientation"])
        skin_tone_var.set(prediction["Skin Tone"])
        jewelry_var.set(prediction["Jewelry"])

# Create the main window
root = tk.Tk()
root.title("Iron Man Jewelry System")
root.geometry("600x800")
root.configure(bg="#1A1A1D")  # Dark background

# Header Label
header_label = tk.Label(
    root, 
    text="Jewelry Recommendation System", 
    font=("Helvetica", 20, "bold"), 
    bg="#1A1A1D", 
    fg="#FF3E00"  # Iron Man red
)
header_label.pack(pady=20)

# Frame for Image Display
img_frame = tk.Frame(root, bg="#1A1A1D")
img_frame.pack(pady=10)
img_label = tk.Label(img_frame, bg="#1A1A1D")
img_label.pack()

# Buttons Panel
button_panel = tk.Frame(root, bg="#1A1A1D")
button_panel.pack(pady=20)

browse_btn = tk.Button(
    button_panel, 
    text="Browse Image", 
    command=browse_and_predict, 
    font=("Helvetica", 14, "bold"), 
    bg="#FF3E00",  # Iron Man red
    fg="#FFFFFF",  # White text
    relief="flat", 
    padx=10, 
    pady=5
)
browse_btn.grid(row=0, column=0, padx=10)

# Results Section
result_frame = tk.Frame(root, bg="#1A1A1D")
result_frame.pack(pady=30)

orientation_var = tk.StringVar()
skin_tone_var = tk.StringVar()
jewelry_var = tk.StringVar()

orientation_label = tk.Label(
    result_frame, 
    text="Orientation:", 
    font=("Helvetica", 14, "bold"), 
    bg="#1A1A1D", 
    fg="#FFC107"  # Gold
)
orientation_label.grid(row=0, column=0, sticky="e", padx=10, pady=5)
orientation_value = tk.Label(
    result_frame, 
    textvariable=orientation_var, 
    font=("Helvetica", 14), 
    bg="#1A1A1D", 
    fg="#FFFFFF"
)
orientation_value.grid(row=0, column=1, sticky="w", padx=10, pady=5)

skin_tone_label = tk.Label(
    result_frame, 
    text="Skin Tone:", 
    font=("Helvetica", 14, "bold"), 
    bg="#1A1A1D", 
    fg="#FFC107"
)
skin_tone_label.grid(row=1, column=0, sticky="e", padx=10, pady=5)
skin_tone_value = tk.Label(
    result_frame, 
    textvariable=skin_tone_var, 
    font=("Helvetica", 14), 
    bg="#1A1A1D", 
    fg="#FFFFFF"
)
skin_tone_value.grid(row=1, column=1, sticky="w", padx=10, pady=5)

jewelry_label = tk.Label(
    result_frame, 
    text="Jewelry Recommendation:", 
    font=("Helvetica", 14, "bold"), 
    bg="#1A1A1D", 
    fg="#FFC107"
)
jewelry_label.grid(row=2, column=0, sticky="e", padx=10, pady=5)
jewelry_value = tk.Label(
    result_frame, 
    textvariable=jewelry_var, 
    font=("Helvetica", 14), 
    bg="#1A1A1D", 
    fg="#FFFFFF"
)
jewelry_value.grid(row=2, column=1, sticky="w", padx=10, pady=5)

# Add a footer
footer_label = tk.Label(
    root, 
    text="LIVE DEEN", 
    font=("Helvetica", 10, "italic"), 
    bg="#1A1A1D", 
    fg="#FF3E00"
)
footer_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
